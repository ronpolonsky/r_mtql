#!/usr/bin/env python
"""Evaluate MTQLTransformerRealAgent checkpoints on a live DROID robot."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

from absl import app, flags
import jax
import numpy as np
from ml_collections import config_flags

from agents import agents
from expo_ft.env.env_client import EnvClientWrapper
from utils.flax_utils import restore_agent
from utils.mtql_droid import (
    DEFAULT_IMAGE_SIZE,
    DroidHistoryBuffer,
    OpenPINormalizer,
    convert_droid_observation,
)
from utils.mtql_orbax import (
    initialize_checkpoint_dir,
    restore_checkpoint as restore_orbax_checkpoint,
)


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "agent",
    "agents/mtql_transformer_real.py",
    "Real-agent config matching the training checkpoint.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_task",
    None,
    "EXPO DROID task config used by the rollout server.",
    lock_config=False,
)

flags.DEFINE_string(
    "norm_stats_path",
    None,
    "OpenPI norm-stats directory or exact norm_stats.json path.",
)
flags.DEFINE_string(
    "restore_path",
    None,
    "Native MTQL checkpoint directory or EXPO-style Orbax checkpoint directory.",
)
flags.DEFINE_integer(
    "restore_epoch",
    None,
    "Native MTQL checkpoint epoch or Orbax checkpoint step to restore.",
)
flags.DEFINE_enum(
    "checkpoint_format",
    "native",
    ("native", "orbax"),
    "MTQL checkpoint format. Native loads params_<epoch>.pkl; Orbax uses "
    "EXPO's initialize_checkpoint_dir manager.",
)
flags.DEFINE_integer("seed", 0, "Policy sampling seed.")
flags.DEFINE_integer(
    "image_size",
    DEFAULT_IMAGE_SIZE,
    "Square image size used when this checkpoint was trained.",
)
flags.DEFINE_integer(
    "hist_length",
    20,
    "MTQL history length used during training (matches m_real.py default).",
)
flags.DEFINE_integer(
    "hist_stride",
    7,
    "MTQL history stride used during training (matches m_real.py default).",
)
flags.DEFINE_integer(
    "action_chunk_size",
    25,
    "Action chunk size used during training (matches m_real.py default).",
)
flags.DEFINE_integer(
    "action_exec_horizon",
    0,
    "Actions executed before replanning; zero executes the complete chunk.",
)
flags.DEFINE_integer("num_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_bool(
    "wait_for_start",
    False,
    "Deprecated compatibility flag. Episodes start automatically after a one-second delay.",
)
flags.DEFINE_enum(
    "gripper_mode",
    "hold",
    ("policy", "hold"),
    "7D gripper execution mode: policy sends the checkpoint's seventh "
    "action to the gripper; hold ignores that action and executes arm-only "
    "motion without any gripper RPC.",
)
flags.DEFINE_bool(
    "inject_training_gripper_state",
    False,
    "Replace the gripper-width observation seen by the policy with the "
    "training q01 reference, while leaving the physical gripper unchanged.",
)
flags.DEFINE_enum(
    "cue_mode",
    "visual",
    ("visual", "language", "none"),
    "Conditioning used during training: draw target slots into the base image, "
    "pass the target as a language prompt ID, or use no task cue (Egg).",
)
flags.DEFINE_string("client_host", "localhost", "DROID rollout-server host.")
flags.DEFINE_integer("client_port", 8102, "DROID rollout-server port.")
flags.DEFINE_string(
    "video_dir",
    None,
    "Optional server-side directory for evaluation videos.",
)
flags.DEFINE_string(
    "results_dir",
    None,
    "Directory for evaluation results. Defaults to the restored experiment's "
    "evaluation/step_<restore_epoch> directory.",
)


def _add_batch_dimension(tree):
    return jax.tree_util.tree_map(
        lambda value: np.asarray(value)[None],
        tree,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_results_dir(restore_path: str, restore_epoch: int) -> Path:
    checkpoint_path = Path(restore_path)
    experiment_dir = (
        checkpoint_path.parent
        if checkpoint_path.name == "checkpoints"
        else checkpoint_path.parent
    )
    return experiment_dir / "evaluation" / f"step_{int(restore_epoch)}"


def _next_trial_dir(root: Path) -> Path:
    """Allocate the next non-overwriting trial_N directory under root."""
    root.mkdir(parents=True, exist_ok=True)
    trial_numbers = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("trial_"):
            continue
        suffix = child.name[len("trial_"):]
        if suffix.isdigit():
            trial_numbers.append(int(suffix))
    return root / f"trial_{max(trial_numbers, default=0) + 1}"


def _write_results(path: Path, results: dict) -> None:
    """Atomically write the current evaluation manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _update_result_summary(results: dict) -> None:
    episodes = results["episodes"]
    successes = sum(bool(item["success"]) for item in episodes)
    results["episodes_completed"] = len(episodes)
    results["successes"] = successes
    results["success_rate"] = (successes / len(episodes)) if episodes else 0.0
    by_target = {}
    for item in episodes:
        target = str(item["target_count"])
        bucket = by_target.setdefault(target, {"episodes": 0, "successes": 0})
        bucket["episodes"] += 1
        bucket["successes"] += int(bool(item["success"]))
    for bucket in by_target.values():
        bucket["success_rate"] = (
            bucket["successes"] / bucket["episodes"]
            if bucket["episodes"]
            else 0.0
        )
    results["by_target"] = by_target


def _target_count(observation) -> int:
    if "target_count" not in observation:
        raise KeyError(
            "Candy-scoop evaluation requires observation['target_count']; "
            "check that the candy_scoop task config is being used."
        )
    return int(np.asarray(observation["target_count"]).reshape(-1)[0])


def _task_name(task_config) -> str:
    return str(task_config.get("env_name", "")).lower()


def _inject_training_gripper_state(observation, gripper_position):
    """Return a policy-only observation with the training gripper width."""
    if gripper_position is None:
        return observation
    policy_observation = dict(observation)
    policy_observation["gripper_position"] = np.asarray(
        gripper_position,
        dtype=np.float32,
    )
    return policy_observation


def _build_example_batch(
    observation,
    history_buffer,
    action_chunk_size,
    cue_mode,
):
    target_count = (
        _target_count(observation)
        if cue_mode in ("visual", "language")
        else None
    )
    mtql_observation = convert_droid_observation(
        observation,
        history_buffer.normalizer,
        image_size=history_buffer.image_size,
        target_count=target_count if cue_mode == "visual" else None,
    )
    example_batch = {
        "observations": _add_batch_dimension(mtql_observation),
        "actions": np.zeros(
            (1, action_chunk_size, 7),
            dtype=np.float32,
        ),
    }
    if cue_mode == "language":
        example_batch["language_prompt_ids"] = np.asarray(
            [target_count], dtype=np.int32
        )
    history_observations = history_buffer.history_observations()
    if history_observations is not None:
        example_batch["history_observations"] = _add_batch_dimension(
            history_observations
        )
    return example_batch


def _restore_mtql_agent(agent):
    if FLAGS.checkpoint_format == "native":
        return restore_agent(
            agent,
            FLAGS.restore_path,
            FLAGS.restore_epoch,
        )

    checkpoint_dir = Path(FLAGS.restore_path)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Orbax checkpoint directory does not exist: {checkpoint_dir}"
        )

    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=None,
        overwrite=False,
        resume=True,
    )
    try:
        if not resuming:
            raise FileNotFoundError(
                f"No resumable Orbax checkpoints found in {checkpoint_dir}."
            )

        available_steps = tuple(checkpoint_manager.all_steps())
        if FLAGS.restore_epoch not in available_steps:
            raise ValueError(
                f"Orbax step {FLAGS.restore_epoch} is unavailable; "
                f"available steps: {available_steps}."
            )

        return restore_orbax_checkpoint(
            checkpoint_manager,
            agent,
            step=FLAGS.restore_epoch,
            restore_optimizer=False,
        )
    finally:
        checkpoint_manager.close()


def _validate_required_flags():
    required = {
        "config_task": FLAGS.config_task,
        "norm_stats_path": FLAGS.norm_stats_path,
        "restore_path": FLAGS.restore_path,
        "restore_epoch": FLAGS.restore_epoch,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Missing required flags: "
            + ", ".join(f"--{name}" for name in missing)
        )

    if FLAGS.hist_length < 0:
        raise ValueError("--hist_length must be nonnegative.")
    if FLAGS.hist_stride < 1:
        raise ValueError("--hist_stride must be at least one.")
    if FLAGS.image_size < 1:
        raise ValueError("--image_size must be positive.")
    if FLAGS.action_chunk_size < 1:
        raise ValueError("--action_chunk_size must be at least one.")
    if FLAGS.action_exec_horizon < 0:
        raise ValueError("--action_exec_horizon must be nonnegative.")


def main(argv):
    if len(argv) != 1:
        raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
    _validate_required_flags()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = FLAGS.agent
    if config.get("agent_name") not in (
        "mtql_transformer_real",
        "mtql_mlp_real",
        "new_bc_flow_transformer_real",
        # Preserve compatibility with the earlier real-robot entry point.
        "mtql_transformer_v2",
        "mtql_transformer_language_real",
        "mtql_mlp_language_real",
        "new_bc_flow_transformer_language_real",
    ):
        raise ValueError(
            "The robot evaluator requires agent_name='mtql_transformer_real' "
            "or 'mtql_mlp_real' or 'new_bc_flow_transformer_real' (or a legacy mtql_transformer_v2 checkpoint)."
        )

    task_config = FLAGS.config_task
    task_name = _task_name(task_config)
    # Egg variants (for example egg_v2) share the same cue-free evaluator
    # behavior as the original Egg task.
    egg_task = task_name == "egg" or task_name.startswith("egg_")
    shuffle_task = task_name == "shuffle"
    language_agent = "language" in config.get("agent_name", "")
    if language_agent and FLAGS.cue_mode != "language":
        raise ValueError(
            "cue_mode must match the checkpoint: use cue_mode=language with a "
            "*_language_real agent."
        )
    if not language_agent and FLAGS.cue_mode == "language":
        raise ValueError(
            "cue_mode=language requires a *_language_real agent."
        )
    if egg_task and FLAGS.cue_mode != "none":
        raise ValueError(
            "Egg evaluation is cue-free: use cue_mode=none."
        )
    if not egg_task and not shuffle_task and FLAGS.cue_mode == "none":
        raise ValueError(
            "cue_mode=none is only supported for the Egg task."
        )

    gripper_mode = FLAGS.gripper_mode
    # Only policy mode may issue gripper RPCs. Hold deliberately uses the
    # arm-only path: even a neutral 7D gripper action can refresh a low-force
    # Robotiq goto command and relax a manually set grip.
    physical_control_gripper = gripper_mode == "policy"

    # Match the configuration adjustment performed by m_main.py during training.
    hidden_dim = config.get("hidden_dim")
    if hidden_dim and "num_heads" in config:
        if hidden_dim % 32:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by 32."
            )
        config["num_heads"] = hidden_dim // 32
    config["action_chunk_size"] = FLAGS.action_chunk_size

    results_root = (
        Path(FLAGS.results_dir)
        if FLAGS.results_dir
        else _default_results_dir(FLAGS.restore_path, FLAGS.restore_epoch)
    )
    results_dir = _next_trial_dir(results_root)
    results_path = results_dir / "results.json"
    logger.info("Evaluation trial directory: %s", results_dir)
    if FLAGS.video_dir:
        os.makedirs(FLAGS.video_dir, exist_ok=True)

    results = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "results_path": str(results_path),
        "checkpoint": {
            "restore_path": str(FLAGS.restore_path),
            "restore_epoch": int(FLAGS.restore_epoch),
            "checkpoint_format": FLAGS.checkpoint_format,
        },
        "agent": str(FLAGS.agent),
        "agent_name": config["agent_name"],
        "task_name": task_name,
        "cue_mode": FLAGS.cue_mode,
        "norm_stats_path": str(FLAGS.norm_stats_path),
        "task_config": str(FLAGS.config_task),
        "hist_length": int(FLAGS.hist_length),
        "hist_stride": int(FLAGS.hist_stride),
        "image_size": int(FLAGS.image_size),
        "action_chunk_size": int(FLAGS.action_chunk_size),
        "action_exec_horizon": int(FLAGS.action_exec_horizon),
        "max_episode_steps": int(FLAGS.config_task.auto_reset_steps),
        "control_hz": float(FLAGS.config_task.control_hz),
        "video_dir": FLAGS.video_dir,
        "gripper_mode": gripper_mode,
        "episodes_requested": int(FLAGS.num_episodes),
        "gripper_diagnostics": [],
        "episodes": [],
    }
    _update_result_summary(results)
    _write_results(results_path, results)

    normalizer = OpenPINormalizer.from_path(FLAGS.norm_stats_path)
    state_q01 = normalizer.norm_stats["state"].q01
    if state_q01 is None:
        raise ValueError("State normalization stats must include q01.")
    state_q01_array = np.asarray(state_q01, dtype=np.float32).reshape(-1)
    state_q99 = normalizer.norm_stats["state"].q99
    if state_q99 is None or state_q01_array.size == 0:
        raise ValueError("State normalization stats must include gripper q01/q99.")
    state_q99_array = np.asarray(state_q99, dtype=np.float32).reshape(-1)
    if state_q99_array.shape != state_q01_array.shape:
        raise ValueError("State q01 and q99 statistics have different dimensions.")
    # The v2 raw data keeps the gripper closed at one constant width. Use the
    # training q01 value (the exact value used by OpenPI normalization), and
    # verify that q99 agrees with it instead of silently injecting a changing
    # or mismatched gripper value.
    training_gripper_position = float(state_q01_array[-1])
    training_gripper_q99 = float(state_q99_array[-1])
    if FLAGS.inject_training_gripper_state and not np.isclose(
        training_gripper_position, training_gripper_q99, atol=1e-4
    ):
        raise ValueError(
            "Cannot inject a constant training gripper state: "
            f"q01={training_gripper_position:.9f}, q99={training_gripper_q99:.9f}."
        )
    injected_gripper_position = (
        training_gripper_position
        if FLAGS.inject_training_gripper_state
        else None
    )
    results["inject_training_gripper_state"] = bool(
        FLAGS.inject_training_gripper_state
    )
    results["training_gripper_reference"] = training_gripper_position
    results["training_gripper_q99"] = training_gripper_q99
    results["injected_policy_gripper_position"] = injected_gripper_position
    _write_results(results_path, results)
    if injected_gripper_position is not None:
        logger.warning(
            "Injecting constant training gripper_position=%.9f into every policy observation; "
            "physical gripper observations remain unchanged.",
            injected_gripper_position,
        )
    history_buffer = DroidHistoryBuffer(
        normalizer,
        hist_length=FLAGS.hist_length,
        hist_stride=FLAGS.hist_stride,
        image_size=FLAGS.image_size,
        visual_cues=FLAGS.cue_mode == "visual",
    )

    example_action = np.asarray(
        task_config.example_action,
        dtype=np.float32,
    )
    env = EnvClientWrapper(
        env_creation_request={
            "example_action": example_action,
            "env_usage": "eval",
            "video_dir": FLAGS.video_dir or "",
        },
        host=FLAGS.client_host,
        port=FLAGS.client_port,
    )

    logger.info(
        "Connecting to DROID rollout server at %s:%d",
        FLAGS.client_host,
        FLAGS.client_port,
    )
    # This reset also becomes evaluation episode 1. Reusing it avoids an
    # extra unnumbered target reset before the first rollout.
    physical_observation = env.reset(
        eval_episode=1,
        eval_num_episodes=FLAGS.num_episodes,
    )
    observation = _inject_training_gripper_state(
        physical_observation,
        injected_gripper_position,
    )
    history_buffer.reset(observation)

    def log_gripper_state(label, obs):
        raw_position = float(np.asarray(obs["gripper_position"]).reshape(-1)[0])
        physical_state = np.concatenate(
            [
                np.asarray(obs["cartesian_position"], dtype=np.float32).reshape(-1),
                np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1),
            ]
        )
        policy_obs = _inject_training_gripper_state(obs, injected_gripper_position)
        policy_state = np.concatenate(
            [
                np.asarray(policy_obs["cartesian_position"], dtype=np.float32).reshape(-1),
                np.asarray(policy_obs["gripper_position"], dtype=np.float32).reshape(-1),
            ]
        )
        physical_normalized = float(normalizer.normalize_state(physical_state)[-1])
        policy_normalized = float(normalizer.normalize_state(policy_state)[-1])
        policy_position = float(np.asarray(policy_obs["gripper_position"]).reshape(-1)[0])
        logger.info(
            "Gripper diagnostic (%s): physical=%.9f physical_normalized=%.6g "
            "policy=%.9f policy_normalized=%.6g training_q01=%.9f mode=%s",
            label,
            raw_position,
            physical_normalized,
            policy_position,
            policy_normalized,
            training_gripper_position,
            gripper_mode,
        )
        results["gripper_diagnostics"].append(
            {
                "label": label,
                "physical_position": raw_position,
                "physical_normalized_state": physical_normalized,
                "policy_position": policy_position,
                "policy_normalized_state": policy_normalized,
                "training_reference_position": training_gripper_position,
            }
        )
        _write_results(results_path, results)

    log_gripper_state("initial reset", physical_observation)

    example_batch = _build_example_batch(
        observation,
        history_buffer,
        FLAGS.action_chunk_size,
        FLAGS.cue_mode,
    )
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )
    agent = _restore_mtql_agent(agent)
    logger.info(
        "Checkpoint loaded successfully: format=%s step=%d path=%s",
        FLAGS.checkpoint_format,
        FLAGS.restore_epoch,
        FLAGS.restore_path,
    )

    sampling_rng = jax.random.PRNGKey(FLAGS.seed)
    control_hz = float(task_config.control_hz)
    control_period = 1.0 / control_hz
    max_episode_steps = int(task_config.auto_reset_steps)

    successes = []
    completed_episodes = 0
    use_initial_observation = True
    while completed_episodes < FLAGS.num_episodes:
        eval_episode = completed_episodes + 1
        logger.info(
            "Starting evaluation episode %d/%d",
            eval_episode,
            FLAGS.num_episodes,
        )
        if not use_initial_observation:
            physical_observation = env.reset(
                eval_episode=eval_episode,
                eval_num_episodes=FLAGS.num_episodes,
            )
            observation = _inject_training_gripper_state(
                physical_observation,
                injected_gripper_position,
            )
        use_initial_observation = False
        target_count = (
            None
            if egg_task or shuffle_task
            else _target_count(observation)
        )
        if egg_task:
            print(f"EGG EVALUATION {eval_episode}/{FLAGS.num_episodes}", flush=True)
        elif shuffle_task:
            print(
                f"SHUFFLE EVALUATION {eval_episode}/{FLAGS.num_episodes}",
                flush=True,
            )
        else:
            print(f"EVALUATION TARGET: {target_count}", flush=True)
        print("WAITING 1 SECOND BEFORE START", flush=True)
        time.sleep(1.0)
        print("STARTING EVALUATION", flush=True)
        history_buffer.reset(observation)
        log_gripper_state(
            f"episode {eval_episode} reset",
            physical_observation,
        )
        action_plan = deque()
        success = False
        done = False
        episode_steps = 0

        for step in range(max_episode_steps):
            loop_start = time.monotonic()
            physical_observation = env.get_observation()
            done, success, _, _ = env.get_info_for_step()
            if done:
                break
            observation = _inject_training_gripper_state(
                physical_observation,
                injected_gripper_position,
            )

            if not action_plan:
                target_count = (
                    None
                    if egg_task or shuffle_task
                    else _target_count(observation)
                )
                mtql_observation = convert_droid_observation(
                    observation,
                    normalizer,
                    image_size=FLAGS.image_size,
                    target_count=(
                        target_count if FLAGS.cue_mode == "visual" else None
                    ),
                )
                history_observations = (
                    history_buffer.history_observations()
                )

                sampling_rng, action_rng = jax.random.split(sampling_rng)
                sample_kwargs = {
                    "observations": _add_batch_dimension(mtql_observation),
                    "history_observations": (
                        _add_batch_dimension(history_observations)
                        if history_observations is not None
                        else None
                    ),
                    "seed": action_rng,
                    "temperature": 0,
                }
                if FLAGS.cue_mode == "language":
                    sample_kwargs["language_prompt_ids"] = np.asarray(
                        [target_count], dtype=np.int32
                    )
                normalized_actions = agent.sample_actions(**sample_kwargs)
                normalized_actions = np.asarray(
                    jax.device_get(normalized_actions)
                )[0]
                raw_actions = normalizer.unnormalize_actions(
                    normalized_actions
                )

                if raw_actions.ndim == 1:
                    raw_actions = raw_actions[None]
                if raw_actions.shape[-1] != 7:
                    raise ValueError(
                        "Expected raw 7D DROID commands, got shape "
                        f"{raw_actions.shape}."
                    )

                execution_horizon = (
                    FLAGS.action_exec_horizon
                    if FLAGS.action_exec_horizon > 0
                    else len(raw_actions)
                )
                action_plan.extend(
                    raw_actions[:execution_horizon]
                )

            raw_action = np.asarray(
                action_plan.popleft(),
                dtype=np.float32,
            )

            # The history contains observations before previously executed
            # actions, matching HistoryDataset's temporal convention.
            history_buffer.append(observation)

            elapsed = time.monotonic() - loop_start
            remaining = control_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

            if gripper_mode == "hold":
                # Keep the model's 7D shape for logging, but the rollout
                # server executes only the first six arm dimensions because
                # physical_control_gripper is false in hold mode.
                raw_action[6] = 0.0
            executed_action, action_type = env.step(
                raw_action.tolist(),
                allow_human_override=False,
                control_gripper=physical_control_gripper,
            )
            if episode_steps < 3:
                executed_array = np.asarray(executed_action, dtype=np.float32)
                logger.info(
                    "Episode %d action %d: policy_max_abs=%.6g executed_max_abs=%.6g action_type=%s raw=%s executed=%s",
                    eval_episode,
                    episode_steps + 1,
                    float(np.max(np.abs(raw_action))),
                    float(np.max(np.abs(executed_array))),
                    action_type,
                    np.array2string(np.asarray(raw_action), precision=5),
                    np.array2string(executed_array, precision=5),
                )
            episode_steps += 1

        # The loop exits after the last action without another pre-action
        # check. Query once more so a timeout closes the episode and flushes
        # its video in the rollout server.
        if not done:
            done, success, _, _ = env.get_info_for_step()

        if success is None:
            logger.info(
                "Episode attempt discarded: target=%s, steps=%d; "
                "completed=%d/%d, no result recorded",
                "none" if target_count is None else target_count,
                episode_steps,
                completed_episodes,
                FLAGS.num_episodes,
            )
            print(
                f"DISCARDED ATTEMPT — completed {completed_episodes}/"
                f"{FLAGS.num_episodes} (not counted)",
                flush=True,
            )
            # Reset immediately rather than relying on the next loop
            # iteration.  This guarantees that option 4 sends the robot back
            # to base before another attempt begins.  The reset observation
            # is reused on the next iteration so we do not reset twice.
            next_episode = completed_episodes + 1
            logger.info(
                "Resetting after discarded attempt for evaluation episode %d/%d",
                next_episode,
                FLAGS.num_episodes,
            )
            physical_observation = env.reset(
                eval_episode=next_episode,
                eval_num_episodes=FLAGS.num_episodes,
            )
            observation = _inject_training_gripper_state(
                physical_observation,
                injected_gripper_position,
            )
            use_initial_observation = True
            continue

        episode_result = {
            "episode": eval_episode,
            "target_count": target_count,
            "success": bool(success),
            "steps": int(episode_steps),
        }
        results["episodes"].append(episode_result)
        completed_episodes += 1
        _update_result_summary(results)
        _write_results(results_path, results)
        successes.append(bool(success))
        logger.info(
            "Episode %d finished: success=%s, target=%s, steps=%d; results=%s",
            eval_episode,
            success,
            "none" if target_count is None else target_count,
            episode_steps,
            results_path,
        )
        print(
            f"COMPLETED EVALUATIONS: {completed_episodes}/"
            f"{FLAGS.num_episodes} | success={bool(success)}",
            flush=True,
        )

    results["status"] = "complete"
    results["finished_at_utc"] = _utc_now()
    _update_result_summary(results)
    _write_results(results_path, results)
    logger.info(
        "Evaluation complete: %d/%d successes (%.1f%%); results saved to %s",
        sum(successes),
        len(successes),
        100.0 * np.mean(successes) if successes else 0.0,
        results_path,
    )


if __name__ == "__main__":
    app.run(main)
