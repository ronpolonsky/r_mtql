#!/usr/bin/env python3
"""Offline Egg MTQL training with separate critic and actor batches.

This is an isolated variant of ``m_real.py``. The critic samples all Egg
outcomes, while the actor independently samples successful transitions only.
The existing trainer and agents remain unchanged.
"""

from __future__ import annotations

import json
import os
import random
import time

import imageio.v2 as imageio
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents as base_agents
from agents.mtql_mlp_success_actor_real import (
    MTQLMLPSuccessActorRealAgent,
)
from agents.mtql_transformer_success_actor_real import (
    MTQLTransformerSuccessActorRealAgent,
)
from utils.flax_utils import print_param_stats
from utils.mtql_orbax import (
    initialize_checkpoint_dir,
    restore_checkpoint,
    restore_rng_metadata,
    save_checkpoint,
)
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.mtql_droid import (
    DEFAULT_IMAGE_SIZE,
    OpenPINormalizer,
    add_droid_target_cues,
    add_droid_target_cues_jax,
    augment_droid_batch,
    create_droid_history_dataset,
    load_egg_history_dataset,
    load_droid_history_dataset,
    process_droid_dataset,
    process_egg_dataset,
)


agents = {
    **base_agents,
    "mtql_transformer_success_actor_real": (
        MTQLTransformerSuccessActorRealAgent
    ),
    "mtql_mlp_success_actor_real": MTQLMLPSuccessActorRealAgent,
}
SUCCESS_ACTOR_AGENT_NAMES = frozenset(
    {
        "mtql_transformer_success_actor_real",
        "mtql_mlp_success_actor_real",
    }
)


FLAGS = flags.FLAGS

flags.DEFINE_integer("enable_wandb", 1, "Whether to use wandb.")
flags.DEFINE_string("wandb_run_group", "droid_real", "W&B run group.")
flags.DEFINE_string("project", "dfrl", "W&B project.")
flags.DEFINE_string("wandb_mode", "online", "W&B mode.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("save_dir", "exp/", "Output directory.")
flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Orbax checkpoint directory; defaults to <save_dir>/checkpoints.",
)
flags.DEFINE_string(
    "restore_path",
    None,
    "Orbax checkpoint directory to restore from.",
)
flags.DEFINE_integer(
    "restore_epoch",
    None,
    "Orbax checkpoint step to restore; resume uses the latest step when omitted.",
)
flags.DEFINE_bool(
    "resume",
    False,
    "Resume from the latest checkpoint in checkpoint_dir.",
)
flags.DEFINE_bool(
    "overwrite",
    False,
    "Delete an existing checkpoint_dir before starting a new run.",
)
flags.DEFINE_integer(
    "checkpoint_max_to_keep",
    100,
    "Maximum number of recent Orbax checkpoints to retain in addition to "
    "checkpoints protected by checkpoint_keep_period.",
)
flags.DEFINE_integer(
    "checkpoint_keep_period",
    None,
    "Keep every Nth Orbax checkpoint permanently; None uses Orbax defaults.",
)
flags.DEFINE_bool(
    "save_checkpoints",
    True,
    "Whether to write Orbax checkpoints during training.",
)

flags.DEFINE_string("dataset_path", None, "Directory containing DROID episodes.")
flags.DEFINE_enum(
    "dataset_kind",
    "candy",
    ["candy", "egg"],
    "Dataset layout. Egg mode loads every finalized episode and has no cue.",
)
flags.DEFINE_string(
    "dataset_cache",
    None,
    "Optional preprocessed DROID cache directory. When set, raw HDF5 files "
    "are not decoded at trainer startup.",
)
flags.DEFINE_string(
    "norm_stats_path",
    None,
    "OpenPI norm-stats directory or exact norm_stats.json path.",
)
flags.DEFINE_integer(
    "n_succ",
    -1,
    "Total successful episodes to load for Egg, split near-evenly across "
    "card colors. For non-Egg data, this is balanced across target counts 1-3; "
    "-1 loads all.",
)
flags.DEFINE_integer(
    "n_fails",
    -1,
    "Total failed episodes to load for Egg, split near-evenly across card colors. "
    "For non-Egg data, this is balanced across target counts 1-3; -1 loads all.",
)
flags.DEFINE_integer(
    "image_size",
    DEFAULT_IMAGE_SIZE,
    "Square image size used by the real-data adapter.",
)
config_flags.DEFINE_config_file(
    "config_task",
    None,
    "EXPO-FT DROID task config used to select action keys.",
    lock_config=False,
)

flags.DEFINE_integer("train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Training/logging interval.")
flags.DEFINE_integer("save_interval", 200000, "Checkpoint interval.")
flags.DEFINE_integer("hist_length", 20, "Number of sparse history observations.")
flags.DEFINE_integer("hist_stride", 7, "Stride between sparse history observations.")
flags.DEFINE_integer(
    "action_chunk_size",
    25,
    "Number of consecutive actions predicted per training sample.",
)
flags.DEFINE_float(
    "p_aug",
    1.0,
    "Probability of applying EXPO-style image augmentation.",
)
flags.DEFINE_enum(
    "cue_mode",
    "visual",
    ["none", "visual", "language"],
    "Conditioning mode. Use none for egg data.",
)
flags.DEFINE_string(
    "save_augmented_video",
    None,
    "Optional MP4 path for one augmented offline training sample; a JSON sidecar is written too.",
)

config_flags.DEFINE_config_file(
    "agent",
    "agents/mtql_transformer_success_actor_real.py",
    lock_config=False,
)


def _checkpoint_metadata(**training_contract: object) -> dict[str, object]:
    """Capture RNG state and data contract for an exact continuation."""
    metadata = {
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
    }
    metadata.update(training_contract)
    return metadata


def _save_augmented_training_sample(
    batch: dict,
    cue_targets: np.ndarray | None,
    episode_success: np.ndarray | None,
    output_mp4: str,
) -> None:
    """Save one augmented training sample and available source metadata."""
    observations = batch["observations"]
    history = batch.get("history_observations")
    base = np.asarray(observations["image"])[0]
    wrist = np.asarray(observations["wrist_image"])[0]
    if history is not None:
        base = np.concatenate([np.asarray(history["image"])[0], base[None]], axis=0)
        wrist = np.concatenate(
            [np.asarray(history["wrist_image"])[0], wrist[None]], axis=0
        )
    frames = np.concatenate([base, wrist], axis=2)

    output_path = os.path.abspath(output_mp4)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=4,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))

    sample_index = 0
    metadata = {
        "video": output_path,
        "kind": "augmented_offline_training_sample",
        "camera_layout": "exterior_left__wrist_right",
        "num_frames": int(len(frames)),
        "sampled_transition_reward": float(
            np.asarray(batch["rewards"])[sample_index]
        ),
        "note": (
            "This is one augmented offline training sample, not a "
            "policy-generated environment rollout."
        ),
    }
    if cue_targets is not None:
        metadata["target_cue"] = int(np.asarray(cue_targets)[sample_index])
    if episode_success is not None:
        source_success = bool(np.asarray(episode_success)[sample_index])
        metadata["source_episode_success"] = source_success
        metadata["source_outcome"] = "success" if source_success else "failure"
    with open(os.path.splitext(output_path)[0] + ".json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[m_real] saved augmented training sample: {output_path}", flush=True)


def _validate_flags() -> None:
    required = {
        "config_task": FLAGS.config_task,
        "dataset_path": FLAGS.dataset_path,
        "norm_stats_path": FLAGS.norm_stats_path,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Missing required flags: "
            + ", ".join(f"--{name}" for name in missing)
        )
    if FLAGS.dataset_cache is not None and not os.path.isdir(FLAGS.dataset_cache):
        raise ValueError(
            f"DROID dataset cache directory not found: {FLAGS.dataset_cache}"
        )
    if FLAGS.agent.get("agent_name") not in {
        "mtql_transformer_success_actor_real",
        "mtql_mlp_success_actor_real",
    }:
        raise ValueError(
            "m_real_success_actor.py received an unsupported agent name."
        )
    language_agent = "language" in FLAGS.agent.get("agent_name", "")
    if FLAGS.dataset_kind == "egg":
        if FLAGS.cue_mode != "none":
            raise ValueError("Egg training requires --cue_mode=none.")
        if language_agent:
            raise ValueError(
                "Egg training uses raw visual observations and requires a "
                "non-language *_real agent."
            )
    elif FLAGS.cue_mode == "none":
        raise ValueError("Candy training requires visual or language cues.")
    elif (FLAGS.cue_mode == "language") != language_agent:
        raise ValueError(
            "cue_mode must match the agent: use cue_mode=language with a "
            "*_language_real agent and cue_mode=visual with the existing agents."
        )
    if FLAGS.n_succ < -1:
        raise ValueError("--n_succ must be -1 or nonnegative.")
    if FLAGS.n_fails < -1:
        raise ValueError("--n_fails must be -1 or nonnegative.")
    if FLAGS.train_steps < 1:
        raise ValueError("--train_steps must be positive.")
    if FLAGS.log_interval < 1:
        raise ValueError("--log_interval must be positive.")
    if FLAGS.save_interval < 1:
        raise ValueError("--save_interval must be positive.")
    if FLAGS.hist_length < 0:
        raise ValueError("--hist_length must be nonnegative.")
    if FLAGS.hist_stride < 1:
        raise ValueError("--hist_stride must be at least one.")
    if FLAGS.action_chunk_size < 1:
        raise ValueError("--action_chunk_size must be positive.")
    if FLAGS.image_size < 1:
        raise ValueError("--image_size must be positive.")
    if not 0.0 <= FLAGS.p_aug <= 1.0:
        raise ValueError("--p_aug must be between zero and one.")
    if FLAGS.restore_epoch is not None and FLAGS.restore_path is None:
        raise ValueError("--restore_epoch requires --restore_path.")
    if FLAGS.resume and FLAGS.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if FLAGS.restore_path is not None and FLAGS.overwrite:
        raise ValueError("--overwrite cannot be used with --restore_path.")
    if FLAGS.checkpoint_max_to_keep < 1:
        raise ValueError("--checkpoint_max_to_keep must be positive.")
    if FLAGS.checkpoint_keep_period is not None and FLAGS.checkpoint_keep_period < 1:
        raise ValueError("--checkpoint_keep_period must be positive when set.")
    if not FLAGS.save_checkpoints and (FLAGS.resume or FLAGS.restore_path is not None):
        raise ValueError(
            "--save_checkpoints=false cannot be combined with --resume or --restore_path."
        )


def _print_language_token_summary(config, transitions) -> None:
    """Print the prompt-token and sequence accounting for smoke/training logs."""
    if FLAGS.cue_mode != "language":
        return

    tokenization_mode = config.get("tokenization_mode", "per_modality")
    action_dim = int(np.asarray(transitions[0]["actions"]).shape[-1])
    if tokenization_mode == "per_modality":
        obs_tokens = 1
        action_tokens = int(config.get("num_action_tokens", 1))
        token_detail = "per_modality"
    elif tokenization_mode == "linear_projected_per_dim":
        obs_tokens = int(config.get("obs_token_dim", 32))
        action_tokens = int(config.get("action_token_dim", 8))
        token_detail = "linear_projected_per_dim"
    elif tokenization_mode == "per_dim":
        # For the real DROID encoder, per_dim is normally not used; the raw
        # action dimensionality is nevertheless known here for diagnostics.
        obs_tokens = "encoded_obs_dim"
        action_tokens = action_dim
        token_detail = "per_dim"
    else:
        raise ValueError(f"Unsupported tokenization_mode: {tokenization_mode}")

    prompt_ids = sorted({int(transition["cue_target"]) for transition in transitions})
    embedding_count = int(config.get("language_prompt_num_embeddings", 4))
    hidden_dim = int(config.get("hidden_dim", 0))
    if isinstance(obs_tokens, int):
        actor_tokens = 1 + FLAGS.hist_length * obs_tokens + obs_tokens + 1
        critic_tokens = (
            1
            + FLAGS.hist_length * obs_tokens
            + obs_tokens
            + FLAGS.action_chunk_size * action_tokens
            + 1
        )
        sequence_text = f"actor={actor_tokens}"
        if config.get("agent_name") != "new_bc_flow_transformer_language_real":
            sequence_text += f" critic={critic_tokens}"
        sequence_text += " (including CLS)"
    else:
        sequence_text = (
            "actor=1(prompt)+hist*encoded_obs_dim+encoded_obs_dim+1(CLS); "
            "critic=1(prompt)+hist*encoded_obs_dim+encoded_obs_dim+"
            f"{FLAGS.action_chunk_size}*{action_tokens}+1(CLS)"
        )
    print(
        f"[m_real] language cue enabled: prompt_token_count=1 "
        f"embedding_table=({embedding_count},{hidden_dim}) "
        f"prompt_ids_present={prompt_ids} tokenization={token_detail}",
        flush=True,
    )
    print(
        f"[m_real] language-conditioned token counts per sample: "
        f"{sequence_text}; batch_size={config.get('batch_size')}",
        flush=True,
    )


def main(argv):
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    if len(argv) != 1:
        raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
    _validate_flags()

    config = FLAGS.agent
    config["train_steps"] = FLAGS.train_steps
    config["action_chunk_size"] = FLAGS.action_chunk_size
    if config.get("agent_name") not in SUCCESS_ACTOR_AGENT_NAMES:
        raise ValueError(
            "m_real_success_actor.py only supports the new separate-batch "
            "agents; got agent_name="
            f"{config.get('agent_name')!r}."
        )
    if FLAGS.dataset_kind != "egg" or FLAGS.cue_mode != "none":
        raise ValueError(
            "Success-only actor batching is currently restricted to "
            "--dataset_kind=egg --cue_mode=none."
        )

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    exp_name = get_exp_name(
        FLAGS.seed,
        algo=config.get("agent_name"),
        hist_length=FLAGS.hist_length,
        image_obs=True,
    )
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    dim = config.get("hidden_dim")
    if dim and "num_heads" in config:
        if dim % 32:
            raise ValueError(f"hidden_dim ({dim}) must be divisible by 32")
        config["num_heads"] = dim // 32

    if FLAGS.enable_wandb:
        wandb_run, trigger_sync = setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project=FLAGS.project,
            group=FLAGS.wandb_run_group,
            name=exp_name,
            mode=FLAGS.wandb_mode,
        )
    else:
        trigger_sync = None

    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as handle:
        json.dump(get_flag_dict(), handle, default=str)

    checkpoint_dir = (
        FLAGS.checkpoint_dir
        or os.path.join(FLAGS.save_dir, "checkpoints")
    )
    same_checkpoint_source = (
        FLAGS.restore_path is not None
        and os.path.realpath(FLAGS.restore_path) == os.path.realpath(checkpoint_dir)
    )
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir,
        max_to_keep=FLAGS.checkpoint_max_to_keep,
        keep_period=FLAGS.checkpoint_keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume or same_checkpoint_source,
    )
    restore_manager = checkpoint_manager
    if FLAGS.restore_path is not None and not same_checkpoint_source:
        restore_manager, _ = initialize_checkpoint_dir(
            FLAGS.restore_path,
            max_to_keep=FLAGS.checkpoint_max_to_keep,
            resume=True,
        )

    if FLAGS.dataset_kind == "egg":
        if FLAGS.dataset_cache:
            train_dataset, cache_info = load_egg_history_dataset(
                FLAGS.dataset_cache,
                dataset_path=FLAGS.dataset_path,
                norm_stats_path=FLAGS.norm_stats_path,
                task_config=FLAGS.config_task,
                image_size=FLAGS.image_size,
                hist_length=FLAGS.hist_length,
                hist_stride=FLAGS.hist_stride,
                action_chunk_size=FLAGS.action_chunk_size,
                discount=config.get("discount", 0.99),
                n_success=FLAGS.n_succ,
                n_failure=FLAGS.n_fails,
            )
            transitions = [
                {
                    "actions": action,
                    "episode_success": episode_success,
                }
                for action, episode_success in zip(
                    cache_info["raw_actions"],
                    cache_info["episode_success"],
                )
            ]
        else:
            transitions = process_egg_dataset(
                FLAGS.dataset_path,
                FLAGS.config_task,
                n_success=FLAGS.n_succ,
                n_failure=FLAGS.n_fails,
            )
            normalizer = OpenPINormalizer.from_path(FLAGS.norm_stats_path)
            train_dataset = create_droid_history_dataset(
                transitions,
                normalizer,
                hist_length=FLAGS.hist_length,
                hist_stride=FLAGS.hist_stride,
                action_chunk_size=FLAGS.action_chunk_size,
                discount=config.get("discount", 0.99),
                image_size=FLAGS.image_size,
                store_next_observations=False,
            )
    elif FLAGS.dataset_cache:
        train_dataset, cache_info = load_droid_history_dataset(
            FLAGS.dataset_cache,
            dataset_path=FLAGS.dataset_path,
            norm_stats_path=FLAGS.norm_stats_path,
            task_config=FLAGS.config_task,
            image_size=FLAGS.image_size,
            n_success=FLAGS.n_succ,
            n_failure=FLAGS.n_fails,
            hist_length=FLAGS.hist_length,
            hist_stride=FLAGS.hist_stride,
            action_chunk_size=FLAGS.action_chunk_size,
            discount=config.get("discount", 0.99),
        )
        # Keep the existing diagnostics and language-token accounting without
        # duplicating the cached image/proprio arrays in Python objects.
        transitions = [
            {
                "actions": action,
                "cue_target": cue_target,
                "episode_success": episode_success,
            }
            for action, cue_target, episode_success in zip(
                cache_info["raw_actions"],
                cache_info["cue_targets"],
                cache_info["episode_success"],
            )
        ]
    else:
        transitions = process_droid_dataset(
            FLAGS.dataset_path,
            FLAGS.config_task,
            n_success=FLAGS.n_succ,
            n_failure=FLAGS.n_fails,
        )
        normalizer = OpenPINormalizer.from_path(FLAGS.norm_stats_path)
        train_dataset = create_droid_history_dataset(
            transitions,
            normalizer,
            hist_length=FLAGS.hist_length,
            hist_stride=FLAGS.hist_stride,
            action_chunk_size=FLAGS.action_chunk_size,
            discount=config.get("discount", 0.99),
            image_size=FLAGS.image_size,
        )
    sampling_policy = "critic_all_outcomes_actor_success_only"
    sampling_transition_count = int(train_dataset.size)
    actor_sampling_policy = "success_only"
    actor_sampling_transition_count = None
    actor_sampling_idxs = None
    egg_success_episode_count = None
    egg_failure_episode_count = None
    egg_success_transition_count = None
    egg_failure_transition_count = None
    if FLAGS.dataset_kind == "egg":
        if "episode_success" not in train_dataset:
            raise ValueError(
                "Egg training requires per-transition episode outcomes."
            )
        episode_success = np.asarray(train_dataset["episode_success"], dtype=bool)
        success_idxs = np.flatnonzero(episode_success)
        failure_idxs = np.flatnonzero(~episode_success)
        terminal_locs = np.asarray(train_dataset.terminal_locs)
        terminal_success = episode_success[terminal_locs]
        egg_success_episode_count = int(terminal_success.sum())
        egg_failure_episode_count = int((~terminal_success).sum())
        egg_success_transition_count = int(len(success_idxs))
        egg_failure_transition_count = int(len(failure_idxs))
        if egg_success_transition_count == 0:
            raise ValueError(
                "Success-only actor training requires at least one selected "
                "successful Egg transition."
            )

        print(
            "[m_real] Egg data used for training: "
            f"episodes={egg_success_episode_count + egg_failure_episode_count} "
            f"(success={egg_success_episode_count}, "
            f"failure={egg_failure_episode_count}); "
            f"transitions={int(train_dataset.size)} "
            f"(success={egg_success_transition_count}, "
            f"failure={egg_failure_transition_count})",
            flush=True,
        )

        actor_sampling_idxs = success_idxs
        actor_sampling_transition_count = egg_success_transition_count
        print(
            "[m_real_success_actor] critic sampling: all outcomes "
            f"({egg_success_episode_count} successes + "
            f"{egg_failure_episode_count} failures, "
            f"{sampling_transition_count} transitions); actor sampling: "
            f"success-only ({egg_success_episode_count} episodes, "
            f"{actor_sampling_transition_count} transitions)",
            flush=True,
        )

    action_array = np.asarray([transition["actions"] for transition in transitions])
    if FLAGS.dataset_kind == "egg" and sampling_policy == "success_only":
        action_array = action_array[np.asarray(train_dataset.sampling_idxs)]
    gripper_action = action_array[:, -1]
    gripper_nonzero_fraction = float(np.mean(np.abs(gripper_action) > 1e-6))
    gripper_abs_max = float(np.max(np.abs(gripper_action)))
    if gripper_abs_max <= 1e-6:
        print(
            "[m_real] WARNING: every selected gripper action is zero. "
            "This dataset cannot teach gripper open/close behavior; "
            "recollect with gripper button events before relying on it.",
            flush=True,
        )
    else:
        print(
            f"[m_real] gripper action coverage: "
            f"nonzero_fraction={gripper_nonzero_fraction:.4f}, "
            f"abs_max={gripper_abs_max:.6g}",
            flush=True,
        )
    _print_language_token_summary(config, transitions)

    if FLAGS.enable_wandb:
        # Record derived data/preprocessing facts alongside the command-line
        # configuration so each run is self-describing in W&B.
        wandb.config.update(
            {
                "dataset_transitions": len(transitions),
                "training_sampling_policy": sampling_policy,
                "training_sampling_transitions": sampling_transition_count,
                "actor_sampling_policy": actor_sampling_policy,
                "actor_sampling_transitions": actor_sampling_transition_count,
                "dataset_requested_n_succ": FLAGS.n_succ,
                "dataset_requested_n_fails": FLAGS.n_fails,
                "dataset_gripper_action_nonzero_fraction": gripper_nonzero_fraction,
                "dataset_gripper_action_abs_max": gripper_abs_max,
                "preprocessing_hist_length": FLAGS.hist_length,
                "preprocessing_hist_stride": FLAGS.hist_stride,
                "preprocessing_action_chunk_size": FLAGS.action_chunk_size,
                "preprocessing_image_size": FLAGS.image_size,
                "preprocessing_augmentation_probability": FLAGS.p_aug,
                "conditioning_cue_mode": FLAGS.cue_mode,
                "normalization_stats_path": FLAGS.norm_stats_path,
                "dataset_cache": FLAGS.dataset_cache,
                "egg_success_episodes": egg_success_episode_count,
                "egg_failure_episodes": egg_failure_episode_count,
                "egg_success_transitions": egg_success_transition_count,
                "egg_failure_transitions": egg_failure_transition_count,
            },
            allow_val_change=True,
        )

    checkpoint_training_contract = {
        "training_sampling_policy": sampling_policy,
        "training_sampling_transitions": sampling_transition_count,
        "actor_sampling_policy": actor_sampling_policy,
        "actor_sampling_transitions": actor_sampling_transition_count,
    }
    with open(
        os.path.join(FLAGS.save_dir, "data_contract.json"), "w"
    ) as handle:
        json.dump(
            {
                **checkpoint_training_contract,
                "dataset_kind": FLAGS.dataset_kind,
                "dataset_transitions": len(transitions),
                "egg_success_episodes": egg_success_episode_count,
                "egg_failure_episodes": egg_failure_episode_count,
                "egg_success_transitions": egg_success_transition_count,
                "egg_failure_transitions": egg_failure_transition_count,
                "dataset_requested_n_succ": FLAGS.n_succ,
                "dataset_requested_n_fails": FLAGS.n_fails,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    example_batch = train_dataset.sample(1)
    example_cue_targets = example_batch.pop("cue_targets", None)
    example_batch.pop("episode_success", None)
    if FLAGS.cue_mode == "language":
        example_batch["language_prompt_ids"] = example_cue_targets
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(FLAGS.seed, example_batch, config)
    print_param_stats(agent)

    start_step = 1
    restore_step = None
    if FLAGS.restore_path is not None:
        if FLAGS.restore_epoch is not None:
            restore_step = int(FLAGS.restore_epoch)
        else:
            available_steps = tuple(restore_manager.all_steps())
            if not available_steps:
                raise ValueError(
                    f"No Orbax checkpoints found under {FLAGS.restore_path}."
                )
            restore_step = max(available_steps)
    elif resuming:
        available_steps = tuple(checkpoint_manager.all_steps())
        if available_steps:
            restore_step = max(available_steps)
    if restore_step is not None:
        agent, metadata = restore_checkpoint(
            restore_manager,
            agent,
            restore_step,
            return_metadata=True,
        )
        restored_sampling_policy = metadata.get("training_sampling_policy")
        if restored_sampling_policy != sampling_policy:
            raise ValueError(
                "Checkpoint sampling policy does not match this run: "
                f"checkpoint={restored_sampling_policy!r}, "
                f"requested={sampling_policy!r}."
            )
        restored_actor_sampling_policy = metadata.get(
            "actor_sampling_policy"
        )
        if restored_actor_sampling_policy != actor_sampling_policy:
            raise ValueError(
                "Checkpoint actor sampling policy does not match this run: "
                f"checkpoint={restored_actor_sampling_policy!r}, "
                f"requested={actor_sampling_policy!r}."
            )
        if metadata:
            restore_rng_metadata(metadata)
        start_step = restore_step + 1
        print(f"[m_real] restored Orbax checkpoint at step {restore_step}")

    if start_step > FLAGS.train_steps:
        print(f"[m_real] start_step ({start_step}) exceeds train_steps; nothing to do.")
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        if restore_manager is not checkpoint_manager:
            restore_manager.close()
        if FLAGS.enable_wandb:
            wandb.finish()
        return

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    first_time = time.time()
    last_time = first_time
    sample_time = update_time = 0.0
    saved_augmented_video = False

    try:
        for step in tqdm.tqdm(
            range(start_step, FLAGS.train_steps + 1),
            smoothing=0.1,
            dynamic_ncols=True,
        ):
            start = time.time()
            batch = train_dataset.sample(config["batch_size"])
            actor_idxs = actor_sampling_idxs[
                np.random.randint(
                    len(actor_sampling_idxs),
                    size=config["batch_size"],
                )
            ]
            actor_batch = train_dataset.sample_actor_batch(
                config["batch_size"],
                idxs=actor_idxs,
            )
            sample_time += time.time() - start
            cue_targets = batch.pop("cue_targets", None)
            episode_success = batch.pop("episode_success", None)
            actor_cue_targets = actor_batch.pop("cue_targets", None)
            actor_episode_success = actor_batch.pop(
                "episode_success", None
            )
            if actor_cue_targets is not None:
                raise ValueError(
                    "Success-only actor batches do not support cue targets."
                )
            if (
                actor_episode_success is None
                or not bool(np.asarray(actor_episode_success).all())
            ):
                raise ValueError(
                    "Actor batch contained a failed or unlabeled transition."
                )
            critic_success_fraction = float(
                np.asarray(episode_success, dtype=np.float32).mean()
            )
            actor_success_fraction = float(
                np.asarray(
                    actor_episode_success,
                    dtype=np.float32,
                ).mean()
            )

            if FLAGS.p_aug > 0.0 and np.random.rand() < FLAGS.p_aug:
                critic_aug_rng = jax.random.PRNGKey(FLAGS.seed + step)
                actor_aug_rng = jax.random.fold_in(critic_aug_rng, 1)
                batch = augment_droid_batch(batch, critic_aug_rng)
                actor_batch = augment_droid_batch(
                    actor_batch,
                    actor_aug_rng,
                )
            if FLAGS.cue_mode == "visual":
                assert cue_targets is not None
                batch = add_droid_target_cues_jax(batch, cue_targets)
            elif FLAGS.cue_mode == "language":
                # The target is constant within each trajectory, including
                # the bootstrap next state. No image pixels are modified.
                assert cue_targets is not None
                batch["language_prompt_ids"] = jax.device_put(cue_targets)
                batch["next_language_prompt_ids"] = jax.device_put(cue_targets)

            if FLAGS.save_augmented_video and not saved_augmented_video:
                _save_augmented_training_sample(
                    actor_batch,
                    None,
                    actor_episode_success,
                    FLAGS.save_augmented_video,
                )
                saved_augmented_video = True

            start = time.time()
            agent, update_info = agent.update(
                batch,
                actor_batch,
                step=step,
            )
            update_info = dict(update_info)
            update_info["batch/critic_success_fraction"] = (
                critic_success_fraction
            )
            update_info["batch/actor_success_fraction"] = (
                actor_success_fraction
            )
            update_time += time.time() - start

            if step % FLAGS.log_interval == 0:
                metrics = {f"training/{key}": value for key, value in update_info.items()}
                interval_time = time.time() - last_time
                metrics["time/epoch_time"] = interval_time / FLAGS.log_interval
                metrics["time/total_time"] = time.time() - first_time
                metrics["time/steps_per_sec"] = FLAGS.log_interval / max(interval_time, 1e-9)
                metrics["time/sample_ms"] = sample_time / FLAGS.log_interval * 1000
                metrics["time/update_ms"] = update_time / FLAGS.log_interval * 1000
                sample_time = update_time = 0.0
                last_time = time.time()
                if FLAGS.enable_wandb:
                    wandb.log(metrics, step=step)
                    if FLAGS.wandb_mode == "offline":
                        trigger_sync()
                train_logger.log(metrics, step=step)

            if FLAGS.save_checkpoints and step % FLAGS.save_interval == 0:
                checkpoint_start = time.time()
                save_checkpoint(
                    checkpoint_manager,
                    agent,
                    step,
                    metadata=_checkpoint_metadata(
                        **checkpoint_training_contract
                    ),
                )
                checkpoint_manager.wait_until_finished()
                if FLAGS.enable_wandb:
                    wandb.log(
                        {
                            "checkpoint/saved": 1,
                            "checkpoint/step": step,
                            "time/checkpoint_ms": (time.time() - checkpoint_start) * 1000,
                        },
                        step=step,
                    )

        if FLAGS.save_checkpoints and FLAGS.train_steps % FLAGS.save_interval != 0:
            checkpoint_start = time.time()
            save_checkpoint(
                checkpoint_manager,
                agent,
                FLAGS.train_steps,
                metadata=_checkpoint_metadata(**checkpoint_training_contract),
            )
            checkpoint_manager.wait_until_finished()
            if FLAGS.enable_wandb:
                wandb.log(
                    {
                        "checkpoint/saved": 1,
                        "checkpoint/step": FLAGS.train_steps,
                        "time/checkpoint_ms": (time.time() - checkpoint_start) * 1000,
                    },
                    step=FLAGS.train_steps,
                )
    finally:
        train_logger.close()
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        if restore_manager is not checkpoint_manager:
            restore_manager.close()
        if FLAGS.enable_wandb:
            wandb.finish()


if __name__ == "__main__":
    app.run(main)
