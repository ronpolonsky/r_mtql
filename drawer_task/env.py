"""Training/evaluation wrapper and lazy dataset loader for drawer_task."""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from drawer_task.collect import add_cube_cue
from utils.datasets import LazyEpisodeReplayBuffer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANISKILL_ROOT = PROJECT_ROOT / "cabinet-memory-sim" / "ManiSkill"
DATASET_DIR = PROJECT_ROOT / "drawer_task" / "four_cube_memory_dataset_v1"
if str(MANISKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MANISKILL_ROOT))

ARM_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
ARM_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)
ACTION_LOWER = np.concatenate([ARM_LOWER, np.array([-1.0], np.float32)])
ACTION_UPPER = np.concatenate([ARM_UPPER, np.array([1.0], np.float32)])


def _episode_metadata():
    """Return collector metadata keyed by absolute episode path."""
    records = {}
    for manifest_path in sorted((DATASET_DIR / "manifests").glob("manifest_*.json")):
        for record in json.loads(manifest_path.read_text()):
            records[str(Path(record["path"]).resolve())] = record
    return records


def _balanced_episode_subset(paths, count, success, metadata):
    """Select a deterministic, categorically balanced episode subset."""
    if count == len(paths):
        return list(paths)
    records = []
    lengths = np.asarray(
        [metadata[str(Path(path).resolve())]["length"] for path in paths],
        dtype=np.float64,
    )
    length_edges = np.quantile(lengths, [0.25, 0.5, 0.75])
    for path in paths:
        record = metadata[str(Path(path).resolve())]
        length_bin = int(np.searchsorted(length_edges, record["length"], side="right"))
        primary = record["strategy"] if success else record["failure_reason"]
        records.append(
            {
                "path": path,
                "primary": primary,
                "drawer": record["target_drawer"],
                "color": record["target_color"],
                "length_bin": str(length_bin),
                "primary_drawer": f"{primary}|{record['target_drawer']}",
            }
        )

    feature_weights = {
        "primary": 4.0,
        "drawer": 3.0,
        "color": 1.0,
        "length_bin": 1.0,
        "primary_drawer": 1.0,
    }
    categories = {
        feature: sorted({record[feature] for record in records})
        for feature in feature_weights
    }
    counts = {feature: Counter() for feature in feature_weights}
    selected = []
    remaining = list(records)
    while len(selected) < count:
        next_size = len(selected) + 1

        def imbalance(record):
            total = 0.0
            for feature, weight in feature_weights.items():
                target = next_size / len(categories[feature])
                total += weight * sum(
                    (
                        counts[feature][category]
                        + (category == record[feature])
                        - target
                    )
                    ** 2
                    for category in categories[feature]
                )
            return total, record["path"]

        chosen = min(remaining, key=imbalance)
        selected.append(chosen["path"])
        remaining.remove(chosen)
        for feature in feature_weights:
            counts[feature][chosen[feature]] += 1
    return sorted(selected)


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def _denormalize(action):
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1, 1)
    raw = (action + 1) * 0.5 * (ACTION_UPPER - ACTION_LOWER + 1e-8) + ACTION_LOWER
    return raw[None, :]


class DrawerMemoryEnv(gymnasium.Env):
    """Four-cube visual-memory retrieval environment.

    The base cabinet simulator is unchanged. This wrapper replaces its
    drawer-open termination with target-placement success and adds the visual
    cue, task rewards, wrong-object termination, and a 2,000-step timeout.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        seed: int = 0,
        image_size: int = 128,
        cue_steps: int = 10,
        max_control_steps: int = 2000,
        step_penalty: float = -0.0001,
    ):
        super().__init__()
        self._seed = int(seed)
        self._image_size = int(image_size)
        self._cue_steps = int(cue_steps)
        self._max_control_steps = int(max_control_steps)
        self._step_penalty = float(step_penalty)
        self._episode_index = 0
        self._ms_env = None
        self._build_env()

    def _build_env(self):
        import mani_skill.envs  # noqa: F401

        if self._ms_env is not None:
            self._ms_env.close()
        self._ms_env = gymnasium.make(
            "OpenCabinetPanda-v0",
            render_mode="rgb_array",
            reward_mode="sparse",
            control_mode="pd_joint_pos",
            robot_uids="panda",
            shuffle_colors=True,
            color_shuffle_seed=self._seed,
            num_cubes=4,
            max_episode_steps=self._max_control_steps + 1,
            render_backend="gpu",
            width=self._image_size,
            height=self._image_size,
            randomize_cabinet_pose=False,
        )
        self._ms_env.reset(seed=self._seed)
        sample = self._raw_obs()
        self.observation_space = DictSpace(
            {
                "agent_view": Box(0, 255, sample["agent_view"].shape, np.uint8),
                "wrist_view": Box(0, 255, sample["wrist_view"].shape, np.uint8),
                "proprio": Box(-np.inf, np.inf, sample["proprio"].shape, np.float32),
            }
        )
        self.action_space = Box(-1.0, 1.0, (8,), np.float32)

    def _raw_obs(self):
        base = self._ms_env.unwrapped
        return {
            "agent_view": base.render_rgb_array("render_camera").cpu().numpy()[0].astype(np.uint8),
            "wrist_view": base.render_rgb_array("wrist_camera").cpu().numpy()[0].astype(np.uint8),
            "proprio": base.agent.robot.get_qpos().cpu().numpy().reshape(-1).astype(np.float32),
        }

    def _obs(self):
        obs = self._raw_obs()
        if self._step <= self._cue_steps:
            alpha = max(0.0, 1.0 - self._step / max(self._cue_steps, 1))
            obs["agent_view"] = add_cube_cue(obs["agent_view"], self.target_color, alpha)
            obs["wrist_view"] = add_cube_cue(obs["wrist_view"], self.target_color, alpha)
        return obs

    def _drawer_open(self, cube_index: int) -> bool:
        base = self._ms_env.unwrapped
        cabinet_index, drawer_type, _ = base.cube_slot_assignments[cube_index]
        drawer_rank = 0 if drawer_type == "top" else 1
        link = base._all_cabinet_drawer_links[cabinet_index][drawer_rank]
        limits = link.joint.limits
        threshold = limits[..., 0] + (limits[..., 1] - limits[..., 0]) * base.min_open_frac
        return bool(_scalar((link.joint.qpos >= threshold).reshape(-1)[0]))

    def _is_grasping(self, cube_index: int) -> bool:
        value = self._ms_env.unwrapped.agent.is_grasping(
            self._ms_env.unwrapped.cubes[cube_index]
        )
        return bool(_scalar(value.reshape(-1)[0]))

    def _target_on_table(self) -> bool:
        # Cabinet cube slots are behind y≈0.1; the scripted table placement is
        # around y≈0.7. Requiring a prior grasp prevents accidental success.
        pos = (
            self._ms_env.unwrapped.get_cube_pose(self.target_cube_index)
            .p[0]
            .cpu()
            .numpy()
        )
        return bool(
            self._target_was_grasped
            and not self._is_grasping(self.target_cube_index)
            and -0.65 < pos[0] < 0.65
            and 0.50 < pos[1] < 1.05
            and 0.12 < pos[2] < 0.55
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = int(seed)
        else:
            self._seed += 1
        # Rebuild so both cube colors and slot assignments vary by episode.
        self._build_env()
        self._episode_index += 1
        self._step = 0
        self.target_cube_index = 0
        base = self._ms_env.unwrapped
        self.target_color = str(base.cube_colors[self.target_cube_index])
        cab, drawer, _ = base.cube_slot_assignments[self.target_cube_index]
        self.target_drawer = f"{drawer}_{cab + 1}"
        self._target_open_rewarded = False
        self._target_was_grasped = False
        return self._obs(), {
            "success": False,
            "fail": False,
            "target_color": self.target_color,
            "target_drawer": self.target_drawer,
        }

    def step(self, action):
        self._step += 1
        _, _, _, _, raw_info = self._ms_env.step(_denormalize(action))
        reward = self._step_penalty
        success = False
        fail = False
        failure_reason = ""

        target_open = self._drawer_open(self.target_cube_index)
        if target_open and not self._target_open_rewarded:
            reward += 0.25
            self._target_open_rewarded = True

        for cube_index in range(1, 4):
            if self._is_grasping(cube_index):
                fail = True
                failure_reason = "wrong_object_grasp"
                break

        if self._is_grasping(self.target_cube_index):
            self._target_was_grasped = True
        if not fail and self._target_on_table():
            success = True
            reward = 0.75
        elif not fail and self._target_was_grasped and not self._is_grasping(self.target_cube_index):
            fail = True
            failure_reason = "target_dropped"
        elif not fail and self._step >= self._max_control_steps:
            fail = True
            failure_reason = "timeout"

        if fail:
            reward = -1.0
        info = {}
        for key, value in raw_info.items():
            try:
                info[key] = _scalar(value)
            except Exception:
                info[key] = value
        info.update(
            success=success,
            fail=fail,
            failure_reason=failure_reason,
            target_open=target_open,
            target_color=self.target_color,
            target_drawer=self.target_drawer,
        )
        return self._obs(), float(reward), bool(success or fail), False, info

    def render(self):
        return self._ms_env.unwrapped.render_rgb_array().cpu().numpy()[0].astype(np.uint8)

    def close(self):
        if self._ms_env is not None:
            self._ms_env.close()
            self._ms_env = None


def make_drawer_env(seed=0, image_size=128):
    return DrawerMemoryEnv(seed=seed, image_size=image_size)


def make_drawer_env_and_datasets(
    hist_length,
    hist_stride,
    online_buf_size,
    ep_cache_size,
    chunk_reload_interval,
    action_chunk_size,
    discount,
    successful_demos_only=False,
    num_success_demos=-1,
    num_failure_demos=-1,
):
    episode_dir = DATASET_DIR / "episodes"
    success_paths = sorted(glob.glob(str(episode_dir / "success_*.npz")))
    failure_paths = sorted(glob.glob(str(episode_dir / "failure_*.npz")))
    if len(success_paths) != 300 or len(failure_paths) != 50:
        raise RuntimeError(
            f"drawer dataset incomplete: found {len(success_paths)} successes and "
            f"{len(failure_paths)} failures; expected 300 and 50"
        )
    if num_success_demos < -1 or num_failure_demos < -1:
        raise ValueError("num_success_demos and num_failure_demos must be >= -1")
    if successful_demos_only and num_failure_demos not in (-1, 0):
        raise ValueError(
            "successful_demos_only cannot be combined with a positive "
            "num_failure_demos"
        )

    requested_successes = (
        len(success_paths) if num_success_demos == -1 else num_success_demos
    )
    requested_failures = (
        0
        if successful_demos_only
        else len(failure_paths) if num_failure_demos == -1 else num_failure_demos
    )
    if requested_successes > len(success_paths):
        raise ValueError(
            f"requested {requested_successes} successful drawer episodes, but only "
            f"{len(success_paths)} are available"
        )
    if requested_failures > len(failure_paths):
        raise ValueError(
            f"requested {requested_failures} failed drawer episodes, but only "
            f"{len(failure_paths)} are available"
        )

    metadata = _episode_metadata()
    selected_success_paths = _balanced_episode_subset(
        success_paths, requested_successes, True, metadata
    )
    selected_failure_paths = _balanced_episode_subset(
        failure_paths, requested_failures, False, metadata
    )
    train_paths = selected_success_paths + selected_failure_paths
    if not train_paths:
        raise ValueError("drawer training selection is empty")
    print(
        "[drawer_task] train: selected "
        f"{len(selected_success_paths)} successful + "
        f"{len(selected_failure_paths)} failed episodes; source files unchanged",
        flush=True,
    )
    kwargs = dict(
        hist_length=hist_length,
        hist_stride=hist_stride,
        ep_cache_size=min(ep_cache_size, len(train_paths)),
        chunk_reload_interval=chunk_reload_interval,
        action_chunk_size=action_chunk_size,
        discount=discount,
    )
    train = LazyEpisodeReplayBuffer(
        train_paths, online_buf_size=online_buf_size, **kwargs
    )
    # Validation losses are currently disabled in m_main; retain a small,
    # deterministic buffer for interface compatibility.
    val_paths = success_paths[-8:] + ([] if successful_demos_only else failure_paths[-2:])
    val_kwargs = dict(kwargs)
    val_kwargs["ep_cache_size"] = len(val_paths)
    val = LazyEpisodeReplayBuffer(val_paths, online_buf_size=0, **val_kwargs)
    env = make_drawer_env(seed=0)
    eval_env = make_drawer_env(seed=10_000)
    return env, eval_env, train, val
