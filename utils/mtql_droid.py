"""DROID-to-MTQL data conversion and augmentation utilities.

This module keeps EXPO-FT's two real camera inputs while presenting them in
MTQL's native observation structure.  Images remain uint8; only proprioception
and actions pass through OpenPI normalization.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import h5py
import jax
import numpy as np
from tqdm import tqdm

from utils.datasets import HistoryDataset


DROID_BASE_IMAGE_KEY = "exterior_image_1_left"
DROID_WRIST_IMAGE_KEY = "wrist_image_left"
DROID_CARTESIAN_POSITION_KEY = "cartesian_position"
DROID_GRIPPER_POSITION_KEY = "gripper_position"

MTQL_BASE_IMAGE_KEY = "image"
MTQL_WRIST_IMAGE_KEY = "wrist_image"
MTQL_PROPRIO_KEY = "proprio"
MTQL_IMAGE_KEYS = (MTQL_BASE_IMAGE_KEY, MTQL_WRIST_IMAGE_KEY)
DEFAULT_IMAGE_SIZE = 224
DROID_CACHE_FORMAT_VERSION = 1
EGG_CACHE_FORMAT_VERSION = 1


def _discover_episode_dirs(base_path: str) -> list[str]:
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"DROID dataset directory not found: {base_path}")
    episode_dirs = []
    for root, _, files in os.walk(base_path):
        if "traj.hdf5" in files and os.path.basename(root).isdigit():
            episode_dirs.append(root)
    episode_dirs.sort(
        key=lambda path: tuple(
            int(part) if part.isdigit() else part
            for part in Path(path).relative_to(base_path).parts
        )
    )
    return episode_dirs


def _episode_manifest(base_path: str) -> tuple[str, int]:
    """Fingerprint finalized episodes so a full-data cache cannot go stale."""
    entries = []
    for episode_dir in _discover_episode_dirs(base_path):
        trajectory_path = os.path.join(episode_dir, "traj.hdf5")
        stat = os.stat(trajectory_path)
        entries.append(
            {
                "path": os.path.relpath(trajectory_path, base_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest, len(entries)


_TARGET_NUMBERS = (1, 2, 3)


def _episode_outcome(path: str) -> str:
    """Return the labeled outcome encoded in an episode path."""
    parts = {part.lower() for part in Path(path).parts}
    if "success" in parts:
        return "success"
    if "failure" in parts:
        return "failure"
    raise ValueError(
        "Could not infer episode outcome; expected a success or failure "
        f"directory: {path}"
    )


def _episode_card_color(path: str, base_path: str) -> str:
    """Return the card-color label encoded in an Egg episode path."""
    relative_parts = Path(path).relative_to(base_path).parts
    if not relative_parts or relative_parts[0] not in {
        "card_white",
        "card_black",
        "card_orange",
    }:
        raise ValueError(
            "Egg episodes must be under card_white or card_black: "
            f"{path}"
        )
    return relative_parts[0]


def _select_egg_episode_dirs(
    episode_dirs: Sequence[str],
    base_path: str,
    *,
    n_success: int = -1,
    n_failure: int = -1,
) -> list[str]:
    """Select a deterministic, color-balanced Egg subset.

    For each requested outcome, ``n`` is split as evenly as possible across
    the card colors present. For the original two-color Egg data, odd totals
    assign the extra success to ``card_white`` and the extra failure to
    ``card_black``. With additional colors, remaining episodes are assigned
    in the same deterministic color order. Episode IDs are sorted numerically,
    so selection is independent of filesystem traversal and RNG seeds. ``-1``
    means use every episode for that outcome.
    """
    for name, value in (("n_success", n_success), ("n_failure", n_failure)):
        if value < -1:
            raise ValueError(f"{name} must be -1 or nonnegative, got {value}.")

    grouped: dict[tuple[str, str], list[str]] = {}
    for path in episode_dirs:
        outcome = _episode_outcome(path)
        color = _episode_card_color(path, base_path)
        grouped.setdefault((outcome, color), []).append(path)

    for paths in grouped.values():
        paths.sort(key=lambda path: int(Path(path).name))

    colors = tuple(
        color
        for color in ("card_white", "card_black", "card_orange")
        if any(group_color == color for _, group_color in grouped)
    )
    if not colors:
        raise ValueError(f"No supported card colors found under {base_path}.")

    selected_paths: set[str] = set()
    requested_by_outcome = {
        "success": n_success,
        "failure": n_failure,
    }
    for outcome, requested in requested_by_outcome.items():
        if requested == -1:
            for color in colors:
                selected_paths.update(grouped.get((outcome, color), []))
            continue

        base, remainder = divmod(requested, len(colors))
        preferred_color = "card_white" if outcome == "success" else "card_black"
        allocation_order = (
            (preferred_color,) + tuple(c for c in colors if c != preferred_color)
            if preferred_color in colors
            else colors
        )
        quotas = {color: base for color in colors}
        for color in allocation_order[:remainder]:
            quotas[color] += 1
        for color in colors:
            quota = quotas[color]
            candidates = grouped.get((outcome, color), [])
            if len(candidates) < quota:
                raise ValueError(
                    f"Requested {quota} {outcome} episodes from {color}, "
                    f"but only {len(candidates)} are available."
                )
            selected_paths.update(candidates[:quota])

    # Preserve the cache's canonical episode order while selecting by the
    # explicit numeric-ID rule above.
    selected = [path for path in episode_dirs if path in selected_paths]
    requested_total = sum(
        value for value in (n_success, n_failure) if value >= 0
    )
    for outcome in ("success", "failure"):
        for color in ("card_white", "card_black"):
            selected_ids = [
                int(Path(path).name)
                for path in selected
                if _episode_outcome(path) == outcome
                and _episode_card_color(path, base_path) == color
            ]
            print(
                f"[egg-data] selected {color}/{outcome}: "
                f"len={len(selected_ids)} ids={selected_ids}",
                flush=True,
            )
    if n_success >= 0 or n_failure >= 0:
        print(
            "[egg-data] deterministic subset: "
            f"requested={requested_total} episodes, selected={len(selected)} "
            f"(n_succ={n_success}, n_fails={n_failure}; "
            "near-even card-color split)",
            flush=True,
        )
    return selected


def _episode_target_number(path: str, base_path: str) -> int:
    """Return target number for target_N or arbitrary/target_num_N layouts."""
    relative_parts = Path(path).relative_to(base_path).parts
    if not relative_parts:
        raise ValueError(f"Episode path is outside dataset root: {path}")
    group = relative_parts[0]
    if group.startswith("target_") and group[len("target_") :].isdigit():
        target_number = int(group[len("target_") :])
    elif group == "arbitrary" and len(relative_parts) > 1:
        target_group = relative_parts[1]
        prefix = "target_num_"
        if target_group.startswith(prefix) and target_group[len(prefix) :].isdigit():
            target_number = int(target_group[len(prefix) :])
        else:
            raise ValueError(
                "Arbitrary episode must be under arbitrary/target_num_N: "
                f"{path}"
            )
    else:
        raise ValueError(
            "Episode must be under target_N or arbitrary/target_num_N: "
            f"{path}"
        )
    if target_number not in _TARGET_NUMBERS:
        raise ValueError(
            f"Unsupported target number {target_number} in episode path: {path}"
        )
    return target_number


def _round_robin(paths_by_source: Mapping[str, Sequence[str]]) -> list[str]:
    """Interleave sources in stable order, falling back when one is empty."""
    sources = tuple(sorted(paths_by_source))
    ordered = []
    index = 0
    while True:
        added = False
        for source in sources:
            paths = paths_by_source[source]
            if index < len(paths):
                ordered.append(paths[index])
                added = True
        if not added:
            return ordered
        index += 1


def _balanced_quotas(total: int, groups: Sequence[int]) -> dict[int, int]:
    """Split a total deterministically, assigning remainders in group order."""
    if total < 0:
        raise ValueError(f"Expected a nonnegative total, got {total}.")
    base, remainder = divmod(total, len(groups))
    return {
        group: base + int(index < remainder)
        for index, group in enumerate(groups)
    }


def _select_droid_episode_dirs(
    episode_dirs: Sequence[str],
    base_path: str,
    *,
    n_success: int = -1,
    n_failure: int = -1,
) -> list[str]:
    """Select a deterministic, target-balanced subset of DROID episodes.

    Totals are distributed as evenly as possible across target counts 1, 2,
    and 3, with remainders assigned in ascending target order. Fixed-target
    and arbitrary-target failures are interleaved so arbitrary failures are
    included rather than hidden behind the fixed directory's sort order.
    ``-1`` means load all episodes for that outcome.
    """
    for name, value in (("n_success", n_success), ("n_failure", n_failure)):
        if value < -1:
            raise ValueError(f"{name} must be -1 or nonnegative, got {value}.")

    by_outcome_target: dict[tuple[str, int], dict[str, list[str]]] = {}
    for path in episode_dirs:
        outcome = _episode_outcome(path)
        target_number = _episode_target_number(path, base_path)
        relative_parts = Path(path).relative_to(base_path).parts
        source = "arbitrary" if relative_parts[0] == "arbitrary" else "fixed"
        by_outcome_target.setdefault((outcome, target_number), {}).setdefault(
            source, []
        ).append(path)

    def path_key(path: str):
        return tuple(
            int(part) if part.isdigit() else part
            for part in Path(path).relative_to(base_path).parts
        )

    for source_paths in by_outcome_target.values():
        for paths in source_paths.values():
            paths.sort(key=path_key)

    selected: list[str] = []
    for outcome, requested in (("success", n_success), ("failure", n_failure)):
        if requested == -1:
            for target_number in _TARGET_NUMBERS:
                selected.extend(
                    _round_robin(
                        by_outcome_target.get((outcome, target_number), {})
                    )
                )
            continue

        quotas = _balanced_quotas(requested, _TARGET_NUMBERS)
        outcome_start = len(selected)
        for target_number in _TARGET_NUMBERS:
            candidates = _round_robin(
                by_outcome_target.get((outcome, target_number), {})
            )
            selected.extend(candidates[: quotas[target_number]])
        selected_outcome = len(selected) - outcome_start
        if selected_outcome != requested:
            raise ValueError(
                f"Requested {requested} {outcome} episodes, but only "
                f"{selected_outcome} are available with balanced target counts."
            )
    return selected

def _load_droid_group(group: h5py.Group) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            result[key] = _load_droid_group(value)
            continue
        array = np.asarray(value)
        if array.dtype.kind == "S":
            result[key] = array.astype("U")
        elif array.dtype == object and array.size and isinstance(array.flat[0], bytes):
            result[key] = np.array(
                [item.decode("utf-8") for item in array.flat]
            ).reshape(array.shape)
        else:
            result[key] = array
    return result


def _droid_timestep(value: Any, index: int) -> Any:
    if isinstance(value, dict):
        return {key: _droid_timestep(item, index) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value[index]
    return value


def process_droid_dataset(
    datapath: str,
    task_config: Any,
    n_success: int = -1,
    n_failure: int = -1,
) -> list[dict[str, Any]]:
    """Load raw EXPO DROID HDF5 episodes for offline MTQL training."""
    episode_dirs = _discover_episode_dirs(datapath)
    episode_dirs = _select_droid_episode_dirs(
        episode_dirs,
        datapath,
        n_success=n_success,
        n_failure=n_failure,
    )

    print(
        f"Found {len(_discover_episode_dirs(datapath))} episodes; "
        f"using {len(episode_dirs)}"
    )
    action_key = task_config.action_space
    gripper_key = f"gripper_{task_config.gripper_action_space}"
    transitions: list[dict[str, Any]] = []

    for episode_dir in tqdm(episode_dirs, desc="Loading DROID episodes"):
        trajectory_path = os.path.join(episode_dir, "traj.hdf5")
        with h5py.File(trajectory_path, "r") as handle:
            observations = _load_droid_group(handle["saved_observation"])
            action_group = handle["action"]
            cartesian = np.asarray(action_group[action_key])
            gripper = np.asarray(action_group[gripper_key])
            if gripper.ndim == 1:
                gripper = gripper[:, None]
            actions = np.concatenate([cartesian, gripper], axis=-1)

            length = len(actions)
            if length == 0:
                continue
            terminal_reward = (
                1.0 if _episode_outcome(episode_dir) == "success" else 0.0
            )
            episode_success = bool(terminal_reward > 0.0)
            cue_target = _episode_target_number(episode_dir, datapath)
            terminals = np.zeros(length, dtype=np.float32)
            terminals[-1] = 1.0
            for index in range(length):
                transitions.append(
                    {
                        "observations": _droid_timestep(observations, index),
                        "actions": actions[index],
                        "rewards": terminal_reward * terminals[index],
                        "masks": 1.0 - terminals[index],
                        "dones": terminals[index],
                        "episode_success": episode_success,
                        # The path is authoritative, including for
                        # counterfactual failure trajectories.
                        "cue_target": cue_target,
                    }
                )

    if not transitions:
        raise ValueError(f"No transitions found under {datapath}.")
    return transitions


def process_egg_dataset(
    datapath: str,
    task_config: Any,
    n_success: int = -1,
    n_failure: int = -1,
) -> list[dict[str, Any]]:
    """Load every finalized egg episode without candy-specific conditioning.

    Egg episodes are discovered using the same finalized-episode rule as the
    candy data: the directory containing ``traj.hdf5`` must have a numeric
    name.  This deliberately excludes ``tmp/session_*`` recordings.  The
    black/white card and success/failure directory names are not model inputs.
    The outcome is retained as transition metadata so RL can train on every
    finalized episode while a pure-BC trainer can sample successes only.

    Only the observation arrays consumed by MTQL are read from HDF5.  This
    avoids decoding the unused second exterior camera, calibration matrices,
    and collection metadata present in ``saved_observation``.
    """
    all_episode_dirs = _discover_episode_dirs(datapath)
    episode_dirs = _select_egg_episode_dirs(
        all_episode_dirs,
        datapath,
        n_success=n_success,
        n_failure=n_failure,
    )
    print(
        f"Found {len(all_episode_dirs)} finalized egg episodes; "
        f"using {len(episode_dirs)}",
        flush=True,
    )
    action_key = task_config.action_space
    gripper_key = f"gripper_{task_config.gripper_action_space}"
    observation_keys = (
        DROID_BASE_IMAGE_KEY,
        DROID_WRIST_IMAGE_KEY,
        DROID_CARTESIAN_POSITION_KEY,
        DROID_GRIPPER_POSITION_KEY,
    )
    transitions: list[dict[str, Any]] = []

    for episode_dir in tqdm(episode_dirs, desc="Loading egg episodes"):
        trajectory_path = os.path.join(episode_dir, "traj.hdf5")
        with h5py.File(trajectory_path, "r") as handle:
            observation_group = handle["saved_observation"]
            missing_observations = [
                key for key in observation_keys if key not in observation_group
            ]
            if missing_observations:
                raise KeyError(
                    f"{trajectory_path} is missing required observations: "
                    f"{missing_observations}"
                )
            observations = {
                key: np.asarray(observation_group[key])
                for key in observation_keys
            }
            action_group = handle["action"]
            cartesian = np.asarray(action_group[action_key])
            gripper = np.asarray(action_group[gripper_key])
            if gripper.ndim == 1:
                gripper = gripper[:, None]
            actions = np.concatenate([cartesian, gripper], axis=-1)

            lengths = {
                "actions": len(actions),
                **{key: len(value) for key, value in observations.items()},
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"Mismatched trajectory lengths in {trajectory_path}: "
                    f"{lengths}"
                )
            length = len(actions)
            if length == 0:
                continue

            # The directory label is more reliable than stale HDF5 attrs in
            # the collected egg data.  It is never a policy input; it supplies
            # the RL reward and success-only BC sampling metadata.
            terminal_reward = (
                1.0 if _episode_outcome(episode_dir) == "success" else 0.0
            )
            terminals = np.zeros(length, dtype=np.float32)
            terminals[-1] = 1.0
            for index in range(length):
                transitions.append(
                    {
                        "observations": {
                            key: value[index]
                            for key, value in observations.items()
                        },
                        "actions": actions[index],
                        "rewards": terminal_reward * terminals[index],
                        "masks": 1.0 - terminals[index],
                        "dones": terminals[index],
                        "episode_success": bool(terminal_reward),
                    }
                )

    if not transitions:
        raise ValueError(f"No finalized egg transitions found under {datapath}.")
    print(
        f"[egg-data] loaded transitions: len={len(transitions)} "
        f"from episodes={len(episode_dirs)}",
        flush=True,
    )
    return transitions


@dataclass(frozen=True)
class _NormStats:
    """The subset of OpenPI's NormStats needed by MTQL."""

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None
    q99: np.ndarray | None


def _load_norm_stats_json(path: Path) -> dict[str, _NormStats]:
    """Read OpenPI's ``norm_stats.json`` without importing OpenPI.

    OpenPI's JSON format is intentionally simple (a ``norm_stats`` mapping of
    arrays), so parsing it here keeps training independent of OpenPI's runtime
    imports while remaining compatible with EXPO/OpenPI statistics.
    """
    payload = json.loads(path.read_text())
    raw_stats = payload.get("norm_stats", payload)
    if not isinstance(raw_stats, dict):
        raise ValueError(f"Expected a norm_stats mapping in {path}.")

    stats = {}
    for key, raw in raw_stats.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid statistics for {key!r} in {path}.")
        missing = [name for name in ("mean", "std") if name not in raw]
        if missing:
            raise KeyError(f"Statistics for {key!r} are missing {missing}.")
        stats[key] = _NormStats(
            mean=np.asarray(raw["mean"], dtype=np.float32),
            std=np.asarray(raw["std"], dtype=np.float32),
            q01=(None if raw.get("q01") is None else np.asarray(raw["q01"], dtype=np.float32)),
            q99=(None if raw.get("q99") is None else np.asarray(raw["q99"], dtype=np.float32)),
        )
    return stats


def load_openpi_norm_stats(norm_stats_path: str | Path):
    """Load the exact OpenPI ``norm_stats.json`` at or below ``norm_stats_path``."""
    path = Path(norm_stats_path)
    directory = path.parent if path.name == "norm_stats.json" else path
    stats_path = directory / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {stats_path}")
    return _load_norm_stats_json(stats_path)


class OpenPINormalizer:
    """Apply OpenPI PI0.5 quantile normalization to MTQL state and actions."""

    def __init__(self, norm_stats: Mapping[str, Any]):
        if "state" not in norm_stats or "actions" not in norm_stats:
            raise KeyError("OpenPI norm stats must contain 'state' and 'actions'.")

        self.norm_stats = norm_stats
        for key in ("state", "actions"):
            if norm_stats[key].q01 is None or norm_stats[key].q99 is None:
                raise ValueError(
                    f"Quantile statistics q01/q99 are required for {key!r}."
                )

    @classmethod
    def from_path(cls, norm_stats_path: str | Path) -> "OpenPINormalizer":
        return cls(load_openpi_norm_stats(norm_stats_path))

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        return self._normalize_quantile(state, self.norm_stats["state"])

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        return self._normalize_quantile(actions, self.norm_stats["actions"])

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Invert action normalization and return the original action dimension."""
        actions = np.asarray(actions, dtype=np.float32)
        output_dim = actions.shape[-1]
        stats = self.norm_stats["actions"]
        assert stats.q01 is not None and stats.q99 is not None
        stats_dim = stats.q01.shape[-1]
        if output_dim < stats_dim:
            padding = [(0, 0)] * actions.ndim
            padding[-1] = (0, stats_dim - output_dim)
            actions = np.pad(actions, padding)
        q01, q99 = stats.q01, stats.q99
        raw_actions = (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return np.asarray(raw_actions[..., :output_dim], dtype=np.float32)

    @staticmethod
    def _normalize_quantile(values: np.ndarray, stats: _NormStats) -> np.ndarray:
        assert stats.q01 is not None and stats.q99 is not None
        q01 = stats.q01[..., : values.shape[-1]]
        q99 = stats.q99[..., : values.shape[-1]]
        return np.asarray(
            (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0,
            dtype=np.float32,
        )


def _parse_droid_image(image: Any) -> np.ndarray:
    """Match OpenPI's DROID image parsing while preserving MTQL's uint8 input."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}.")
    return image.astype(np.uint8, copy=False)


def _resize_droid_image(image: Any, image_size: int) -> np.ndarray:
    """Match OpenPI's PIL ``resize_with_pad`` implementation exactly."""
    if image_size < 1:
        raise ValueError(f"image_size must be positive, got {image_size}.")

    from PIL import Image

    image = _parse_droid_image(image)
    if image.shape[-3:-1] == (image_size, image_size):
        return image

    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / image_size, cur_height / image_size)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = Image.fromarray(image).resize(
        (resized_width, resized_height), resample=Image.BILINEAR
    )
    canvas = Image.new(resized.mode, (image_size, image_size), 0)
    pad_height = max(0, int((image_size - resized_height) / 2))
    pad_width = max(0, int((image_size - resized_width) / 2))
    canvas.paste(resized, (pad_width, pad_height))
    return np.asarray(canvas, dtype=np.uint8)


def convert_droid_observation(
    observation: Mapping[str, Any],
    normalizer: OpenPINormalizer,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    target_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Map one raw EXPO DROID observation to an MTQL observation.

    ``target_count`` is used only for inference-time visual conditioning.
    Training draws the same cue in ``add_droid_target_cues`` after image
    augmentation; applying it here keeps live evaluation aligned with those
    training inputs.
    """
    cartesian_position = np.asarray(
        observation[DROID_CARTESIAN_POSITION_KEY], dtype=np.float32
    ).reshape(-1)
    gripper_position = np.asarray(
        observation[DROID_GRIPPER_POSITION_KEY], dtype=np.float32
    ).reshape(-1)
    state = np.concatenate([cartesian_position, gripper_position], axis=-1)

    base_image = _resize_droid_image(
        observation[DROID_BASE_IMAGE_KEY], image_size
    )
    if target_count is not None:
        # _draw_target_cue operates on (T,H,W,3), matching the training path.
        base_image = np.array(base_image, copy=True)
        _draw_target_cue(base_image[None], int(target_count))

    return {
        MTQL_BASE_IMAGE_KEY: base_image,
        MTQL_WRIST_IMAGE_KEY: _resize_droid_image(
            observation[DROID_WRIST_IMAGE_KEY], image_size
        ),
        MTQL_PROPRIO_KEY: normalizer.normalize_state(state),
    }


def _stack_observations(observations: Sequence[Mapping[str, np.ndarray]]):
    if not observations:
        raise ValueError("At least one DROID observation is required.")
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


class DroidHistoryDataset(HistoryDataset):
    """HistoryDataset with episode-safe DROID action-chunk targets."""

    def sample_actor_batch(self, batch_size: int, idxs=None):
        """Sample only the fields consumed by the MTQL actor loss.

        The critic needs a complete transition, including the bootstrap
        observation and its history.  The actor only consumes the current
        observation, current history, and demonstrated action chunk.  Keeping
        this path separate avoids gathering and copying the actor batch's
        unused next-state image tensors.

        This method is intentionally opt-in.  The regular ``sample`` method
        and all existing critic/BC training paths retain their current
        behavior.
        """
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.ndim != 1 or len(idxs) != batch_size:
            raise ValueError(
                "Actor sample indices must be one-dimensional with length "
                f"{batch_size}, got shape {idxs.shape}."
            )

        episode_positions = np.searchsorted(
            self.initial_locs,
            idxs,
            side="right",
        ) - 1
        episode_starts = self.initial_locs[episode_positions]
        terminal_positions = np.searchsorted(
            self.terminal_locs,
            idxs,
            side="left",
        )
        episode_ends = self.terminal_locs[terminal_positions]

        action_chunk_size = int(self.action_chunk_size or 1)
        action_offsets = np.arange(action_chunk_size, dtype=np.int64)
        action_indices = np.minimum(
            idxs[:, None] + action_offsets[None, :],
            episode_ends[:, None],
        )

        batch = {
            "observations": jax.tree_util.tree_map(
                lambda array: array[idxs], self["observations"]
            ),
            "actions": self["actions"][action_indices],
        }
        if "episode_success" in self:
            batch["episode_success"] = self["episode_success"][idxs]

        if self.frame_stack is not None:
            current_frames = []
            for frame_offset in reversed(range(self.frame_stack)):
                frame_indices = np.maximum(
                    idxs - frame_offset,
                    episode_starts,
                )
                current_frames.append(
                    jax.tree_util.tree_map(
                        lambda array: array[frame_indices],
                        self["observations"],
                    )
                )
            batch["observations"] = jax.tree_util.tree_map(
                lambda *frames: np.concatenate(frames, axis=-1),
                *current_frames,
            )

        if self.hist_length > 0:
            if self.frame_stack is None:
                history_offsets = (
                    np.arange(self.hist_length, dtype=np.int64)
                    - self.hist_length
                ) * self.hist_stride
                history_indices = np.maximum(
                    idxs[:, None] + history_offsets[None, :],
                    episode_starts[:, None],
                )
                batch["history_observations"] = (
                    jax.tree_util.tree_map(
                        lambda array: array[history_indices],
                        self["observations"],
                    )
                )
            else:
                raw_history, _, _ = self._get_history(
                    idxs,
                    episode_starts,
                )
                raw_history = self._apply_frame_stack(
                    raw_history,
                    idxs,
                    episode_starts,
                )
                batch["history_observations"] = self._stack_history(
                    raw_history
                )

        return batch

    def _get_next_history_indices(self, idxs, action_chunk_size):
        """Clamp Bellman-target history anchors to each current episode."""
        terminal_indices = np.searchsorted(
            self.terminal_locs,
            idxs,
            side="left",
        )
        episode_ends = self.terminal_locs[terminal_indices]
        return np.minimum(idxs + action_chunk_size, episode_ends)

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        idxs = np.asarray(idxs, dtype=np.int64)
        batch = super().sample(batch_size, idxs=idxs)

        action_chunk_size = int(self.action_chunk_size or 1)
        if action_chunk_size == 1:
            batch["actions"] = batch["actions"][:, None, :]

        terminal_indices = np.searchsorted(
            self.terminal_locs,
            idxs,
            side="left",
        )
        episode_ends = self.terminal_locs[terminal_indices]

        offsets = np.arange(action_chunk_size, dtype=np.int64)
        raw_chunk_indices = idxs[:, None] + offsets[None, :]
        valid_steps = raw_chunk_indices <= episode_ends[:, None]
        chunk_indices = np.minimum(
            raw_chunk_indices,
            episode_ends[:, None],
        )

        # Derive this from the single stored observation stream.  Besides
        # making the episode boundary semantics explicit, this lets the egg
        # cache avoid a second ~10 GB copy of current images under a
        # ``next_observations`` name.
        next_idxs = np.minimum(
            idxs + action_chunk_size,
            episode_ends,
        )
        batch["next_observations"] = {
            key: value[next_idxs]
            for key, value in self["observations"].items()
        }

        chunk_rewards = self["rewards"][chunk_indices]
        discount_weights = self.discount ** offsets.astype(np.float32)
        batch["rewards"] = np.sum(
            chunk_rewards * valid_steps * discount_weights[None, :],
            axis=1,
        ).astype(np.float32)

        chunk_hits_terminal = (
            idxs + action_chunk_size - 1 >= episode_ends
        )
        batch["masks"] = np.where(
            chunk_hits_terminal,
            0.0,
            batch["masks"],
        ).astype(np.float32)

        return batch


def create_droid_history_dataset(
    transitions: Sequence[Mapping[str, Any]],
    normalizer: OpenPINormalizer,
    *,
    hist_length: int,
    hist_stride: int = 1,
    action_chunk_size: int = 1,
    discount: float = 0.99,
    image_size: int = DEFAULT_IMAGE_SIZE,
    store_next_observations: bool = True,
) -> DroidHistoryDataset:
    """Convert raw ``process_droid_dataset`` transitions to ``HistoryDataset``."""
    if not transitions:
        raise ValueError("Cannot create an MTQL dataset from zero transitions.")
    if action_chunk_size < 1:
        raise ValueError(
            "action_chunk_size must be positive, got "
            f"{action_chunk_size}."
        )

    observations = [
        convert_droid_observation(
            transition["observations"],
            normalizer,
            image_size=image_size,
        )
        for transition in transitions
    ]
    terminals = np.asarray(
        [transition.get("terminals", transition.get("dones", 0.0)) for transition in transitions],
        dtype=np.float32,
    )
    if not bool(terminals[-1]):
        raise ValueError(
            "The final DROID transition must terminate its episode."
        )

    actions = normalizer.normalize_actions(
        np.stack([np.asarray(transition["actions"]) for transition in transitions])
    )
    rewards = np.asarray(
        [transition.get("rewards", 0.0) for transition in transitions], dtype=np.float32
    )
    masks = np.asarray(
        [transition.get("masks", 1.0 - terminals[i]) for i, transition in enumerate(transitions)],
        dtype=np.float32,
    )
    fields = dict(
        hist_length=hist_length,
        hist_stride=hist_stride,
        observations=_stack_observations(observations),
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        masks=masks,
    )
    if store_next_observations:
        next_observations = []
        for index, observation in enumerate(observations):
            has_same_episode_successor = (
                index + 1 < len(observations) and not bool(terminals[index])
            )
            next_observations.append(
                observations[index + 1]
                if has_same_episode_successor
                else observation
            )
        fields["next_observations"] = _stack_observations(next_observations)
    has_cue_targets = ["cue_target" in transition for transition in transitions]
    if any(has_cue_targets) and not all(has_cue_targets):
        raise ValueError("cue_target must be present on every transition or none.")
    if all(has_cue_targets):
        fields["cue_targets"] = np.asarray(
            [transition["cue_target"] for transition in transitions],
            dtype=np.int32,
        )
    has_episode_success = [
        "episode_success" in transition for transition in transitions
    ]
    if any(has_episode_success) and not all(has_episode_success):
        raise ValueError(
            "episode_success must be present on every transition or none."
        )
    if all(has_episode_success):
        fields["episode_success"] = np.asarray(
            [transition["episode_success"] for transition in transitions],
            dtype=np.bool_,
        )

    dataset = DroidHistoryDataset.create(**fields)
    dataset.action_chunk_size = action_chunk_size
    dataset.discount = discount
    return dataset


def save_egg_cache(
    dataset: DroidHistoryDataset,
    raw_actions: np.ndarray,
    *,
    cache_dir: str | Path,
    dataset_path: str | Path,
    norm_stats_path: str | Path,
    task_config: Any,
    image_size: int,
    overwrite: bool = False,
) -> None:
    """Save full egg preprocessing with one copy of each observation frame."""
    cache_path = Path(cache_dir).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Cache directory already exists: {cache_path}. "
                "Use --overwrite only when replacing it intentionally."
            )
        import shutil

        shutil.rmtree(cache_path)

    staging_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    if staging_path.exists():
        raise FileExistsError(f"Temporary cache directory already exists: {staging_path}")
    staging_path.mkdir(parents=True, exist_ok=False)

    arrays = {
        "observations_image": np.asarray(
            dataset["observations"][MTQL_BASE_IMAGE_KEY]
        ),
        "observations_wrist_image": np.asarray(
            dataset["observations"][MTQL_WRIST_IMAGE_KEY]
        ),
        "observations_proprio": np.asarray(
            dataset["observations"][MTQL_PROPRIO_KEY]
        ),
        "actions": np.asarray(dataset["actions"]),
        "raw_actions": np.asarray(raw_actions, dtype=np.float32),
        "rewards": np.asarray(dataset["rewards"]),
        "masks": np.asarray(dataset["masks"]),
        "terminals": np.asarray(dataset["terminals"]),
    }
    if "episode_success" in dataset:
        arrays["episode_success"] = np.asarray(
            dataset["episode_success"], dtype=np.bool_
        )
    for name, array in arrays.items():
        np.save(staging_path / f"{name}.npy", array, allow_pickle=False)

    episode_manifest, manifest_episode_count = _episode_manifest(
        str(dataset_path)
    )
    if manifest_episode_count != len(dataset.terminal_locs):
        raise ValueError(
            "Egg dataset changed while preprocessing: discovered "
            f"{manifest_episode_count} finalized episodes but processed "
            f"{len(dataset.terminal_locs)}. Re-run preprocessing."
        )
    metadata = {
        "cache_format_version": EGG_CACHE_FORMAT_VERSION,
        "dataset_kind": "egg",
        "full_finalized_dataset": True,
        "dataset_path": os.path.realpath(dataset_path),
        "norm_stats_dir": _canonical_norm_stats_dir(norm_stats_path),
        "action_space": str(task_config.action_space),
        "gripper_action_space": str(task_config.gripper_action_space),
        "image_size": int(image_size),
        "num_transitions": int(dataset.size),
        "num_episodes": int(len(dataset.terminal_locs)),
        "episode_manifest_sha256": episode_manifest,
        "arrays": sorted(arrays),
    }
    with open(staging_path / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    os.replace(staging_path, cache_path)
    print(
        f"[egg-cache] wrote {metadata['num_transitions']} transitions from "
        f"{metadata['num_episodes']} episodes to {cache_path}",
        flush=True,
    )


def load_egg_history_dataset(
    cache_dir: str | Path,
    *,
    dataset_path: str | Path,
    norm_stats_path: str | Path,
    task_config: Any,
    image_size: int,
    hist_length: int,
    hist_stride: int,
    action_chunk_size: int,
    discount: float,
    n_success: int = -1,
    n_failure: int = -1,
) -> tuple[DroidHistoryDataset, dict[str, np.ndarray]]:
    """Load a validated Egg cache and apply deterministic episode selection."""
    cache_path = Path(cache_dir).expanduser()
    metadata_path = cache_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Egg cache metadata not found: {metadata_path}. "
            "Run scripts/preprocess_egg_cache.py first."
        )
    with open(metadata_path) as handle:
        metadata = json.load(handle)

    episode_manifest, manifest_episode_count = _episode_manifest(
        str(dataset_path)
    )
    expected = {
        "cache_format_version": EGG_CACHE_FORMAT_VERSION,
        "dataset_kind": "egg",
        "full_finalized_dataset": True,
        "dataset_path": os.path.realpath(dataset_path),
        "norm_stats_dir": _canonical_norm_stats_dir(norm_stats_path),
        "action_space": str(task_config.action_space),
        "gripper_action_space": str(task_config.gripper_action_space),
        "image_size": int(image_size),
        "num_episodes": manifest_episode_count,
        "episode_manifest_sha256": episode_manifest,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: cache={cached!r}, requested={requested!r}"
            for key, (cached, requested) in mismatches.items()
        )
        raise ValueError(
            f"Egg cache metadata does not match this run ({details}). "
            "Rebuild the cache with matching preprocessing inputs."
        )

    def load_array(name: str) -> np.ndarray:
        if name not in metadata.get("arrays", []):
            raise ValueError(f"Egg cache is missing array {name!r}.")
        return np.load(
            cache_path / f"{name}.npy",
            allow_pickle=False,
            mmap_mode="r",
        )

    all_episode_dirs = _discover_episode_dirs(str(dataset_path))
    selected_episode_dirs = _select_egg_episode_dirs(
        all_episode_dirs,
        str(dataset_path),
        n_success=n_success,
        n_failure=n_failure,
    )
    full_terminal_array = np.asarray(load_array("terminals"))
    full_terminal_locs = np.flatnonzero(full_terminal_array > 0)
    if len(full_terminal_locs) != len(all_episode_dirs):
        raise ValueError(
            "Egg cache terminal count does not match the finalized episode "
            f"count: terminals={len(full_terminal_locs)}, "
            f"episodes={len(all_episode_dirs)}."
        )

    path_to_index = {
        os.path.realpath(path): index
        for index, path in enumerate(all_episode_dirs)
    }
    selected_episode_indices = [
        path_to_index[os.path.realpath(path)] for path in selected_episode_dirs
    ]
    if len(selected_episode_indices) == len(all_episode_dirs):
        transition_indices = None
    else:
        starts = np.concatenate(([0], full_terminal_locs[:-1] + 1))
        ranges = [
            np.arange(starts[index], full_terminal_locs[index] + 1)
            for index in selected_episode_indices
        ]
        if not ranges:
            raise ValueError("Egg subset selected zero episodes.")
        transition_indices = np.concatenate(ranges)

    def select_array(name: str) -> np.ndarray:
        array = load_array(name)
        if transition_indices is None:
            return array
        return np.asarray(array)[transition_indices]

    rewards = select_array("rewards")
    terminals = select_array("terminals")
    if "episode_success" in metadata.get("arrays", []):
        full_episode_success = np.asarray(load_array("episode_success"), dtype=np.bool_)
    else:
        # Backward compatibility for existing full egg caches.  Egg rewards
        # are terminal-only: 1 marks a successful episode and 0 a failure.
        if len(full_terminal_locs) == 0 or full_terminal_locs[-1] != len(full_terminal_array) - 1:
            raise ValueError(
                "Egg cache terminals must mark every episode and terminate "
                "the final transition."
            )
        episode_lengths = np.diff(np.concatenate(([-1], full_terminal_locs)))
        full_rewards = np.asarray(load_array("rewards"))
        terminal_success = full_rewards[full_terminal_locs] > 0
        full_episode_success = np.repeat(terminal_success, episode_lengths)
        if len(full_episode_success) != len(full_terminal_array):
            raise ValueError(
                "Could not reconstruct per-transition egg outcomes from "
                "terminal rewards."
            )
    episode_success = (
        full_episode_success
        if transition_indices is None
        else full_episode_success[transition_indices]
    )

    dataset = DroidHistoryDataset.create(
        hist_length=hist_length,
        hist_stride=hist_stride,
        observations={
            MTQL_BASE_IMAGE_KEY: select_array("observations_image"),
            MTQL_WRIST_IMAGE_KEY: select_array("observations_wrist_image"),
            MTQL_PROPRIO_KEY: select_array("observations_proprio"),
        },
        actions=select_array("actions"),
        rewards=rewards,
        terminals=terminals,
        masks=select_array("masks"),
        episode_success=episode_success,
        actor_mask=episode_success.astype(np.float32),
    )
    dataset.action_chunk_size = action_chunk_size
    dataset.discount = discount
    info = {
        "raw_actions": select_array("raw_actions"),
        "episode_success": episode_success,
    }
    print(
        f"[egg-cache] loaded {len(rewards)} transitions from "
        f"{len(selected_episode_dirs)} of {metadata['num_episodes']} episodes "
        f"at {cache_path}",
        flush=True,
    )
    return dataset, info


def _canonical_norm_stats_dir(norm_stats_path: str | Path) -> str:
    """Return the canonical directory containing ``norm_stats.json``."""
    path = Path(norm_stats_path)
    if path.name == "norm_stats.json":
        path = path.parent
    return os.path.realpath(path)


def save_droid_cache(
    dataset: DroidHistoryDataset,
    raw_actions: np.ndarray,
    *,
    cache_dir: str | Path,
    dataset_path: str | Path,
    norm_stats_path: str | Path,
    task_config: Any,
    image_size: int,
    n_success: int,
    n_failure: int,
    overwrite: bool = False,
) -> None:
    """Save deterministic DROID preprocessing for reuse by many trainers.

    The cache deliberately contains no augmentation and no history windows.
    Those remain runtime choices, so one cache supports different history
    lengths, strides, agents, seeds, and augmentation RNG streams.
    """
    cache_path = Path(cache_dir).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Cache directory already exists: {cache_path}. "
                "Use --overwrite only when replacing it intentionally."
            )
        import shutil

        shutil.rmtree(cache_path)

    staging_path = cache_path.with_name(
        f"{cache_path.name}.tmp-{os.getpid()}"
    )
    if staging_path.exists():
        raise FileExistsError(f"Temporary cache directory already exists: {staging_path}")
    staging_path.mkdir(parents=True, exist_ok=False)

    arrays = {
        "observations_image": np.asarray(dataset["observations"][MTQL_BASE_IMAGE_KEY]),
        "observations_wrist_image": np.asarray(
            dataset["observations"][MTQL_WRIST_IMAGE_KEY]
        ),
        "observations_proprio": np.asarray(
            dataset["observations"][MTQL_PROPRIO_KEY]
        ),
        "next_observations_image": np.asarray(
            dataset["next_observations"][MTQL_BASE_IMAGE_KEY]
        ),
        "next_observations_wrist_image": np.asarray(
            dataset["next_observations"][MTQL_WRIST_IMAGE_KEY]
        ),
        "next_observations_proprio": np.asarray(
            dataset["next_observations"][MTQL_PROPRIO_KEY]
        ),
        "actions": np.asarray(dataset["actions"]),
        "raw_actions": np.asarray(raw_actions, dtype=np.float32),
        "rewards": np.asarray(dataset["rewards"]),
        "masks": np.asarray(dataset["masks"]),
        "terminals": np.asarray(dataset["terminals"]),
        "cue_targets": np.asarray(dataset["cue_targets"]),
        "episode_success": np.asarray(dataset["episode_success"]),
    }
    for name, array in arrays.items():
        np.save(staging_path / f"{name}.npy", array, allow_pickle=False)

    metadata = {
        "cache_format_version": DROID_CACHE_FORMAT_VERSION,
        "dataset_path": os.path.realpath(dataset_path),
        "norm_stats_dir": _canonical_norm_stats_dir(norm_stats_path),
        "action_space": str(task_config.action_space),
        "gripper_action_space": str(task_config.gripper_action_space),
        "image_size": int(image_size),
        "n_success": int(n_success),
        "n_failure": int(n_failure),
        "num_transitions": int(dataset.size),
        "num_episodes": int(len(dataset.terminal_locs)),
        "arrays": sorted(arrays),
    }
    with open(staging_path / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    os.replace(staging_path, cache_path)
    print(
        f"[droid-cache] wrote {metadata['num_transitions']} transitions from "
        f"{metadata['num_episodes']} episodes to {cache_path}",
        flush=True,
    )


def load_droid_history_dataset(
    cache_dir: str | Path,
    *,
    dataset_path: str | Path,
    norm_stats_path: str | Path,
    task_config: Any,
    image_size: int,
    n_success: int,
    n_failure: int,
    hist_length: int,
    hist_stride: int,
    action_chunk_size: int,
    discount: float,
) -> tuple[DroidHistoryDataset, dict[str, np.ndarray]]:
    """Load a validated deterministic DROID cache and build history lazily."""
    cache_path = Path(cache_dir).expanduser()
    metadata_path = cache_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"DROID cache metadata not found: {metadata_path}. "
            "Run scripts/preprocess_droid_cache.py first."
        )
    with open(metadata_path) as handle:
        metadata = json.load(handle)

    expected = {
        "cache_format_version": DROID_CACHE_FORMAT_VERSION,
        "dataset_path": os.path.realpath(dataset_path),
        "norm_stats_dir": _canonical_norm_stats_dir(norm_stats_path),
        "action_space": str(task_config.action_space),
        "gripper_action_space": str(task_config.gripper_action_space),
        "image_size": int(image_size),
        "n_success": int(n_success),
        "n_failure": int(n_failure),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: cache={cached!r}, requested={requested!r}"
            for key, (cached, requested) in mismatches.items()
        )
        raise ValueError(
            f"DROID cache metadata does not match this run ({details}). "
            "Rebuild the cache with matching preprocessing inputs."
        )

    def load_array(name: str) -> np.ndarray:
        if name not in metadata.get("arrays", []):
            raise ValueError(f"DROID cache is missing array {name!r}.")
        # The image arrays are ~31 GB in total.  Map them instead of eagerly
        # reading all of them from shared storage during trainer startup.
        return np.load(
            cache_path / f"{name}.npy",
            allow_pickle=False,
            mmap_mode="r",
        )

    observations = {
        MTQL_BASE_IMAGE_KEY: load_array("observations_image"),
        MTQL_WRIST_IMAGE_KEY: load_array("observations_wrist_image"),
        MTQL_PROPRIO_KEY: load_array("observations_proprio"),
    }
    next_observations = {
        MTQL_BASE_IMAGE_KEY: load_array("next_observations_image"),
        MTQL_WRIST_IMAGE_KEY: load_array("next_observations_wrist_image"),
        MTQL_PROPRIO_KEY: load_array("next_observations_proprio"),
    }
    fields = {
        "hist_length": hist_length,
        "hist_stride": hist_stride,
        "observations": observations,
        "actions": load_array("actions"),
        "rewards": load_array("rewards"),
        "next_observations": next_observations,
        "terminals": load_array("terminals"),
        "masks": load_array("masks"),
        "cue_targets": load_array("cue_targets"),
        "episode_success": load_array("episode_success"),
    }
    dataset = DroidHistoryDataset.create(**fields)
    dataset.action_chunk_size = action_chunk_size
    dataset.discount = discount
    info = {
        "raw_actions": load_array("raw_actions"),
        "cue_targets": np.asarray(dataset["cue_targets"]),
        "episode_success": np.asarray(dataset["episode_success"]),
    }
    print(
        f"[droid-cache] loaded {metadata['num_transitions']} transitions from "
        f"{metadata['num_episodes']} episodes at {cache_path}",
        flush=True,
    )
    return dataset, info


class DroidHistoryBuffer:
    """Rolling live-observation history with ``HistoryDataset`` clamp semantics."""

    def __init__(
        self,
        normalizer: OpenPINormalizer,
        *,
        hist_length: int,
        hist_stride: int = 1,
        image_size: int = DEFAULT_IMAGE_SIZE,
        visual_cues: bool = False,
    ):
        if hist_length < 0:
            raise ValueError(f"hist_length must be >= 0, got {hist_length}.")
        if hist_stride < 1:
            raise ValueError(f"hist_stride must be >= 1, got {hist_stride}.")
        if image_size < 1:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        self.normalizer = normalizer
        self.hist_length = hist_length
        self.hist_stride = hist_stride
        self.image_size = image_size
        self.visual_cues = bool(visual_cues)
        self._window = hist_length * hist_stride
        self._observations: list[dict[str, np.ndarray]] = []

    def reset(self, initial_observation: Mapping[str, Any]) -> None:
        """Clamp pre-episode history to the converted initial observation."""
        target_count = None
        if self.visual_cues:
            if "target_count" not in initial_observation:
                raise KeyError(
                    "Visual-cue evaluation requires observation['target_count']."
                )
            target_count = int(
                np.asarray(initial_observation["target_count"]).reshape(-1)[0]
            )
        observation = convert_droid_observation(
            initial_observation,
            self.normalizer,
            image_size=self.image_size,
            target_count=target_count,
        )
        self._observations = [observation] * self._window

    def append(self, observation: Mapping[str, Any]) -> None:
        """Record the observation used for the action that was just executed."""
        if self._window == 0:
            return
        if not self._observations:
            raise RuntimeError("Call reset() before append().")
        target_count = None
        if self.visual_cues:
            if "target_count" not in observation:
                raise KeyError(
                    "Visual-cue evaluation requires observation['target_count']."
                )
            target_count = int(
                np.asarray(observation["target_count"]).reshape(-1)[0]
            )
        converted = convert_droid_observation(
            observation,
            self.normalizer,
            image_size=self.image_size,
            target_count=target_count,
        )
        self._observations = self._observations[1:] + [converted]

    def history_observations(self) -> dict[str, np.ndarray] | None:
        """Return unbatched sparse history ready for MTQL inference."""
        if self.hist_length == 0:
            return None
        if not self._observations:
            raise RuntimeError("Call reset() before requesting history.")
        return _stack_observations(self._observations[:: self.hist_stride])


def _import_augmentation():
    try:
        import augmax
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "augmax and JAX are required for EXPO-style DROID augmentation."
        ) from exc
    return augmax, jax, jnp


_AUGMENTER_CACHE: dict[tuple[int, int], Any] = {}
_TARGET_CUE_DRAWER_CACHE: dict[tuple[int, int], Any] = {}


def _get_jitted_augmenter(image_height: int, image_width: int) -> Any:
    """Return a fused JAX augmenter for the fixed image resolution.

    The leading dimension is the number of camera/temporal groups and the
    remaining image dimensions are ``(batch, time, height, width, channels)``.
    A separate random key is supplied for every group and batch item; that key
    is reused for every frame in the item's temporal window.
    """
    cache_key = (int(image_height), int(image_width))
    if cache_key in _AUGMENTER_CACHE:
        return _AUGMENTER_CACHE[cache_key]

    augmax, jax, jnp = _import_augmentation()
    transform = augmax.Chain(
        augmax.RandomCrop(int(image_width * 0.95), int(image_height * 0.95)),
        augmax.Resize(image_width, image_height),
        augmax.Rotate((-5, 5)),
        augmax.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    )

    def augment_sequences(sequences, group_rngs):
        # sequences: (groups, batch, time, height, width, channels)
        sequences = jnp.asarray(sequences, dtype=jnp.float32) / 255.0
        batch_size = sequences.shape[1]
        sample_keys = jax.vmap(
            lambda key: jax.random.split(key, batch_size)
        )(group_rngs)

        def augment_one(sample_key, frames):
            frame_keys = jnp.broadcast_to(
                sample_key, (frames.shape[0], sample_key.shape[0])
            )
            return jax.vmap(transform)(frame_keys, frames)

        augment_batch = jax.vmap(augment_one, in_axes=(0, 0))
        augmented = jax.vmap(augment_batch, in_axes=(0, 0))(
            sample_keys, sequences
        )
        return jnp.clip(jnp.rint(augmented * 255.0), 0, 255).astype(jnp.uint8)

    _AUGMENTER_CACHE[cache_key] = jax.jit(augment_sequences)
    return _AUGMENTER_CACHE[cache_key]


def _get_jitted_target_cue_drawer(image_height: int, image_width: int) -> Any:
    """Return a device-side drawer for the post-augmentation target cue."""
    cache_key = (int(image_height), int(image_width))
    if cache_key in _TARGET_CUE_DRAWER_CACHE:
        return _TARGET_CUE_DRAWER_CACHE[cache_key]

    _, jax, jnp = _import_augmentation()
    height, width = cache_key
    scale = min(height, width)
    margin = max(4, int(round(scale * 0.025)))
    panel_height = min(max(28, int(round(height * 0.27))), height - margin)
    panel_width = min(max(96, int(round(width * 0.41))), width - margin)
    border = max(1, int(round(scale * 0.012)))
    radius = max(6, int(round(scale * 0.055)))
    y0, x0 = margin, margin
    y1, x1 = y0 + panel_height, x0 + panel_width

    yy, xx = jnp.meshgrid(
        jnp.arange(height), jnp.arange(width), indexing="ij"
    )
    panel_mask = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
    # Match _draw_target_cue exactly. Its bottom-edge assignments cover only
    # the bottom portions of the two vertical borders; preserving that detail
    # keeps the device fast path pixel-identical to the original CPU path.
    border_mask = (
        ((yy >= y0) & (yy < y0 + border) & (xx >= x0) & (xx < x1))
        | ((yy >= y1 - border) & (yy < y1) & (xx >= x0) & (xx < x0 + border))
        | ((yy >= y1 - border) & (yy < y1) & (xx >= x1 - border) & (xx < x1))
        | ((yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x0 + border))
        | ((yy >= y0) & (yy < y1) & (xx >= x1 - border) & (xx < x1))
    )

    center_y = y0 + panel_height // 2
    centers_x = np.linspace(
        x0 + panel_width * 0.22,
        x0 + panel_width * 0.78,
        3,
    ).round().astype(int)
    inner_radius = max(1, radius - border)
    circle_masks = []
    for center_x in centers_x:
        distance = (yy - center_y) ** 2 + (xx - int(center_x)) ** 2
        circle_masks.append(
            (distance <= radius**2, distance <= inner_radius**2)
        )

    def draw(frames, targets):
        # frames: (batch, time, height, width, channels)
        frames = jnp.asarray(frames)
        targets = jnp.asarray(targets, dtype=jnp.int32).reshape(-1)
        white = jnp.asarray(255, dtype=frames.dtype)
        black = jnp.asarray(0, dtype=frames.dtype)
        overlay = jnp.zeros_like(frames)
        overlay = jnp.where(
            border_mask[None, None, :, :, None], white, overlay
        )
        for slot, (outer, inner) in enumerate(circle_masks, start=1):
            overlay = jnp.where(
                outer[None, None, :, :, None], white, overlay
            )
            inner_value = jnp.where(
                targets[:, None, None, None, None] >= slot,
                white,
                black,
            )
            overlay = jnp.where(
                inner[None, None, :, :, None], inner_value, overlay
            )
        return jnp.where(
            panel_mask[None, None, :, :, None], overlay, frames
        )

    _TARGET_CUE_DRAWER_CACHE[cache_key] = jax.jit(draw)
    return _TARGET_CUE_DRAWER_CACHE[cache_key]


def _draw_target_cue(frames: np.ndarray, target: int) -> None:
    """Draw the path-derived target cue into a sequence of HWC RGB frames."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 (T,H,W,3) frames, got {frames.shape} {frames.dtype}"
        )
    _, height, width, _ = frames.shape
    scale = min(height, width)
    margin = max(4, int(round(scale * 0.025)))
    panel_height = min(max(28, int(round(height * 0.27))), height - margin)
    panel_width = min(max(96, int(round(width * 0.41))), width - margin)
    border = max(1, int(round(scale * 0.012)))
    radius = max(6, int(round(scale * 0.055)))
    y0, x0 = margin, margin
    y1, x1 = y0 + panel_height, x0 + panel_width

    frames[:, y0:y1, x0:x1, :] = 0
    frames[:, y0:y0 + border, x0:x1, :] = 255
    frames[:, y1 - border:y1, x0:x0 + border, :] = 255
    frames[:, y1 - border:y1, x1 - border:x1, :] = 255
    frames[:, y0:y1, x0:x0 + border, :] = 255
    frames[:, y0:y1, x1 - border:x1, :] = 255

    yy, xx = np.ogrid[:height, :width]
    center_y = y0 + panel_height // 2
    centers_x = np.linspace(
        x0 + panel_width * 0.22,
        x0 + panel_width * 0.78,
        3,
    ).round().astype(int)
    inner_radius = max(1, radius - border)
    for slot, center_x in enumerate(centers_x, start=1):
        distance = (yy - center_y) ** 2 + (xx - center_x) ** 2
        outer = distance <= radius**2
        inner = distance <= inner_radius**2
        frames[:, outer, :] = 255
        if target >= slot:
            frames[:, inner, :] = 255
        else:
            frames[:, inner, :] = 0


def add_droid_target_cues(
    batch: Mapping[str, Any], cue_targets: np.ndarray
) -> dict[str, Any]:
    """Draw target cues after image augmentation on all temporal observations."""
    result: MutableMapping[str, Any] = dict(batch)
    cue_targets = np.asarray(cue_targets, dtype=np.int32).reshape(-1)
    sequence_fields = (
        "history_observations",
        "observations",
        "next_history_observations",
        "next_observations",
    )
    for field in sequence_fields:
        if field not in result or MTQL_BASE_IMAGE_KEY not in result[field]:
            continue
        observations = dict(result[field])
        base = np.asarray(observations[MTQL_BASE_IMAGE_KEY])
        if base.ndim == 4:
            sequence = base[:, None, ...].copy()
            restore_single = True
        elif base.ndim == 5:
            sequence = base.copy()
            restore_single = False
        else:
            raise ValueError(
                f"Expected {field}/{MTQL_BASE_IMAGE_KEY} with 4 or 5 dimensions, "
                f"got {base.shape}."
            )
        if sequence.shape[0] != cue_targets.shape[0]:
            raise ValueError(
                f"Cue target batch has shape {cue_targets.shape}, but {field} "
                f"has batch shape {sequence.shape}."
            )
        for batch_index, target in enumerate(cue_targets):
            _draw_target_cue(sequence[batch_index], int(target))
        observations[MTQL_BASE_IMAGE_KEY] = (
            sequence[:, 0] if restore_single else sequence
        )
        result[field] = observations
    return dict(result)


def add_droid_target_cues_jax(
    batch: Mapping[str, Any], cue_targets: Any
) -> dict[str, Any]:
    """Draw target cues on-device after augmentation.

    This is the training fast path. Unlike ``add_droid_target_cues``, it keeps
    the resulting image arrays as JAX arrays, avoiding a device-to-host copy
    immediately before the jitted agent update.
    """
    _, jax, jnp = _import_augmentation()
    del jax  # The cached drawer owns the JIT boundary.
    result: MutableMapping[str, Any] = dict(batch)
    cue_targets = jnp.asarray(cue_targets, dtype=jnp.int32).reshape(-1)
    sequence_fields = (
        "history_observations",
        "observations",
        "next_history_observations",
        "next_observations",
    )
    for field in sequence_fields:
        if field not in result or MTQL_BASE_IMAGE_KEY not in result[field]:
            continue
        observations = dict(result[field])
        base = jnp.asarray(observations[MTQL_BASE_IMAGE_KEY])
        if base.ndim == 4:
            sequence = base[:, None, ...]
            restore_single = True
        elif base.ndim == 5:
            sequence = base
            restore_single = False
        else:
            raise ValueError(
                f"Expected {field}/{MTQL_BASE_IMAGE_KEY} with 4 or 5 dimensions, "
                f"got {base.shape}."
            )
        if sequence.shape[0] != cue_targets.shape[0]:
            raise ValueError(
                f"Cue target batch has shape {cue_targets.shape}, but {field} "
                f"has batch shape {sequence.shape}."
            )
        drawer = _get_jitted_target_cue_drawer(
            int(sequence.shape[-3]), int(sequence.shape[-2])
        )
        drawn = drawer(sequence, cue_targets)
        observations[MTQL_BASE_IMAGE_KEY] = (
            drawn[:, 0] if restore_single else drawn
        )
        result[field] = observations
    return dict(result)


def augment_droid_batch(
    batch: Mapping[str, Any], rng: Any
) -> dict[str, Any]:
    """Apply EXPO-FT's full image augmentation to an MTQL batch.

    EXPO samples a separate transform for each camera image. MTQL adds a
    temporal history, so each batch item/camera shares one transform across
    history plus current, while next history plus next gets an independent
    draw. Proprioception and actions are left untouched.
    """
    augmax, jax, jnp = _import_augmentation()
    result: MutableMapping[str, Any] = dict(batch)
    sequence_fields = (
        "history_observations",
        "observations",
        "next_history_observations",
        "next_observations",
    )
    for field in sequence_fields:
        if field in result:
            result[field] = dict(result[field])

    present_fields = [field for field in sequence_fields if field in result]
    if not present_fields:
        raise KeyError("Batch has no MTQL observation fields to augment.")

    # Keep a transition's temporal geometry intact. The transform is shared
    # across history + current for each camera, while next history + next
    # observation receive a separate draw, just as EXPO augments current and
    # next observations independently.
    groups = (
        ("current", ("history_observations", "observations")),
        ("next", ("next_history_observations", "next_observations")),
    )
    active_groups = [
        (name, tuple(field for field in fields if field in result))
        for name, fields in groups
        if any(field in result for field in fields)
    ]
    camera_keys = [
        key
        for key in MTQL_IMAGE_KEYS
        if any(key in result[field] for _, fields in active_groups for field in fields)
    ]
    if not camera_keys:
        raise KeyError(
            f"Batch contains none of the expected image keys: {MTQL_IMAGE_KEYS}."
        )

    # Fast path: fuse all camera/current-next groups into one JAX call.  The
    # usual training batch has the same temporal length for every group, so
    # this removes four separate augmentation dispatches and four host/device
    # round trips.  The fallback below keeps support for unusual partial
    # batches with different temporal shapes.
    packed_sequences = []
    packed_specs = []
    for camera_key in camera_keys:
        for _, fields in active_groups:
            group_fields = [
                field for field in fields if camera_key in result[field]
            ]
            if not group_fields:
                continue
            pieces = []
            field_specs = []
            for field in group_fields:
                original = np.asarray(result[field][camera_key])
                if original.ndim == 4:
                    images = original[:, None]
                elif original.ndim == 5:
                    images = original
                else:
                    raise ValueError(
                        f"Expected batched images in {field}/{camera_key}, "
                        f"got {original.shape}."
                    )
                pieces.append(images)
                field_specs.append((field, images.shape[1], original.ndim))
            packed_sequences.append(np.concatenate(pieces, axis=1))
            packed_specs.append((camera_key, field_specs))

    packed_shapes = {sequence.shape[1:] for sequence in packed_sequences}
    if packed_sequences and len(packed_shapes) == 1:
        packed = np.stack(packed_sequences, axis=0)
        image_height, image_width = packed.shape[-3:-1]
        _, jax, _ = _import_augmentation()
        group_rngs = jax.random.split(rng, len(packed_sequences))
        # Keep the result on the accelerator. Converting this JAX array back
        # to NumPy here forced a device-to-host copy on every gradient step;
        # the subsequent jitted agent update then copied it back to the device.
        augmented = _get_jitted_augmenter(image_height, image_width)(
            packed, group_rngs
        )

        for group_index, (camera_key, field_specs) in enumerate(packed_specs):
            offset = 0
            for field, length, original_ndim in field_specs:
                images = augmented[group_index, :, offset : offset + length]
                result[field][camera_key] = (
                    images[:, 0] if original_ndim == 4 else images
                )
                offset += length
        return dict(result)

    first_field = active_groups[0][1][0]
    first_camera = next(
        key for key in camera_keys if key in result[first_field]
    )
    batch_size = np.asarray(result[first_field][first_camera]).shape[0]
    group_rngs = jax.random.split(
        rng, len(camera_keys) * len(active_groups)
    ).reshape(len(camera_keys), len(active_groups), -1)

    for camera_index, camera_key in enumerate(camera_keys):
        for group_index, (_, fields) in enumerate(active_groups):
            group_fields = [field for field in fields if camera_key in result[field]]
            if not group_fields:
                continue
            pieces = []
            piece_lengths = []
            original_ndims = {}
            for field in group_fields:
                original = np.asarray(result[field][camera_key])
                original_ndims[field] = original.ndim
                if original.ndim == 4:
                    images = original[:, None]
                elif original.ndim == 5:
                    images = original
                else:
                    raise ValueError(
                        f"Expected batched images in {field}/{camera_key}, "
                        f"got {original.shape}."
                    )
                pieces.append(images)
                piece_lengths.append(images.shape[1])

            sequence = jnp.asarray(
                np.concatenate(pieces, axis=1), dtype=jnp.float32
            ) / 255.0
            height, width = sequence.shape[-3], sequence.shape[-2]
            transform = augmax.Chain(
                augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                augmax.Resize(width, height),
                augmax.Rotate((-5, 5)),
                augmax.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1
                ),
            )

            # One key per batch item; reuse it for all temporal frames in this
            # current/next group so motion is not changed by augmentation.
            sample_rngs = group_rngs[camera_index, group_index]

            def augment_sequence(key, frames):
                frame_keys = jnp.broadcast_to(
                    key, (frames.shape[0], key.shape[0])
                )
                return jax.vmap(transform)(frame_keys, frames)

            sequence = jax.vmap(augment_sequence)(
                jax.random.split(sample_rngs, batch_size), sequence
            )
            sequence = jnp.clip(
                jnp.rint(sequence * 255.0), 0, 255
            ).astype(jnp.uint8)

            offset = 0
            for field, length in zip(group_fields, piece_lengths):
                images = sequence[:, offset : offset + length]
                result[field][camera_key] = (
                    images[:, 0] if original_ndims[field] == 4 else images
                )
                offset += length

    return dict(result)
