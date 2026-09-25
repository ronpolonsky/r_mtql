"""MTQL environment and offline-dataset adapters for transfer counting."""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

from utils.datasets import (
    HistoryDataset,
    HistoryReplayBuffer,
    LazyEpisodeReplayBuffer,
)


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANISKILL_ROOT = os.path.join(
    _PROJECT_ROOT, "cabinet-memory-sim", "ManiSkill"
)
_DEFAULT_DATASET_DIR = os.environ.get(
    "COUNTING_DATASET",
    os.path.join(_MANISKILL_ROOT, "counting_dataset"),
)
_DEFAULT_ENV_ID = os.environ.get("COUNTING_TASK_ENV_ID", "TransferCountPanda-v0")


def _flat_dataset(path: str) -> dict:
    """Load one merged counting NPZ into the shared MTQL dictionary schema."""
    with np.load(path) as data:
        return {
            "observations": {
                "agent_view": data["obs_image"].copy(),
                "wrist_view": data["obs_wrist_image"].copy(),
                "proprio": data["obs_proprio"].astype(np.float32),
            },
            "actions": data["actions"].astype(np.float32),
            "rewards": data["rewards"].astype(np.float32),
            "terminals": data["terminals"].astype(np.float32),
            "masks": data["masks"].astype(np.float32),
        }


def _truncate_complete_episodes(data: dict, max_demos: int) -> dict:
    if max_demos <= 0:
        return data
    terminal_indices = np.nonzero(data["terminals"] > 0)[0]
    if len(terminal_indices) <= max_demos:
        return data
    cutoff = int(terminal_indices[max_demos - 1]) + 1
    print(
        f"[load_counting_datasets] Truncated to {max_demos} demos "
        f"({cutoff} transitions)"
    )
    return {
        key: (
            {obs_key: value[:cutoff] for obs_key, value in values.items()}
            if isinstance(values, dict)
            else values[:cutoff]
        )
        for key, values in data.items()
    }


def load_counting_datasets(
    dataset_dir: str = _DEFAULT_DATASET_DIR,
    hist_length: int = 4,
    hist_stride: int = 1,
    online_buf_size: int = 20_000,
    ep_cache_size: int = 40,
    chunk_reload_interval: int = 1000,
    action_chunk_size: int = 1,
    discount: float = 1.0,
    lazy: bool = True,
    max_demos: int = 0,
    env_id: str = _DEFAULT_ENV_ID,
):
    """Load counting demonstrations in the same format as cabinet data."""
    if _MANISKILL_ROOT not in sys.path:
        sys.path.insert(0, _MANISKILL_ROOT)
    from counting_task.version import VERSION_FILENAME, task_version_for_env

    task_version = task_version_for_env(env_id)

    version_path = os.path.join(dataset_dir, VERSION_FILENAME)
    if not os.path.isfile(version_path):
        raise RuntimeError(
            f"Counting dataset has no task version marker: {version_path}. "
            f"It predates {task_version!r}; recollect it for this task variant."
        )
    with open(version_path, encoding="utf-8") as version_file:
        dataset_version = version_file.read().strip()
    if dataset_version != task_version:
        raise RuntimeError(
            f"Counting dataset version {dataset_version!r} does not match "
            f"task version {task_version!r}; recollect the dataset."
        )

    if lazy:
        train_manifest = os.path.join(dataset_dir, "train_episodes.txt")
        if os.path.isfile(train_manifest):
            with open(train_manifest, encoding="utf-8") as manifest:
                episode_paths = [
                    os.path.join(dataset_dir, line.strip())
                    for line in manifest
                    if line.strip()
                ]
        else:
            episode_paths = sorted(
                glob.glob(os.path.join(dataset_dir, "episodes", "ep_*.npz"))
            )
        if not episode_paths:
            raise FileNotFoundError(
                "No counting episodes found in "
                f"{os.path.join(dataset_dir, 'episodes')}. Collect them with "
                "`python -m counting_task.collect --output-dir counting_dataset`."
            )
        missing_paths = [path for path in episode_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(
                f"Counting train manifest references missing episode: {missing_paths[0]}"
            )
        train_dataset = LazyEpisodeReplayBuffer(
            episode_paths=episode_paths,
            hist_length=hist_length,
            hist_stride=hist_stride,
            online_buf_size=online_buf_size,
            ep_cache_size=ep_cache_size,
            chunk_reload_interval=chunk_reload_interval,
            action_chunk_size=action_chunk_size,
            discount=discount,
        )
    else:
        train_path = os.path.join(dataset_dir, "train_counting_dataset.npz")
        if not os.path.isfile(train_path):
            raise FileNotFoundError(
                f"Counting training dataset not found: {train_path}. Collect it "
                "with `python -m counting_task.collect --output-dir "
                "counting_dataset`."
            )
        train_data = _truncate_complete_episodes(
            _flat_dataset(train_path), max_demos
        )
        initial_dataset = HistoryDataset.create(
            hist_length=hist_length,
            hist_stride=hist_stride,
            **train_data,
        )
        train_dataset = HistoryReplayBuffer.create_from_initial_dataset(
            initial_dataset._dict,
            size=len(train_data["rewards"]) + online_buf_size,
            hist_length=hist_length,
            hist_stride=hist_stride,
        )

    val_path = os.path.join(dataset_dir, "val_counting_dataset.npz")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(
            f"Counting validation dataset not found: {val_path}. The collector "
            "needs at least two episodes to create a validation split."
        )
    val_data = _flat_dataset(val_path)
    val_dataset = HistoryDataset.create(
        hist_length=hist_length,
        hist_stride=hist_stride,
        **val_data,
    )
    return train_dataset, val_dataset


def make_counting_eval_env(
    seed: int = 0,
    image_size: int = 128,
    env_id: str = _DEFAULT_ENV_ID,
):
    """Construct one normalized-action, two-camera counting environment."""
    if _MANISKILL_ROOT not in sys.path:
        sys.path.insert(0, _MANISKILL_ROOT)
    from counting_task.wrapper import TransferCountGymEnv

    return TransferCountGymEnv(seed=seed, image_size=image_size, env_id=env_id)


def make_counting_env_and_datasets(
    hist_length: int = 4,
    hist_stride: int = 1,
    dataset_dir: str = _DEFAULT_DATASET_DIR,
    seed: int = 0,
    online_buf_size: int = 20_000,
    ep_cache_size: int = 40,
    chunk_reload_interval: int = 1000,
    action_chunk_size: int = 1,
    discount: float = 1.0,
    image_size: int = 128,
    lazy: bool = True,
    max_demos: int = 0,
    env_id: str = _DEFAULT_ENV_ID,
):
    """Return independent train/eval envs and counting train/val datasets."""
    train_dataset, val_dataset = load_counting_datasets(
        dataset_dir=dataset_dir,
        hist_length=hist_length,
        hist_stride=hist_stride,
        online_buf_size=online_buf_size,
        ep_cache_size=ep_cache_size,
        chunk_reload_interval=chunk_reload_interval,
        action_chunk_size=action_chunk_size,
        discount=discount,
        lazy=lazy,
        max_demos=max_demos,
        env_id=env_id,
    )
    env = make_counting_eval_env(seed=seed, image_size=image_size, env_id=env_id)
    eval_env = make_counting_eval_env(
        seed=seed, image_size=image_size, env_id=env_id
    )
    return env, eval_env, train_dataset, val_dataset
