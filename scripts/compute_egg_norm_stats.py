#!/usr/bin/env python3
"""Compute state/action normalization statistics over all finalized egg data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm

from utils.mtql_droid import (
    DROID_CARTESIAN_POSITION_KEY,
    DROID_GRIPPER_POSITION_KEY,
    _discover_episode_dirs,
)


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "dataset_path",
    "/iris/u/ronpo/expo-ft-data/egg_v1",
    "Root containing finalized egg episodes.",
)
flags.DEFINE_string(
    "output_dir",
    "/iris/u/ronpo/expo-ft-data/egg_v1_norm_stats",
    "Directory in which norm_stats.json is written.",
)
flags.DEFINE_bool("overwrite", False, "Replace an existing norm_stats.json.")
config_flags.DEFINE_config_file(
    "config_task",
    None,
    "EXPO-FT egg task config used to select action keys.",
    lock_config=False,
)


def _statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main(argv):
    del argv
    if FLAGS.config_task is None:
        raise app.UsageError("Missing required flag: --config_task")
    if not os.path.isdir(FLAGS.dataset_path):
        raise FileNotFoundError(f"Egg dataset not found: {FLAGS.dataset_path}")

    output_path = Path(FLAGS.output_dir).expanduser() / "norm_stats.json"
    if output_path.exists() and not FLAGS.overwrite:
        raise FileExistsError(
            f"Norm stats already exist: {output_path}. "
            "Use --overwrite only when replacing them intentionally."
        )

    action_key = FLAGS.config_task.action_space
    gripper_action_key = f"gripper_{FLAGS.config_task.gripper_action_space}"
    states = []
    actions = []
    episode_dirs = _discover_episode_dirs(FLAGS.dataset_path)
    for episode_dir in tqdm(episode_dirs, desc="Reading egg state/actions"):
        with h5py.File(os.path.join(episode_dir, "traj.hdf5"), "r") as handle:
            observation_group = handle["saved_observation"]
            cartesian_position = np.asarray(
                observation_group[DROID_CARTESIAN_POSITION_KEY]
            )
            gripper_position = np.asarray(
                observation_group[DROID_GRIPPER_POSITION_KEY]
            )
            if gripper_position.ndim == 1:
                gripper_position = gripper_position[:, None]
            action_group = handle["action"]
            cartesian_action = np.asarray(action_group[action_key])
            gripper_action = np.asarray(action_group[gripper_action_key])
            if gripper_action.ndim == 1:
                gripper_action = gripper_action[:, None]
            episode_state = np.concatenate(
                [cartesian_position, gripper_position], axis=-1
            )
            episode_actions = np.concatenate(
                [cartesian_action, gripper_action], axis=-1
            )
            if len(episode_state) != len(episode_actions):
                raise ValueError(
                    f"State/action length mismatch in {episode_dir}: "
                    f"{len(episode_state)} != {len(episode_actions)}"
                )
            states.append(episode_state)
            actions.append(episode_actions)

    if not states:
        raise ValueError(f"No finalized egg episodes found under {FLAGS.dataset_path}")
    states_array = np.concatenate(states, axis=0)
    actions_array = np.concatenate(actions, axis=0)
    payload = {
        "norm_stats": {
            "state": _statistics(states_array),
            "actions": _statistics(actions_array),
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.name}.tmp-{os.getpid()}"
    )
    with open(temporary_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, output_path)
    print(
        f"Wrote egg norm stats for {len(episode_dirs)} episodes and "
        f"{len(states_array)} transitions to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    app.run(main)
