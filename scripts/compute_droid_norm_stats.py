#!/usr/bin/env python3
"""Compute OpenPI-compatible state/action quantile stats from raw DROID data."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from openpi.shared import normalize


def _episodes(dataset_path: Path, num_data: int) -> list[Path]:
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    # DROID collections are grouped as target_N/{success,failure}/episode and
    # arbitrary/target_num_N/{failure}/episode. Walk recursively, retaining
    # only numeric episode directories; this skips tmp/session_* recordings.
    episodes = sorted(
        (
            path.parent
            for path in dataset_path.rglob("traj.hdf5")
            if path.parent.name.isdigit()
        ),
        key=lambda path: tuple(
            int(part) if part.isdigit() else part
            for part in path.relative_to(dataset_path).parts
        ),
    )
    if num_data > 0:
        episodes = episodes[:num_data]
    if not episodes:
        raise ValueError(f"No numeric episode directories found under {dataset_path}")
    return episodes


def _load_state_and_actions(
    trajectory_path: Path,
    action_space: str,
    gripper_action_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(trajectory_path, "r") as handle:
        observation = handle["saved_observation"]
        cartesian_position = np.asarray(observation["cartesian_position"], dtype=np.float32)
        gripper_position = np.asarray(observation["gripper_position"], dtype=np.float32).reshape(-1, 1)
        state = np.concatenate([cartesian_position, gripper_position], axis=-1)

        actions = handle["action"]
        cartesian_action = np.asarray(actions[action_space], dtype=np.float32)
        gripper_action = np.asarray(actions[f"gripper_{gripper_action_space}"], dtype=np.float32).reshape(-1, 1)
        action = np.concatenate([cartesian_action, gripper_action], axis=-1)

    if len(state) != len(action):
        raise ValueError(
            f"State/action length mismatch in {trajectory_path}: {len(state)} vs {len(action)}"
        )
    return state, action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-data", type=int, default=0)
    parser.add_argument("--action-space", default="cartesian_velocity")
    parser.add_argument("--gripper-action-space", default="velocity")
    args = parser.parse_args()

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    total = 0
    for episode in _episodes(args.dataset_path, args.num_data):
        state, action = _load_state_and_actions(
            episode / "traj.hdf5",
            args.action_space,
            args.gripper_action_space,
        )
        state_stats.update(state)
        action_stats.update(action)
        total += len(state)

    if total < 2:
        raise ValueError("At least two transitions are required to compute statistics.")

    normalize.save(
        args.output_dir,
        {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        },
    )
    print(f"Wrote OpenPI-compatible norm_stats.json using {total} transitions to {args.output_dir}")


if __name__ == "__main__":
    main()

