#!/usr/bin/env python3
"""Create Shuffle edited7 trajectories from the shuffle_v2 recordings.

Each output trajectory keeps the first seven source timesteps, then appends
the source suffix beginning one timestep before the first sustained arm-motion
command.  Every time-aligned HDF5 dataset is sliced with the same index map;
robot actions and state therefore remain aligned with observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


COLORS = ("black", "white", "orange")
OUTCOMES = ("success", "failure")


def first_sustained_motion(action: np.ndarray, threshold: float, consecutive: int) -> int:
    """Return the first arm-command index above threshold for N frames."""
    if action.ndim != 2 or action.shape[1] < 6:
        raise ValueError(f"Expected action shape (T, >=6), got {action.shape}")
    magnitude = np.max(np.abs(action[:, :6]), axis=1)
    active = magnitude > threshold
    for start in range(0, len(active) - consecutive + 1):
        if bool(np.all(active[start : start + consecutive])):
            return start
    active_indices = np.flatnonzero(active)
    if len(active_indices):
        return int(active_indices[0])
    raise ValueError("Trajectory contains no arm-motion command")


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def write_sliced_trajectory(
    source_path: Path,
    target_path: Path,
    *,
    prefix_frames: int,
    pre_motion_frames: int,
    threshold: float,
    consecutive: int,
) -> tuple[int, int, int]:
    with h5py.File(source_path, "r") as source:
        action = np.asarray(source["action/executed_action"])
        source_length = int(action.shape[0])
        motion_start = first_sustained_motion(action, threshold, consecutive)
        suffix_start = max(prefix_frames, motion_start - pre_motion_frames)
        indices = np.concatenate(
            (
                np.arange(min(prefix_frames, source_length), dtype=np.int64),
                np.arange(suffix_start, source_length, dtype=np.int64),
            )
        )
        # The max(prefix_frames, ...) guard makes the two parts disjoint.
        if len(np.unique(indices)) != len(indices):
            raise AssertionError(f"Overlapping index map for {source_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(target_path, "w") as target:
            copy_attrs(source.attrs, target.attrs)
            target.attrs["edited_prefix_frames"] = prefix_frames
            target.attrs["edited_motion_start_original_index"] = motion_start
            target.attrs["edited_pre_motion_frames"] = pre_motion_frames
            target.attrs["edited_source_length"] = source_length
            target.attrs["edited_length"] = int(len(indices))
            target.attrs["edited_motion_threshold"] = threshold
            target.attrs["edited_motion_consecutive"] = consecutive

            def copy_item(name: str, item: h5py.Dataset | h5py.Group) -> None:
                if isinstance(item, h5py.Group):
                    group = target.require_group(name)
                    copy_attrs(item.attrs, group.attrs)
                    return
                parent_name, _, leaf_name = name.rpartition("/")
                parent = target.require_group(parent_name) if parent_name else target
                if item.shape and item.shape[0] == source_length:
                    data = item[indices]
                else:
                    data = item[()]
                dataset = parent.create_dataset(leaf_name, data=data)
                copy_attrs(item.attrs, dataset.attrs)

            source.visititems(copy_item)

    return source_length, motion_start, int(len(indices))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/iris/u/ronpo/expo-ft-data/shuffle_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/iris/u/ronpo/expo-ft-data/shuffle_v2"),
    )
    parser.add_argument("--prefix-frames", type=int, default=7)
    parser.add_argument("--pre-motion-frames", type=int, default=1)
    parser.add_argument("--motion-threshold", type=float, default=1e-4)
    parser.add_argument("--motion-consecutive", type=int, default=2)
    parser.add_argument("--colors", nargs="+", choices=COLORS, default=list(COLORS))
    parser.add_argument("--outcomes", nargs="+", choices=OUTCOMES, default=list(OUTCOMES))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output trajectories that already exist.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.prefix_frames < 1 or args.pre_motion_frames < 0:
        parser.error("prefix-frames must be positive and pre-motion-frames nonnegative")
    if args.motion_threshold <= 0 or args.motion_consecutive < 1:
        parser.error("motion threshold must be positive and consecutive must be >= 1")

    total = 0
    for color in args.colors:
        for outcome in args.outcomes:
            source_group = args.source_root / color / outcome
            target_group = args.output_root / f"{color}_edited7" / outcome
            episodes = sorted(
                (p for p in source_group.iterdir() if p.is_dir() and p.name.isdigit()),
                key=lambda p: int(p.name),
            )
            if not episodes:
                raise FileNotFoundError(f"No episodes found under {source_group}")
            for episode in episodes:
                source_path = episode / "traj.hdf5"
                target_path = target_group / episode.name / "traj.hdf5"
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                if target_path.exists() and args.skip_existing:
                    print(f"skip existing {target_path}", flush=True)
                    total += 1
                    continue
                if target_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Refusing to replace {target_path}; pass --overwrite to replace it"
                    )
                source_length, motion_start, edited_length = write_sliced_trajectory(
                    source_path,
                    target_path,
                    prefix_frames=args.prefix_frames,
                    pre_motion_frames=args.pre_motion_frames,
                    threshold=args.motion_threshold,
                    consecutive=args.motion_consecutive,
                )
                print(
                    f"{color}/{outcome}/{episode.name}: "
                    f"source={source_length} motion={motion_start} "
                    f"suffix_start={max(args.prefix_frames, motion_start - args.pre_motion_frames)} "
                    f"edited={edited_length}",
                    flush=True,
                )
                total += 1
    print(f"Created {total} edited7 trajectories under {args.output_root}")


if __name__ == "__main__":
    main()
