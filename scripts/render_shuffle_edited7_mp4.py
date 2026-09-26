#!/usr/bin/env python3
"""Render the three saved camera streams from Shuffle edited7 HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio


COLORS = ("black", "white", "orange")
OUTCOMES = ("success", "failure")
CAMERAS = (
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
)


def render_episode(traj_path: Path, fps: float, overwrite: bool) -> None:
    output_dir = traj_path.parent / "recordings" / "MP4"
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(traj_path, "r") as handle:
        for camera in CAMERAS:
            key = f"saved_observation/{camera}"
            if key not in handle:
                raise KeyError(f"Missing {key} in {traj_path}")
            output_path = output_dir / f"{camera}.mp4"
            if output_path.exists() and not overwrite:
                print(f"skip existing {output_path}", flush=True)
                continue
            frames = handle[key]
            with imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            ) as writer:
                for index in range(frames.shape[0]):
                    writer.append_data(frames[index])
            print(
                f"{traj_path}: rendered {camera} ({frames.shape[0]} frames)",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/iris/u/ronpo/expo-ft-data/shuffle_v2"),
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--colors", nargs="+", choices=COLORS, default=list(COLORS))
    parser.add_argument("--outcomes", nargs="+", choices=OUTCOMES, default=list(OUTCOMES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    count = 0
    for color in args.colors:
        for outcome in args.outcomes:
            group = args.root / f"{color}_edited7" / outcome
            trajectories = sorted(
                group.glob("*/traj.hdf5"), key=lambda p: int(p.parent.name)
            )
            for traj_path in trajectories:
                render_episode(traj_path, args.fps, args.overwrite)
                count += 1
    print(f"Processed {count} edited7 trajectories at {args.fps:g} fps")


if __name__ == "__main__":
    main()
