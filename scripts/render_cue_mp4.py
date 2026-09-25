#!/usr/bin/env python3
"""Render cue-bearing frames from a materialized DROID traj.hdf5 to MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_hdf5", type=Path)
    parser.add_argument("output_mp4", type=Path)
    parser.add_argument("--camera", default="exterior_image_1_left")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    key = f"saved_observation/{args.camera}"
    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.traj_hdf5, "r") as handle:
        if key not in handle:
            raise KeyError(f"Missing {key} in {args.traj_hdf5}")
        frames = handle[key]
        with imageio.get_writer(
            args.output_mp4,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            for index in range(frames.shape[0]):
                writer.append_data(frames[index])

    print(f"wrote {args.output_mp4} ({frames.shape[0]} frames at {args.fps:g} fps)")


if __name__ == "__main__":
    main()
