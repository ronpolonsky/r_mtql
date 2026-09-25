#!/usr/bin/env python3
"""Materialize the target-only cue in a separate DROID dataset copy.

The source dataset is never opened for writing.  The destination must already
be an independent copy.  The target is inferred from the destination path,
which is authoritative for counterfactual failures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


BASE_KEY = "saved_observation/exterior_image_1_left"
CUE_VERSION = "target_dots_v1"


def target_from_path(path: Path, root: Path) -> int:
    parts = path.relative_to(root).parts
    if parts and parts[0].startswith("target_"):
        value = parts[0][len("target_") :]
    elif len(parts) > 1 and parts[0] == "arbitrary" and parts[1].startswith("target_num_"):
        value = parts[1][len("target_num_") :]
    else:
        raise ValueError(f"Cannot infer target from episode path: {path}")
    if value not in {"1", "2", "3"}:
        raise ValueError(f"Target must be 1, 2, or 3: {path}")
    return int(value)


def draw_cue(frames: np.ndarray, target: int) -> None:
    """Draw a high-contrast three-slot cue in the empty upper-left scene area."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 (N,H,W,3) frames, got {frames.shape} {frames.dtype}")
    _, height, width, _ = frames.shape
    scale = min(height, width)
    margin = max(4, int(round(scale * 0.025)))
    panel_height = max(28, int(round(height * 0.27)))
    panel_width = max(96, int(round(width * 0.41)))
    panel_height = min(panel_height, height - margin)
    panel_width = min(panel_width, width - margin)
    border = max(1, int(round(scale * 0.012)))
    radius = max(6, int(round(scale * 0.055)))
    y0, x0 = margin, margin
    y1, x1 = y0 + panel_height, x0 + panel_width

    frames[:, y0:y1, x0:x1, :] = 0
    frames[:, y0:y0 + border, x0:x1, :] = 255
    frames[:, y1 - border:y1, x0:x1, :] = 255
    frames[:, y0:y1, x0:x0 + border, :] = 255
    frames[:, y0:y1, x1 - border:x1, :] = 255

    yy, xx = np.ogrid[:height, :width]
    center_y = y0 + panel_height // 2
    centers_x = np.linspace(x0 + panel_width * 0.22, x0 + panel_width * 0.78, 3).round().astype(int)
    inner_radius = max(1, radius - border)
    for slot, center_x in enumerate(centers_x, start=1):
        distance = (yy - center_y) ** 2 + (xx - center_x) ** 2
        outer = distance <= radius ** 2
        inner = distance <= inner_radius ** 2
        frames[:, outer, :] = 255
        if target >= slot:
            frames[:, inner, :] = 255
        else:
            frames[:, inner, :] = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    episodes = sorted(
        path for path in root.rglob("traj.hdf5")
        if path.parent.name.isdigit()
    )
    if not episodes:
        raise ValueError(f"No numeric episode traj.hdf5 files under {root}")
    for index, path in enumerate(episodes, start=1):
        target = target_from_path(path, root)
        with h5py.File(path, "r+") as handle:
            if BASE_KEY not in handle:
                raise KeyError(f"Missing {BASE_KEY} in {path}")
            dataset = handle[BASE_KEY]
            if dataset.ndim != 4 or dataset.shape[-1] != 3 or dataset.dtype != np.uint8:
                raise ValueError(f"Unexpected base image dataset in {path}: {dataset.shape} {dataset.dtype}")
            for start in range(0, dataset.shape[0], args.chunk_size):
                stop = min(start + args.chunk_size, dataset.shape[0])
                frames = np.asarray(dataset[start:stop])
                draw_cue(frames, target)
                dataset[start:stop] = frames
            handle.attrs["pixel_cue_version"] = CUE_VERSION
            handle.attrs["pixel_cue_target"] = target
        if index == 1 or index % 10 == 0 or index == len(episodes):
            print(f"processed {index}/{len(episodes)}: {path.relative_to(root)} target={target}", flush=True)


if __name__ == "__main__":
    main()
