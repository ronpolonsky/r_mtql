#!/usr/bin/env python3
"""Merge streamed DROID cache parts without touching the source dataset.

Each input part contains complete episodes and was produced by
``preprocess_droid_cache.py``.  The merger validates the part metadata, then
copies each array into a single staged cache using memory-mapped arrays.  The
final cache is published with an atomic directory rename.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


ARRAYS = (
    "actions",
    "cue_targets",
    "episode_success",
    "masks",
    "next_observations_image",
    "next_observations_proprio",
    "next_observations_wrist_image",
    "observations_image",
    "observations_proprio",
    "observations_wrist_image",
    "raw_actions",
    "rewards",
    "terminals",
)


def _read_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if tuple(metadata.get("arrays", ())) != tuple(sorted(ARRAYS)):
        raise ValueError(f"Unexpected array list in {metadata_path}")
    return metadata


def _validate_parts(part_dirs: list[Path]) -> tuple[list[dict], int, int]:
    metadata = [_read_metadata(path) for path in part_dirs]
    reference = metadata[0]
    invariant_keys = (
        "cache_format_version",
        "norm_stats_dir",
        "action_space",
        "gripper_action_space",
        "image_size",
        "n_success",
        "n_failure",
    )
    for part, current in zip(part_dirs, metadata):
        for key in invariant_keys:
            if current.get(key) != reference.get(key):
                raise ValueError(
                    f"Part {part} disagrees on {key}: "
                    f"{current.get(key)!r} != {reference.get(key)!r}"
                )
        expected_rows = int(current["num_transitions"])
        if int(current["num_episodes"]) <= 0:
            raise ValueError(f"Part {part} has no episodes")
        for name in ARRAYS:
            array_path = part / f"{name}.npy"
            if not array_path.is_file():
                raise FileNotFoundError(f"Missing array: {array_path}")
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if array.shape[0] != expected_rows:
                raise ValueError(
                    f"{array_path} has {array.shape[0]} rows, expected {expected_rows}"
                )
    transitions = sum(int(item["num_transitions"]) for item in metadata)
    episodes = sum(int(item["num_episodes"]) for item in metadata)
    return metadata, transitions, episodes


def _merge(part_dirs: list[Path], output: Path) -> tuple[int, int]:
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output cache: {output}"
        )
    staging = output.with_name(f"{output.name}.tmp-merge-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Temporary output already exists: {staging}")

    metadata, total_rows, total_episodes = _validate_parts(part_dirs)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for name in ARRAYS:
            first = np.load(
                part_dirs[0] / f"{name}.npy", mmap_mode="r", allow_pickle=False
            )
            destination = np.lib.format.open_memmap(
                staging / f"{name}.npy",
                mode="w+",
                dtype=first.dtype,
                shape=(total_rows,) + first.shape[1:],
            )
            offset = 0
            for part in part_dirs:
                source = np.load(
                    part / f"{name}.npy", mmap_mode="r", allow_pickle=False
                )
                if source.dtype != first.dtype or source.shape[1:] != first.shape[1:]:
                    raise ValueError(
                        f"Incompatible {name} shape/dtype in {part}: "
                        f"{source.shape}/{source.dtype} vs "
                        f"{first.shape}/{first.dtype}"
                    )
                end = offset + source.shape[0]
                destination[offset:end] = source
                offset = end
            if offset != total_rows:
                raise AssertionError(f"Copied {offset} rows for {name}, expected {total_rows}")
            destination.flush()
            del destination
            print(f"[droid-cache-merge] wrote {name} ({total_rows} rows)", flush=True)

        final_metadata = {
            "cache_format_version": metadata[0]["cache_format_version"],
            "dataset_path": "/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted",
            "norm_stats_dir": metadata[0]["norm_stats_dir"],
            "action_space": metadata[0]["action_space"],
            "gripper_action_space": metadata[0]["gripper_action_space"],
            "image_size": metadata[0]["image_size"],
            "n_success": metadata[0]["n_success"],
            "n_failure": metadata[0]["n_failure"],
            "num_transitions": total_rows,
            "num_episodes": total_episodes,
            "arrays": sorted(ARRAYS),
        }
        with (staging / "metadata.json").open("w") as handle:
            json.dump(final_metadata, handle, indent=2, sort_keys=True)
        os.replace(staging, output)
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return total_rows, total_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-parts", type=int, default=36)
    args = parser.parse_args()
    if args.num_parts <= 0:
        raise ValueError("--num-parts must be positive")
    part_dirs = [
        args.parts_root / f"part{index:02d}" for index in range(args.num_parts)
    ]
    missing = [str(path) for path in part_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing part directories: " + ", ".join(missing))
    total_rows, total_episodes = _merge(part_dirs, args.output)
    print(
        f"[droid-cache-merge] published {total_rows} transitions from "
        f"{total_episodes} episodes to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
