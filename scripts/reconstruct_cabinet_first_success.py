#!/usr/bin/env python3
"""Reconstruct Cabinet replay data with termination at first raw success.

The original Cabinet collector continued for 50 accumulated raw-success steps.
Those episodes therefore contain many reward=+1 transitions before their final
terminal transition.  With the corrected environment, the first reward=+1
transition is terminal.  Since dynamics and actions before that transition are
unchanged, the corrected episode is exactly a prefix of the original episode.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


TRANSITION_KEYS = (
    "obs_image",
    "obs_wrist_image",
    "obs_proprio",
    "next_obs_image",
    "next_obs_wrist_image",
    "next_obs_proprio",
    "actions",
    "rewards",
    "terminals",
    "masks",
)


def reconstruct_episode(source: Path, destination: Path) -> dict:
    """Write one corrected episode and return reconstruction statistics."""
    with np.load(source, allow_pickle=False) as data:
        missing = [key for key in TRANSITION_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{source} is missing keys: {missing}")

        rewards = data["rewards"]
        original_length = len(rewards)
        if original_length == 0:
            raise ValueError(f"{source} contains no transitions")

        unexpected_rewards = np.setdiff1d(np.unique(rewards), [-1.0, 1.0])
        if len(unexpected_rewards):
            raise ValueError(
                f"{source} has rewards outside {{-1, +1}}: {unexpected_rewards}"
            )

        success_indices = np.flatnonzero(rewards == 1.0)
        successful = len(success_indices) > 0
        corrected_length = int(success_indices[0] + 1) if successful else original_length

        output = {}
        for key in data.files:
            value = data[key]
            if key in TRANSITION_KEYS:
                if len(value) != original_length:
                    raise ValueError(
                        f"{source}: {key} has length {len(value)}, expected {original_length}"
                    )
                output[key] = value[:corrected_length].copy()
            else:
                output[key] = value.copy()

    # The collector treats both successful termination and an unsuccessful
    # episode boundary as terminal.  Rebuild these arrays explicitly so stale
    # delayed-success labels cannot survive the transformation.
    output["terminals"] = np.zeros(corrected_length, dtype=np.float32)
    output["masks"] = np.ones(corrected_length, dtype=np.float32)
    output["terminals"][-1] = 1.0
    output["masks"][-1] = 0.0

    if successful:
        if output["rewards"][-1] != 1.0:
            raise AssertionError(f"{source}: corrected successful episode does not end in +1")
        if np.count_nonzero(output["rewards"] == 1.0) != 1:
            raise AssertionError(f"{source}: corrected episode contains repeated +1 rewards")
    elif np.any(output["rewards"] == 1.0):
        raise AssertionError(f"{source}: unsuccessful episode contains a +1 reward")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **output)
    temporary.replace(destination)

    return {
        "successful": successful,
        "original_length": original_length,
        "corrected_length": corrected_length,
        "removed_transitions": original_length - corrected_length,
    }


def merge_episodes(episode_paths: list[Path], destination: Path) -> int:
    """Merge corrected episode files while loading only one episode at a time."""
    if not episode_paths:
        raise ValueError(f"Cannot create {destination}: no episodes supplied")

    total = 0
    shapes = {}
    dtypes = {}
    for path in episode_paths:
        with np.load(path, allow_pickle=False) as data:
            count = len(data["rewards"])
            total += count
            for key in TRANSITION_KEYS:
                shape = data[key].shape[1:]
                dtype = data[key].dtype
                if key in shapes and (shapes[key] != shape or dtypes[key] != dtype):
                    raise ValueError(f"Inconsistent {key} shape or dtype in {path}")
                shapes[key] = shape
                dtypes[key] = dtype

    merged = {
        key: np.empty((total, *shapes[key]), dtype=dtypes[key])
        for key in TRANSITION_KEYS
    }

    cursor = 0
    for path in episode_paths:
        with np.load(path, allow_pickle=False) as data:
            count = len(data["rewards"])
            target = slice(cursor, cursor + count)
            for key in TRANSITION_KEYS:
                merged[key][target] = data[key]
            cursor += count

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **merged)
    temporary.replace(destination)
    return total


def validate_merged(path: Path, expected_episodes: int) -> dict:
    """Validate reward/terminal invariants in a reconstructed merged split."""
    with np.load(path, allow_pickle=False) as data:
        rewards = data["rewards"]
        terminals = data["terminals"]
        masks = data["masks"]

        if not (len(rewards) == len(terminals) == len(masks)):
            raise AssertionError(f"{path}: reward/terminal/mask lengths differ")
        if int(np.count_nonzero(terminals == 1.0)) != expected_episodes:
            raise AssertionError(f"{path}: terminal count does not match episode count")
        if np.any(masks != 1.0 - terminals):
            raise AssertionError(f"{path}: masks are inconsistent with terminals")
        if np.any((rewards == 1.0) & (terminals != 1.0)):
            raise AssertionError(f"{path}: found a nonterminal +1 reward")

        return {
            "transitions": int(len(rewards)),
            "episodes": int(np.count_nonzero(terminals == 1.0)),
            "successful_episodes": int(np.count_nonzero(rewards == 1.0)),
            "nonterminal_positive_rewards": int(
                np.count_nonzero((rewards == 1.0) & (terminals != 1.0))
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}"
        )

    source_episode_dir = args.input_dir / "episodes"
    source_paths = sorted(source_episode_dir.glob("ep_*.npz"))
    if args.max_episodes > 0:
        source_paths = source_paths[: args.max_episodes]
    if not source_paths:
        raise FileNotFoundError(f"No ep_*.npz files found in {source_episode_dir}")

    output_episode_dir = args.output_dir / "episodes"
    output_episode_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "episodes": len(source_paths),
        "successful_episodes": 0,
        "original_transitions": 0,
        "corrected_transitions": 0,
        "removed_transitions": 0,
    }
    corrected_paths = []
    for index, source in enumerate(source_paths, start=1):
        destination = output_episode_dir / source.name
        stats = reconstruct_episode(source, destination)
        corrected_paths.append(destination)
        totals["successful_episodes"] += int(stats["successful"])
        totals["original_transitions"] += stats["original_length"]
        totals["corrected_transitions"] += stats["corrected_length"]
        totals["removed_transitions"] += stats["removed_transitions"]
        print(
            f"[{index:03d}/{len(source_paths):03d}] {source.name}: "
            f"{stats['original_length']} -> {stats['corrected_length']}"
        )

    summary = dict(totals)
    if not args.skip_merge:
        shuffled = list(corrected_paths)
        random.Random(args.seed).shuffle(shuffled)
        validation_count = max(1, int(len(shuffled) * args.val_fraction))
        train_paths = shuffled[:-validation_count]
        validation_paths = shuffled[-validation_count:]

        train_file = args.output_dir / "train_cabinet_dataset.npz"
        validation_file = args.output_dir / "val_cabinet_dataset.npz"
        print(f"Merging {len(train_paths)} train episodes into {train_file}")
        merge_episodes(train_paths, train_file)
        print(f"Merging {len(validation_paths)} validation episodes into {validation_file}")
        merge_episodes(validation_paths, validation_file)

        summary["train"] = validate_merged(train_file, len(train_paths))
        summary["validation"] = validate_merged(
            validation_file, len(validation_paths)
        )

    summary_path = args.output_dir / "reconstruction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Reconstruction complete: {args.output_dir}")


if __name__ == "__main__":
    main()
