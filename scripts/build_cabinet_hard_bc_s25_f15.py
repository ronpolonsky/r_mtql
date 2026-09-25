#!/usr/bin/env python3
"""Rebuild the exact 25-success/15-failure hard cabinet subset.

The episode identities live in
``analysis/cabinet_offline_rl_hard_bc/selection.json``.  This script never
changes the source dataset; it creates a separate merged training archive and
copies the original validation archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "cabinet-memory-sim/ManiSkill/cabinet_dataset_two_cams_more_rand"
)
DEFAULT_SELECTION = ROOT / "analysis/cabinet_offline_rl_hard_bc/selection.json"
DEFAULT_OUTPUT = (
    ROOT
    / "cabinet-memory-sim/ManiSkill/"
    / "cabinet_dataset_two_cams_more_rand_hard_bc_s25_f15"
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_episode(path: Path, expected_success: bool) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        missing = sorted(set(TRANSITION_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
        episode = {key: archive[key].copy() for key in TRANSITION_KEYS}

    lengths = {len(value) for value in episode.values()}
    if len(lengths) != 1:
        raise ValueError(f"{path} has inconsistent transition lengths: {lengths}")
    if not bool(episode["terminals"][-1]):
        raise ValueError(f"{path} does not end at a terminal transition")
    observed_success = bool(np.any(episode["rewards"] > 0))
    if observed_success != expected_success:
        raise ValueError(
            f"{path} expected success={expected_success}, "
            f"observed success={observed_success}"
        )
    return episode


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    selection_path = args.selection.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}. "
            "Remove it explicitly before rebuilding."
        )

    selection = json.loads(selection_path.read_text())
    success_ids = (
        list(selection["efficient_successes"])
        + list(selection["inefficient_successes"])
    )
    failure_ids = list(selection["targeted_failures"])
    if len(success_ids) != 25 or len(failure_ids) != 15:
        raise ValueError(
            f"Expected 25 successes and 15 failures, got "
            f"{len(success_ids)} and {len(failure_ids)}"
        )
    episode_ids = success_ids + failure_ids
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Selection contains duplicate episode IDs")

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        lengths: dict[str, int] = {}
        for index, episode_id in enumerate(episode_ids, 1):
            path = source / "episodes" / f"{episode_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as archive:
                missing = sorted(set(TRANSITION_KEYS) - set(archive.files))
                if missing:
                    raise ValueError(f"{path} is missing keys: {missing}")
                rewards = archive["rewards"]
                terminals = archive["terminals"]
                lengths[episode_id] = len(rewards)
                if len(terminals) != len(rewards) or not bool(terminals[-1]):
                    raise ValueError(f"{path} has an invalid terminal boundary")
                observed_success = bool(np.any(rewards > 0))
                expected_success = episode_id in success_ids
                if observed_success != expected_success:
                    raise ValueError(
                        f"{path} expected success={expected_success}, "
                        f"observed success={observed_success}"
                    )
            print(
                f"[{index:02d}/{len(episode_ids)}] validated {episode_id}: "
                f"{lengths[episode_id]} transitions",
                flush=True,
            )

        total_transitions = sum(lengths.values())
        array_dir = temp_dir / "arrays"
        array_dir.mkdir()
        first_path = source / "episodes" / f"{episode_ids[0]}.npz"
        merged: dict[str, np.memmap] = {}
        with np.load(first_path) as first:
            for key in TRANSITION_KEYS:
                sample = first[key]
                merged[key] = np.lib.format.open_memmap(
                    array_dir / f"{key}.npy",
                    mode="w+",
                    dtype=sample.dtype,
                    shape=(total_transitions, *sample.shape[1:]),
                )
        merged["actor_mask"] = np.lib.format.open_memmap(
            array_dir / "actor_mask.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_transitions,),
        )

        offset = 0
        for index, episode_id in enumerate(episode_ids, 1):
            length = lengths[episode_id]
            end = offset + length
            path = source / "episodes" / f"{episode_id}.npz"
            with np.load(path) as archive:
                for key in TRANSITION_KEYS:
                    value = archive[key]
                    if value.shape[0] != length:
                        raise ValueError(
                            f"{path}:{key} has {value.shape[0]} rows, expected {length}"
                        )
                    if value.shape[1:] != merged[key].shape[1:]:
                        raise ValueError(
                            f"{path}:{key} has trailing shape {value.shape[1:]}, "
                            f"expected {merged[key].shape[1:]}"
                        )
                    merged[key][offset:end] = value
            merged["actor_mask"][offset:end] = (
                1.0 if episode_id in success_ids else 0.0
            )
            offset = end
            print(
                f"[{index:02d}/{len(episode_ids)}] copied {episode_id}",
                flush=True,
            )

        for value in merged.values():
            value.flush()
        train_path = temp_dir / "train_cabinet_dataset.npz"
        print(f"Writing {train_path} ...", flush=True)
        np.savez_compressed(train_path, **merged)
        for value in merged.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
        merged.clear()
        shutil.rmtree(array_dir)

        source_val = source / "val_cabinet_dataset.npz"
        if not source_val.is_file():
            raise FileNotFoundError(source_val)
        shutil.copy2(source_val, temp_dir / source_val.name)

        success_transitions = sum(lengths[name] for name in success_ids)
        failure_transitions = sum(lengths[name] for name in failure_ids)
        manifest = {
            "version": "hard_bc_s25_f15_v1",
            "source_dataset": str(source),
            "selection_file": str(selection_path),
            "selection_method": "manual two-camera/full-proprio audit",
            "successful_episodes": success_ids,
            "failed_episodes": failure_ids,
            "episode_lengths": lengths,
            "num_successful_episodes": len(success_ids),
            "num_failed_episodes": len(failure_ids),
            "success_transitions": success_transitions,
            "failure_transitions": failure_transitions,
            "total_transitions": total_transitions,
            "actor_mask": "1 for successful episodes; 0 for failed episodes",
            "validation": "copied unchanged from source dataset",
        }
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temp_dir / "train_episodes.txt").write_text(
            "\n".join(episode_ids) + "\n"
        )
        temp_dir.rename(output)
        print(
            f"Created {output}: {len(success_ids)} successes + "
            f"{len(failure_ids)} failures, "
            f"{manifest['total_transitions']} transitions",
            flush=True,
        )
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
