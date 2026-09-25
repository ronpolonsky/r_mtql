#!/usr/bin/env python3
"""Create a cache with new OpenPI stats without rereading the image arrays.

The existing cache stores normalized proprioception/actions plus raw_actions.
This utility re-normalizes those small arrays and hard-links the immutable image
and bookkeeping arrays into a new cache directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from utils.mtql_droid import load_openpi_norm_stats


REWRITE = {
    "actions",
    "observations_proprio",
    "next_observations_proprio",
}


def _unnormalize(values: np.ndarray, stats) -> np.ndarray:
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("q01/q99 are required for cache re-normalization")
    q01 = stats.q01[..., : values.shape[-1]]
    q99 = stats.q99[..., : values.shape[-1]]
    return (values + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _normalize(values: np.ndarray, stats) -> np.ndarray:
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("q01/q99 are required for cache re-normalization")
    q01 = stats.q01[..., : values.shape[-1]]
    q99 = stats.q99[..., : values.shape[-1]]
    return np.asarray(
        (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0,
        dtype=np.float32,
    )


def _write_array(path: Path, values: np.ndarray) -> None:
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=values.shape,
    )
    output[:] = values
    output.flush()
    del output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--output-cache", required=True, type=Path)
    parser.add_argument("--old-stats", required=True, type=Path)
    parser.add_argument("--new-stats", required=True, type=Path)
    args = parser.parse_args()

    source = args.input_cache
    output = args.output_cache
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {output}")

    with (source / "metadata.json").open() as handle:
        metadata = json.load(handle)
    old_stats = load_openpi_norm_stats(args.old_stats)
    new_stats = load_openpi_norm_stats(args.new_stats)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.name}.tmp-renorm-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Temporary cache already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for name in metadata["arrays"]:
            source_array = source / f"{name}.npy"
            destination = staging / f"{name}.npy"
            if name in REWRITE:
                continue
            # The cache and output are on the same shared filesystem. Hard-link
            # large immutable arrays so this does not duplicate image storage.
            os.link(source_array, destination)

        old_state = old_stats["state"]
        new_state = new_stats["state"]
        for name in ("observations_proprio", "next_observations_proprio"):
            values = np.load(source / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            raw = _unnormalize(values, old_state)
            _write_array(staging / f"{name}.npy", _normalize(raw, new_state))

        raw_actions = np.load(source / "raw_actions.npy", mmap_mode="r", allow_pickle=False)
        _write_array(staging / "actions.npy", _normalize(raw_actions, new_stats["actions"]))

        metadata["norm_stats_dir"] = str(args.new_stats.resolve())
        with (staging / "metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        os.replace(staging, output)
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"Created {output} with {metadata['num_episodes']} episodes and "
        f"{metadata['num_transitions']} transitions using {args.new_stats}",
        flush=True,
    )


if __name__ == "__main__":
    main()
