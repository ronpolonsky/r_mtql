#!/usr/bin/env python3
"""Create balanced, read-only-linked views for v2 cache preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE = Path("/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted")
DEFAULT_OUTPUT = Path(
    "/iris/u/ronpo/projects/new_mtql_candy_scooping/"
    ".candy_scoop_v2_jitted_parts"
)
DEFAULT_NUM_PARTS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-parts", type=int, default=DEFAULT_NUM_PARTS)
    args = parser.parse_args()
    output = args.output
    num_parts = args.num_parts
    if num_parts < 1:
        raise ValueError("--num-parts must be positive")
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    episodes = sorted(
        (path.parent for path in SOURCE.rglob("traj.hdf5") if path.is_symlink()),
        key=lambda path: tuple(
            int(part) if part.isdigit() else part
            for part in path.relative_to(SOURCE).parts
        ),
    )
    if len(episodes) != 144:
        raise ValueError(f"Expected 144 valid episodes, found {len(episodes)}")

    bins: list[list[Path]] = [[] for _ in range(num_parts)]
    sizes = [0] * num_parts
    # Greedy largest-first assignment keeps each part's HDF5 load comparable.
    for episode in sorted(episodes, key=lambda p: p.joinpath("traj.hdf5").stat().st_size, reverse=True):
        part = min(range(num_parts), key=sizes.__getitem__)
        bins[part].append(episode)
        sizes[part] += episode.joinpath("traj.hdf5").stat().st_size

    output.mkdir(parents=True)
    manifest = {"source": str(SOURCE), "parts": []}
    for index, part_episodes in enumerate(bins):
        part_root = output / f"part{index:02d}"
        part_root.mkdir()
        for episode in part_episodes:
            relative_episode = episode.relative_to(SOURCE)
            for source_file in episode.rglob("*"):
                if not source_file.is_file() and not source_file.is_symlink():
                    continue
                relative_file = source_file.relative_to(SOURCE)
                destination = part_root / relative_file
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source_file)
        manifest["parts"].append(
            {
                "name": part_root.name,
                "episodes": len(part_episodes),
                "hdf5_bytes": sizes[index],
            }
        )

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
