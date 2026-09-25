#!/usr/bin/env python3
"""Create a symlink-only, three-color view of shuffle edited5 episodes."""

from __future__ import annotations

import argparse
from pathlib import Path


COLORS = ("black", "white", "orange")
OUTCOMES = {"success": 50, "failure": 16}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    view_dir = args.view_dir
    expected_links: set[Path] = set()
    counts: dict[tuple[str, str], int] = {}

    for color in COLORS:
        for outcome, expected_count in OUTCOMES.items():
            source_group = source_root / f"{color}_edited5" / outcome
            if not source_group.is_dir():
                raise FileNotFoundError(f"Missing source directory: {source_group}")
            episodes = sorted(
                (
                    path
                    for path in source_group.iterdir()
                    if path.is_dir() and path.name.isdigit()
                ),
                key=lambda path: int(path.name),
            )
            if len(episodes) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} {color}/{outcome} episodes, "
                    f"found {len(episodes)} in {source_group}"
                )
            counts[(color, outcome)] = len(episodes)

            for episode in episodes:
                source_file = episode / "traj.hdf5"
                if not source_file.is_file():
                    raise FileNotFoundError(f"Missing trajectory: {source_file}")
                target_file = (
                    view_dir
                    / f"card_{color}"
                    / outcome
                    / episode.name
                    / "traj.hdf5"
                )
                expected_links.add(target_file)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if target_file.is_symlink():
                    if target_file.resolve(strict=True) != source_file.resolve(strict=True):
                        raise FileExistsError(
                            f"Existing link points elsewhere: {target_file}"
                        )
                elif target_file.exists():
                    raise FileExistsError(f"Refusing to replace: {target_file}")
                else:
                    target_file.symlink_to(source_file.resolve(strict=True))

    actual_links = {
        path
        for path in view_dir.rglob("traj.hdf5")
        if path.is_symlink()
    }
    if actual_links != expected_links:
        raise ValueError(
            "Shuffle view contains missing or unexpected trajectory links: "
            f"expected={len(expected_links)}, found={len(actual_links)}"
        )

    total = sum(counts.values())
    print(f"Validated {total} shuffle edited5 episodes (symlink view):")
    for color in COLORS:
        print(
            f"  {color}: success={counts[(color, 'success')]}, "
            f"failure={counts[(color, 'failure')]}, "
            f"total={counts[(color, 'success')] + counts[(color, 'failure')]}"
        )
    print(f"Dataset view: {view_dir}")


if __name__ == "__main__":
    main()
