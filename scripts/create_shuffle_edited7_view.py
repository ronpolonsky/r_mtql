#!/usr/bin/env python3
"""Create a symlink-only DROID view for shuffle_v2 edited7 trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path


COLORS = ("black", "white", "orange")
OUTCOMES = {"success": 50, "failure": 15}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    view_dir = args.view_dir
    expected_links: set[Path] = set()

    for color in COLORS:
        for outcome, expected_count in OUTCOMES.items():
            source_group = source_root / f"{color}_edited7" / outcome
            if not source_group.is_dir():
                raise FileNotFoundError(source_group)
            episodes = sorted(
                (p for p in source_group.iterdir() if p.is_dir() and p.name.isdigit()),
                key=lambda p: int(p.name),
            )
            if len(episodes) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} {color}/{outcome} episodes, "
                    f"found {len(episodes)} in {source_group}"
                )
            for episode in episodes:
                source_file = episode / "traj.hdf5"
                if not source_file.is_file():
                    raise FileNotFoundError(source_file)
                target_file = (
                    view_dir / f"card_{color}" / outcome / episode.name / "traj.hdf5"
                )
                expected_links.add(target_file)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if target_file.is_symlink():
                    if target_file.resolve(strict=True) != source_file:
                        raise FileExistsError(f"Existing link points elsewhere: {target_file}")
                elif target_file.exists():
                    raise FileExistsError(f"Refusing to replace: {target_file}")
                else:
                    target_file.symlink_to(source_file)

    actual_links = {p for p in view_dir.rglob("traj.hdf5") if p.is_symlink()}
    if actual_links != expected_links:
        raise ValueError(
            f"View mismatch: expected {len(expected_links)} links, found {len(actual_links)}"
        )
    print(f"Validated {len(expected_links)} shuffle edited7 episodes: {view_dir}")


if __name__ == "__main__":
    main()
