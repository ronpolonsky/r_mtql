#!/usr/bin/env python3
"""Audit edited7 motion timing and virtual cue retention.

This report is independent of model training.  It reads only the edited7
HDF5 action stream and applies the same index construction used by the
device-cache adapter, so candidate H/S choices can be declared before any
benchmark checkpoint is evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from utils.mtql_shuffle_device_cache import build_shuffle_history_indices


DEFAULT_CANDIDATES = (
    (7, 2),
    (10, 6),
    (14, 6),
    (14, 7),
    (19, 7),
)


def first_sustained_motion(actions: np.ndarray) -> int:
    active = np.any(np.abs(np.asarray(actions[:, :6])) > 1e-4, axis=1)
    starts = np.flatnonzero(active[:-1] & active[1:])
    if not len(starts):
        raise ValueError("Trajectory has no two-frame sustained arm motion")
    return int(starts[0])


def discover(dataset: Path) -> list[dict]:
    episodes = []
    for path in sorted(
        p for p in dataset.rglob("traj.hdf5") if p.parent.name.isdigit()
    ):
        with h5py.File(path, "r") as handle:
            actions = np.asarray(handle["action/executed_action"], dtype=np.float32)
        relative = path.relative_to(dataset)
        outcome = "success" if "success" in relative.parts else "failure"
        color = next(
            (part for part in relative.parts if part in {"black_edited7", "white_edited7", "orange_edited7"}),
            "unknown",
        )
        episodes.append(
            {
                "path": str(relative),
                "color": color,
                "outcome": outcome,
                "length": int(len(actions)),
                "motion_start": first_sustained_motion(actions),
            }
        )
    if not episodes:
        raise ValueError(f"No numeric traj.hdf5 episodes under {dataset}")
    return episodes


def summarize(
    dataset: Path,
    episodes: list[dict],
    candidates: tuple[tuple[int, int], ...],
) -> dict:
    lengths = np.asarray([episode["length"] for episode in episodes], dtype=np.int64)
    starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )
    ends = np.cumsum(lengths, dtype=np.int64) - 1
    all_indices = {
        index: (int(start), int(end))
        for index, (start, end) in enumerate(zip(starts, ends))
    }

    # Use the production helper over a contiguous synthetic transition stream.
    episode_starts = np.asarray([item[0] for item in all_indices.values()], dtype=np.int64)
    episode_ends = np.asarray([item[1] for item in all_indices.values()], dtype=np.int64)
    rows = []
    for history_length, hist_stride in candidates:
        indices, padding = build_shuffle_history_indices(
            episode_starts,
            episode_ends,
            size=int(lengths.sum()),
            hist_length=history_length,
            hist_stride=hist_stride,
            cue_frames=7,
        )
        onset_counts = []
        terminal_counts = []
        onset_all = []
        terminal_any = []
        terminal_two = []
        for episode_index, episode in enumerate(episodes):
            start, end = all_indices[episode_index]
            onset = start + int(episode["motion_start"])
            onset_history = indices[onset]
            terminal_history = indices[end]
            onset_unique = len(set(onset_history.tolist()))
            terminal_unique_cues = len(
                {int(index - start) for index in terminal_history if 0 <= index - start < 7}
            )
            onset_counts.append(onset_unique)
            terminal_counts.append(terminal_unique_cues)
            onset_all.append(
                len({int(index - start) for index in onset_history if 0 <= index - start < 7}) == 7
            )
            terminal_any.append(terminal_unique_cues >= 1)
            terminal_two.append(terminal_unique_cues >= 2)
        rows.append(
            {
                "history_length": history_length,
                "hist_stride": hist_stride,
                "nominal_span_seconds": history_length * hist_stride / 10.0,
                "mean_unique_cues_at_motion_onset": float(np.mean(onset_counts)),
                "fraction_all_7_cues_at_motion_onset": float(np.mean(onset_all)),
                "mean_unique_cues_at_terminal": float(np.mean(terminal_counts)),
                "fraction_at_least_1_cue_at_terminal": float(np.mean(terminal_any)),
                "fraction_at_least_2_cues_at_terminal": float(np.mean(terminal_two)),
            }
        )
    return {
        "dataset": str(dataset),
        "episode_count": len(episodes),
        "transition_count": int(lengths.sum()),
        "length_summary": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
        },
        "motion_start_local_indices": sorted(set(int(e["motion_start"]) for e in episodes)),
        "outcomes": {
            "success": int(sum(e["outcome"] == "success" for e in episodes)),
            "failure": int(sum(e["outcome"] == "failure" for e in episodes)),
        },
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.dataset, discover(args.dataset), DEFAULT_CANDIDATES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for row in report["candidates"]:
        print(
            f"H{row['history_length']} S{row['hist_stride']}: "
            f"onset={row['fraction_all_7_cues_at_motion_onset']:.2f} all7, "
            f"terminal>=1={row['fraction_at_least_1_cue_at_terminal']:.2f}, "
            f"terminal>=2={row['fraction_at_least_2_cues_at_terminal']:.2f}"
        )


if __name__ == "__main__":
    main()
