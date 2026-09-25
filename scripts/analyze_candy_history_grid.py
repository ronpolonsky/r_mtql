#!/usr/bin/env python3
"""Analyze Candy-Scoop history/stride coverage without loading camera images.

This reproduces the event-coverage analysis in ``experiment_analysis.md``
using only saved Cartesian positions from the raw DROID HDF5 files.  It is
deliberately independent of the training loader so that changing image
preprocessing or augmentation cannot change the history recommendation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np


TARGETS = (1, 2, 3)
DEFAULT_STRIDES = (1, 5, 7, 10, 12, 15, 17, 20, 24, 25, 30, 40, 50, 60)


def _read_episode(path: Path, dataset: Path) -> dict | None:
    relative = path.relative_to(dataset)
    parts = {part.lower() for part in relative.parts}
    if "success" in parts:
        outcome = "success"
    elif "failure" in parts:
        outcome = "failure"
    else:
        return None

    target = None
    for part in relative.parts:
        if part.startswith("target_") and part[7:].isdigit():
            target = int(part[7:])
            break
        if part.startswith("target_num_") and part[11:].isdigit():
            target = int(part[11:])
            break
    if target not in TARGETS:
        return None

    with h5py.File(path, "r") as handle:
        # Older DROID exports stored this stream under action/.  The v2
        # candy-scoop export stores it under saved_observation/.  Both are
        # the same 10 Hz Cartesian-position stream for this analysis.
        if "action" in handle and "cartesian_position" in handle["action"]:
            position_dataset = handle["action"]["cartesian_position"]
        elif (
            "saved_observation" in handle
            and "cartesian_position" in handle["saved_observation"]
        ):
            position_dataset = handle["saved_observation"]["cartesian_position"]
        else:
            raise KeyError(
                f"{path} has neither action/cartesian_position nor "
                "saved_observation/cartesian_position"
            )
        length = int(position_dataset.shape[0])
        # Failures are included in dataset counts but do not contribute to
        # scoop-event recall. Avoid reading their large camera-backed HDF5
        # chunks; the success-only event analysis is unchanged.
        positions = (
            np.asarray(
                position_dataset[:, :3],
                dtype=np.float32,
            )
            if outcome == "success"
            else None
        )
    return {
        "path": str(relative),
        "outcome": outcome,
        "target": target,
        "length": length,
        "positions": positions,
    }


def discover_episodes(dataset: Path) -> list[dict]:
    # Match the production loader: ignore incomplete tmp/session_* files and
    # keep only numeric episode directories. Parallel reads matter for the
    # large target-3 files on the shared filesystem.
    paths = sorted(
        path for path in dataset.rglob("traj.hdf5") if path.parent.name.isdigit()
    )
    episodes = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_read_episode, path, dataset) for path in paths]
        for index, future in enumerate(as_completed(futures), start=1):
            episode = future.result()
            if episode is not None:
                episodes.append(episode)
            if index % 25 == 0 or index == len(futures):
                print(f"read {index}/{len(futures)} HDF5 files", flush=True)
    episodes.sort(key=lambda episode: episode["path"])
    if not episodes:
        raise ValueError(f"No target-labeled episodes found under {dataset}")
    return episodes


def detect_source_events(positions: np.ndarray) -> list[tuple[int, int]]:
    """Return event windows from source entry through the re-arm crossing."""
    inside = (
        (positions[:, 0] > 0.45)
        & (positions[:, 1] > 0.02)
        & (positions[:, 1] < 0.12)
        & (positions[:, 2] < 0.22)
    )
    events: list[tuple[int, int]] = []
    armed = True
    active_start: int | None = None
    for index in range(len(positions)):
        if armed and inside[index]:
            active_start = index
            armed = False
        if not armed and positions[index, 1] < -0.04:
            if active_start is not None:
                events.append((active_start, index))
                active_start = None
            armed = True
    if active_start is not None:
        events.append((active_start, len(positions) - 1))
    return events


def history_indices(current: int, length: int, stride: int) -> np.ndarray:
    """Match HistoryDataset's t-L*S,...,t-S indices with start clamping."""
    raw = current - np.arange(length, 0, -1, dtype=np.int64) * stride
    return np.maximum(raw, 0)


def event_is_seen(history: np.ndarray, event: tuple[int, int]) -> bool:
    start, end = event
    return bool(np.any((history >= start) & (history <= end)))


def compute_grid(successes: list[dict], strides: tuple[int, ...]) -> list[dict]:
    rows = []
    for length in range(1, 26):
        for stride in strides:
            terminal_hits = 0
            terminal_total = 0
            terminal_all = 0
            positive_chunk_hits = 0
            positive_chunk_total = 0
            successful_episode_all = 0
            for episode in successes:
                terminal = episode["length"] - 1
                events = episode["events"]
                terminal_history = history_indices(terminal, length, stride)
                covered = [event_is_seen(terminal_history, event) for event in events]
                terminal_hits += int(sum(covered))
                terminal_total += len(events)
                terminal_all += int(bool(events) and all(covered))

                if events and all(covered):
                    successful_episode_all += 1

                for current in range(max(0, terminal - 24), terminal + 1):
                    # A chunk beginning before an event has happened should
                    # not be penalized for failing to remember the future.
                    past_events = [event for event in events if event[1] <= current]
                    current_history = history_indices(current, length, stride)
                    positive_chunk_total += 1
                    if all(event_is_seen(current_history, event) for event in past_events):
                        positive_chunk_hits += 1

            rows.append(
                {
                    "history_length": length,
                    "hist_stride": stride,
                    "step_span": length * stride,
                    "nominal_seconds": length * stride / 10.0,
                    "sampling_seconds": stride / 10.0,
                    "actor_tokens": length + 2,
                    "critic_tokens": length + 27,
                    "terminal_event_recall": terminal_hits / terminal_total
                    if terminal_total
                    else 0.0,
                    "terminal_events_seen": terminal_hits,
                    "terminal_events_total": terminal_total,
                    "terminal_all_events_fraction": terminal_all / len(successes),
                    "terminal_all_events_episodes": terminal_all,
                    "positive_chunk_all_events_fraction": positive_chunk_hits
                    / positive_chunk_total,
                    "positive_chunk_all_events": positive_chunk_hits,
                    "positive_chunk_total": positive_chunk_total,
                    "episodes_with_all_terminal_events": successful_episode_all,
                }
            )
    return rows


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def build_report(episodes: list[dict], rows: list[dict], strides: tuple[int, ...]) -> str:
    successes = [episode for episode in episodes if episode["outcome"] == "success"]
    failures = [episode for episode in episodes if episode["outcome"] == "failure"]
    counts = Counter((episode["outcome"], episode["target"]) for episode in episodes)
    transition_count = sum(episode["length"] for episode in episodes)
    success_lengths = defaultdict(list)
    for episode in successes:
        success_lengths[episode["target"]].append(episode["length"])

    event_counts = Counter()
    event_durations = []
    inter_event = []
    for episode in successes:
        events = episode["events"]
        event_counts[episode["target"]] += len(events)
        event_durations.extend(end - start + 1 for start, end in events)
        inter_event.extend(
            (events[index][0] - events[index - 1][0]) / 10.0
            for index in range(1, len(events))
        )

    def row(length: int, stride: int) -> dict:
        return next(
            item
            for item in rows
            if item["history_length"] == length and item["hist_stride"] == stride
        )

    # Highlight the strongest full-event-coverage option for each history
    # length, then the shortest span that achieves at least 95% recall.
    best_by_length = []
    for length in range(1, 26):
        candidates = [item for item in rows if item["history_length"] == length]
        candidates.sort(
            key=lambda item: (
                item["terminal_event_recall"],
                item["positive_chunk_all_events_fraction"],
                -item["step_span"],
            ),
            reverse=True,
        )
        best_by_length.append(candidates[0])
    practical = [item for item in rows if item["terminal_event_recall"] >= 0.95]
    practical.sort(
        key=lambda item: (
            item["step_span"],
            -item["terminal_event_recall"],
            -item["history_length"],
        )
    )
    # Prefer the smallest transformer critic that clears the practical
    # coverage threshold.  The old dataset happened to select H19/S24, but
    # this must be derived from the current dataset rather than hard-coded.
    balanced = min(
        practical or rows,
        key=lambda item: (
            item["critic_tokens"],
            -item["terminal_event_recall"],
            item["step_span"],
        ),
    )
    lower_cost = min(
        practical,
        key=lambda item: (
            item["critic_tokens"],
            -item["terminal_event_recall"],
            item["step_span"],
        ),
    )
    full_coverage_candidates = [
        item for item in rows if item["terminal_event_recall"] >= 0.999
    ]
    if full_coverage_candidates:
        full_coverage = min(
            full_coverage_candidates,
            key=lambda item: (item["step_span"], item["critic_tokens"]),
        )
    else:
        # Some datasets have irreducible event-boundary aliasing on the
        # available sampling grid. Report the actual maximum instead of
        # pretending that a 99.9% setting exists.
        full_coverage = max(
            rows,
            key=lambda item: (
                item["terminal_event_recall"],
                item["terminal_all_events_fraction"],
                -item["step_span"],
            ),
        )

    lines = [
        "# Candy-Scoop History/Stride Analysis with History Length <= 25",
        "",
        "## Scope and method",
        "",
        "This is the same raw-DROID, geometry-based history analysis as "
        "`experiment_analysis.md`, restricted to history lengths 1--25. "
        "It reads only saved Cartesian positions, so image augmentation and "
        "camera loading do not affect the result.",
        "",
        "The scoop detector uses the existing report thresholds: source-region "
        "entry at `x > 0.45`, `0.02 < y < 0.12`, `z < 0.22`, with the event "
        "window ending at the subsequent `y < -0.04` re-arm crossing. History "
        "samples are `t-L*S,...,t-S`, clamped at the episode start, matching "
        "`HistoryDataset`.",
        "",
        "## Bottom line",
        "",
        f"The recommended capped setting is **H{balanced['history_length']}/S{balanced['hist_stride']}**: "
        f"a {balanced['nominal_seconds']:.1f}-second nominal span, "
        f"{balanced['terminal_event_recall']:.1%} terminal event recall, "
        f"{balanced['terminal_all_events_fraction']:.1%} of successful terminals "
        f"with every event represented, and {balanced['critic_tokens']} critic "
        "tokens. It is the best balance between the original report's near-" 
        "complete event coverage and the new history-length cap.",
        "",
        "| Role | Setting | Terminal event recall | Terminal all-events | Span | Critic tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| Lower-cost alternative | H{lower_cost['history_length']}/S{lower_cost['hist_stride']} | "
        f"{lower_cost['terminal_event_recall']:.1%} | {lower_cost['terminal_all_events_fraction']:.1%} | "
        f"{lower_cost['nominal_seconds']:.1f} s | {lower_cost['critic_tokens']} |",
        f"| Recommended balance | H{balanced['history_length']}/S{balanced['hist_stride']} | "
        f"{balanced['terminal_event_recall']:.1%} | {balanced['terminal_all_events_fraction']:.1%} | "
        f"{balanced['nominal_seconds']:.1f} s | {balanced['critic_tokens']} |",
        f"| Maximum measured coverage | H{full_coverage['history_length']}/S{full_coverage['hist_stride']} | "
        f"{full_coverage['terminal_event_recall']:.1%} | {full_coverage['terminal_all_events_fraction']:.1%} | "
        f"{full_coverage['nominal_seconds']:.1f} s | {full_coverage['critic_tokens']} |",
        "",
        "## Dataset and event validation",
        "",
        f"The scan found **{len(episodes)} episodes** and **{transition_count:,} "
        "transitions**: "
        f"{len(successes)} successes and {len(failures)} failures.",
        "",
        "| Outcome | Target 1 | Target 2 | Target 3 | Total |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Success | {counts[('success', 1)]} | {counts[('success', 2)]} | "
        f"{counts[('success', 3)]} | {len(successes)} |",
        f"| Failure | {counts[('failure', 1)]} | {counts[('failure', 2)]} | "
        f"{counts[('failure', 3)]} | {len(failures)} |",
        "",
        "Detected scoop counts on successful episodes:",
        "",
        "| Target | Episodes | Detected events |",
        "| ---: | ---: | ---: |",
    ]
    for target in TARGETS:
        target_successes = [episode for episode in successes if episode["target"] == target]
        lines.append(
            f"| {target} | {len(target_successes)} | "
            f"{event_counts[target]} (expected {target * len(target_successes)}) |"
        )

    lines.extend(
        [
            "",
            "## Successful-episode duration",
            "",
            "| Target | Minimum (s) | Median (s) | 90th percentile (s) | Maximum (s) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in TARGETS:
        values = [length / 10.0 for length in success_lengths[target]]
        lines.append(
            f"| {target} | {min(values):.1f} | {percentile(values, 50):.2f} | "
            f"{percentile(values, 90):.2f} | {max(values):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Scoop timing",
            "",
            f"Detected event windows have median **{percentile(event_durations, 50) / 10:.2f} s**, "
            f"10th percentile **{percentile(event_durations, 10) / 10:.2f} s**, and "
            f"90th percentile **{percentile(event_durations, 90) / 10:.2f} s**. "
            f"Inter-scoop intervals have median **{percentile(inter_event, 50):.2f} s** "
            f"and range from **{min(inter_event):.2f}--{max(inter_event):.2f} s**.",
            "",
            "## Best options for each history length",
            "",
            "These rows maximize terminal event recall for each allowed history "
            "length; ties prefer stronger last-25-chunk coverage and then the "
            "shorter span.",
            "",
            "| H | S | Span (steps) | Span (s) | Terminal event recall | Terminal all-events | Last-25 chunks all-events | Critic tokens |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in best_by_length:
        lines.append(
            f"| {item['history_length']} | {item['hist_stride']} | {item['step_span']} | "
            f"{item['nominal_seconds']:.1f} | {item['terminal_event_recall']:.1%} | "
            f"{item['terminal_all_events_fraction']:.1%} | "
            f"{item['positive_chunk_all_events_fraction']:.1%} | "
            f"{item['critic_tokens']} |"
        )

    lines.extend(
        [
            "",
            "## Practical >=95% event-recall options",
            "",
            "| H | S | Span (steps) | Sampling interval (s) | Terminal event recall | Terminal all-events | Last-25 chunks all-events | Critic tokens |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    seen = set()
    for item in practical:
        key = (item["history_length"], item["hist_stride"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {item['history_length']} | {item['hist_stride']} | {item['step_span']} | "
            f"{item['sampling_seconds']:.1f} | {item['terminal_event_recall']:.1%} | "
            f"{item['terminal_all_events_fraction']:.1%} | "
            f"{item['positive_chunk_all_events_fraction']:.1%} | "
            f"{item['critic_tokens']} |"
        )
        if len(seen) >= 15:
            break

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The grid covers history lengths 1--25 and strides {', '.join(map(str, strides))}. "
            "Increasing stride extends the physical horizon without increasing "
            "the number of image tokens, but it also makes it easier to skip a "
            "short scoop event. The final recommendation should therefore use "
            "the shortest-span option with high event recall, then be verified "
            "against the measured GPU-memory limit.",
            "",
            "Token counts use the same convention as the original report: actor "
            "tokens are `H + 2` and critic tokens are `H + 27` for a 25-action "
            "chunk. This report is a coverage/temporal analysis, not a direct "
            "VRAM benchmark.",
            "",
            "## Reproducibility",
            "",
            "Generated by `scripts/analyze_candy_history_grid.py` from the raw "
            "DROID dataset. The full grid is stored in the companion JSON file.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/iris/u/ronpo/expo-ft-data/candy_scoop"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("analysis/candy_history_grid_h25.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experiment_analysis.md"),
    )
    parser.add_argument(
        "--strides",
        default=":".join(map(str, DEFAULT_STRIDES)),
        help="Colon-separated stride values to evaluate.",
    )
    args = parser.parse_args()
    strides = tuple(int(value) for value in args.strides.split(":"))
    if not strides or any(value < 1 for value in strides):
        raise ValueError("strides must contain positive integers")

    episodes = discover_episodes(args.dataset)
    for episode in episodes:
        episode["events"] = (
            detect_source_events(episode["positions"])
            if episode["positions"] is not None
            else []
        )
        del episode["positions"]
    successes = [episode for episode in episodes if episode["outcome"] == "success"]
    rows = compute_grid(successes, strides)
    payload = {
        "dataset": str(args.dataset),
        "history_length_max": 25,
        "strides": strides,
        "episodes": episodes,
        "grid": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.output_md.write_text(build_report(episodes, rows, strides))
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")
    print(f"episodes={len(episodes)} successes={len(successes)} grid_rows={len(rows)}")


if __name__ == "__main__":
    main()
