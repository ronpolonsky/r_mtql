#!/usr/bin/env python3
"""Bootstrap uncertainty for v2 history/stride coverage metrics.

This is deliberately a data-only diagnostic.  It resamples successful
episodes within each target number, preserving the v2 target balance, and
reports uncertainty for terminal event recall and complete target-3 event
coverage.  It does not claim to measure neural-agent performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_candy_history_grid import (
    detect_source_events,
    discover_episodes,
    event_is_seen,
    history_indices,
)


DEFAULT_CANDIDATES = ((19, 25), (20, 25), (21, 24), (23, 25), (25, 20))


def _metrics(episodes: list[dict], length: int, stride: int) -> dict:
    target_metrics = {}
    for target in (1, 2, 3):
        target_episodes = [
            episode for episode in episodes if episode["target"] == target
        ]
        all_events = []
        event_hits = 0
        event_total = 0
        for episode in target_episodes:
            history = history_indices(episode["length"] - 1, length, stride)
            covered = [
                event_is_seen(history, event) for event in episode["events"]
            ]
            all_events.append(bool(covered) and all(covered))
            event_hits += sum(covered)
            event_total += len(covered)
        target_metrics[target] = {
            "all_events": np.asarray(all_events, dtype=np.float64),
            "event_hits": event_hits,
            "event_total": event_total,
        }
    total_hits = sum(item["event_hits"] for item in target_metrics.values())
    total_events = sum(item["event_total"] for item in target_metrics.values())
    return {
        "target_metrics": target_metrics,
        "event_recall": total_hits / total_events if total_events else 0.0,
    }


def _bootstrap(
    episodes: list[dict],
    length: int,
    stride: int,
    *,
    samples: int,
    seed: int,
) -> dict:
    metrics = _metrics(episodes, length, stride)
    rng = np.random.default_rng(seed)
    by_target = {
        target: [episode for episode in episodes if episode["target"] == target]
        for target in (1, 2, 3)
    }
    target3_all = np.empty(samples, dtype=np.float64)
    event_recall = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        resampled = []
        for target, target_episodes in by_target.items():
            indices = rng.integers(0, len(target_episodes), len(target_episodes))
            resampled.extend(target_episodes[index] for index in indices)
        sample_metrics = _metrics(resampled, length, stride)
        target3_all[sample_index] = np.mean(
            sample_metrics["target_metrics"][3]["all_events"]
        )
        event_recall[sample_index] = sample_metrics["event_recall"]

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.percentile(values, [2.5, 50, 97.5])]

    return {
        "history_length": length,
        "hist_stride": stride,
        "critic_tokens": length + 27,
        "point": {
            "target3_all_events": float(
                np.mean(metrics["target_metrics"][3]["all_events"])
            ),
            "event_recall": float(metrics["event_recall"]),
        },
        "bootstrap_95pct": {
            "target3_all_events": interval(target3_all),
            "event_recall": interval(event_recall),
        },
        "bootstrap_samples": samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("analysis/candy_v2_history_bootstrap.json"),
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--candidates",
        default=",".join(f"{length}:{stride}" for length, stride in DEFAULT_CANDIDATES),
    )
    args = parser.parse_args()
    if args.samples < 100:
        raise ValueError("Use at least 100 bootstrap samples.")
    candidates = []
    for item in args.candidates.split(","):
        length, stride = (int(value) for value in item.split(":"))
        candidates.append((length, stride))

    episodes = discover_episodes(args.dataset)
    for episode in episodes:
        episode["events"] = (
            detect_source_events(episode["positions"])
            if episode["positions"] is not None
            else []
        )
        del episode["positions"]
    successes = [episode for episode in episodes if episode["outcome"] == "success"]
    results = [
        _bootstrap(
            successes,
            length,
            stride,
            samples=args.samples,
            seed=args.seed + index,
        )
        for index, (length, stride) in enumerate(candidates)
    ]
    payload = {
        "dataset": str(args.dataset),
        "episodes": len(episodes),
        "successes": len(successes),
        "candidates": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output_json}")
    for result in results:
        point = result["point"]
        interval = result["bootstrap_95pct"]
        print(
            f"H{result['history_length']}/S{result['hist_stride']}: "
            f"target3={point['target3_all_events']:.1%} "
            f"CI={interval['target3_all_events'][0]:.1%}--"
            f"{interval['target3_all_events'][2]:.1%}; "
            f"recall={point['event_recall']:.1%} "
            f"CI={interval['event_recall'][0]:.1%}--"
            f"{interval['event_recall'][2]:.1%}"
        )


if __name__ == "__main__":
    main()
