#!/usr/bin/env python3
"""Select history/stride settings for a reward-credit-assignment stress test.

This analysis is deliberately separate from architecture performance.  It
uses the raw DROID geometry labels to identify histories that preserve the
long, multi-scoop context on which a reward-conditioned critic could help.
It cannot prove that MTQL will outperform either baseline; that requires
training and held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_candy_history_grid import (  # noqa: E402
    detect_source_events,
    discover_episodes,
    event_is_seen,
    history_indices,
)


TARGETS = (1, 2, 3)
# Prioritize the transformer critic over the stronger of the two baselines.
# All four settings therefore stay in the long-history regime; the last two
# isolate stride at essentially the same maximum history length.
RL_CANDIDATES = ((18, 30), (22, 24), (23, 24), (23, 25))


def parse_candidates(value: str) -> tuple[tuple[int, int], ...]:
    """Parse candidates such as ``19:25,20:25,24:20``."""
    candidates = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError(
                f"Invalid candidate {item!r}; expected HISTORY:STRIDE."
            )
        length, stride = (int(field) for field in fields)
        if length < 1 or stride < 1:
            raise ValueError("History lengths and strides must be positive.")
        candidates.append((length, stride))
    if not candidates:
        raise ValueError("At least one history/stride candidate is required.")
    return tuple(candidates)


def analyze_config(successes: list[dict], length: int, stride: int) -> dict:
    by_target = {
        target: {"episodes": 0, "all_events": 0, "event_hits": 0, "events": 0}
        for target in TARGETS
    }
    for episode in successes:
        target_stats = by_target[episode["target"]]
        target_stats["episodes"] += 1
        history = history_indices(episode["length"] - 1, length, stride)
        covered = [
            event_is_seen(history, event) for event in episode["events"]
        ]
        target_stats["all_events"] += int(all(covered))
        target_stats["event_hits"] += sum(covered)
        target_stats["events"] += len(covered)

    def fraction(target: int, key: str) -> float:
        stats = by_target[target]
        denominator = stats["episodes"] if key == "all_events" else stats["events"]
        return stats[key] / denominator if denominator else 0.0

    multi_episodes = by_target[2]["episodes"] + by_target[3]["episodes"]
    multi_all = by_target[2]["all_events"] + by_target[3]["all_events"]
    total_hits = sum(stats["event_hits"] for stats in by_target.values())
    total_events = sum(stats["events"] for stats in by_target.values())
    # This is a transparent data-coverage proxy, not a prediction of model
    # ranking.  Target 3 receives the largest weight because it contains the
    # longest multi-event trajectories and is the strongest credit test.
    credit_proxy = (
        0.50 * fraction(3, "all_events")
        + 0.30 * (multi_all / multi_episodes if multi_episodes else 0.0)
        + 0.20 * (total_hits / total_events if total_events else 0.0)
    )
    return {
        "history_length": length,
        "hist_stride": stride,
        "span_steps": length * stride,
        "span_seconds": length * stride / 10.0,
        "sampling_seconds": stride / 10.0,
        "critic_tokens": length + 27,
        "target_terminal_all_events": {
            str(target): fraction(target, "all_events") for target in TARGETS
        },
        "target_terminal_event_recall": {
            str(target): fraction(target, "event_hits") for target in TARGETS
        },
        "multi_target_terminal_all_events": (
            multi_all / multi_episodes if multi_episodes else 0.0
        ),
        "terminal_event_recall": total_hits / total_events if total_events else 0.0,
        "rl_credit_coverage_proxy": credit_proxy,
    }


def _load_terminal_positions(episodes: list[dict], dataset: Path) -> list[np.ndarray]:
    """Load only Cartesian positions for terminal reward-separability analysis."""
    positions = []
    for episode in episodes:
        if episode.get("positions") is not None:
            positions.append(np.asarray(episode["positions"], dtype=np.float32))
            continue
        with h5py.File(dataset / episode["path"], "r") as handle:
            positions.append(
                np.asarray(
                    handle["saved_observation"]["cartesian_position"][:, :3],
                    dtype=np.float32,
                )
            )
    return positions


def _terminal_feature_sets(
    positions: list[np.ndarray],
    episodes: list[dict],
    length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ordered, order-invariant, and current-only geometry features."""
    ordered = []
    bagged = []
    current_only = []
    for position, episode in zip(positions, episodes):
        terminal = len(position) - 1
        indices = history_indices(terminal, length, stride)
        current = position[terminal, :3]
        relative_history = position[indices, :3] - current
        ordered.append(np.concatenate([current, relative_history.reshape(-1)]))
        bagged.append(
            np.concatenate(
                [
                    current,
                    relative_history.mean(axis=0),
                    relative_history.std(axis=0),
                    relative_history.min(axis=0),
                    relative_history.max(axis=0),
                ]
            )
        )
        current_only.append(current)
    del episode
    return (
        np.asarray(ordered, dtype=np.float32),
        np.asarray(bagged, dtype=np.float32),
        np.asarray(current_only, dtype=np.float32),
    )


def _balanced_knn_accuracy(
    features: np.ndarray,
    episodes: list[dict],
) -> float:
    """Leave-one-out same-target 1-NN balanced accuracy.

    This is a deliberately simple geometry proxy. It asks whether terminal
    success/failure is easier to separate from ordered history than from a
    permutation-invariant summary or the current position alone.
    """
    labels = np.asarray([episode["outcome"] == "success" for episode in episodes])
    targets = np.asarray([episode["target"] for episode in episodes])
    predictions = np.zeros_like(labels)
    for target in TARGETS:
        indices = np.flatnonzero(targets == target)
        values = features[indices]
        scale = values.std(axis=0) + 1e-6
        normalized = values / scale
        distances = np.sum(
            (normalized[:, None, :] - normalized[None, :, :]) ** 2,
            axis=-1,
        )
        np.fill_diagonal(distances, np.inf)
        nearest = np.argmin(distances, axis=1)
        predictions[indices] = labels[indices[nearest]]

    recalls = []
    for label in (False, True):
        selected = labels == label
        if np.any(selected):
            recalls.append(float(np.mean(predictions[selected] == label)))
    return float(np.mean(recalls)) if recalls else 0.0


def analyze_transformer_opportunity(
    episodes: list[dict],
    terminal_positions: list[np.ndarray],
    length: int,
    stride: int,
) -> dict:
    """Estimate ordered-history reward information beyond current state.

    The order gain is not a claim about implementation-level transformer vs
    MLP capacity. It is a pre-training diagnostic for whether temporal order
    carries terminal reward information that a history-aware critic can use.
    """
    ordered, bagged, current_only = _terminal_feature_sets(
        terminal_positions, episodes, length, stride
    )
    ordered_accuracy = _balanced_knn_accuracy(ordered, episodes)
    bagged_accuracy = _balanced_knn_accuracy(bagged, episodes)
    current_accuracy = _balanced_knn_accuracy(current_only, episodes)
    return {
        "reward_balanced_accuracy_ordered": ordered_accuracy,
        "reward_balanced_accuracy_bagged": bagged_accuracy,
        "reward_balanced_accuracy_current": current_accuracy,
        "ordered_history_gain_over_bagged": ordered_accuracy - bagged_accuracy,
        "ordered_history_gain_over_current": ordered_accuracy - current_accuracy,
    }


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
        default=Path("analysis/candy_rl_history_leverage.json"),
    )
    parser.add_argument(
        "--candidates",
        default=",".join(f"{length}:{stride}" for length, stride in RL_CANDIDATES),
        help="Comma-separated HISTORY:STRIDE candidates to score.",
    )
    args = parser.parse_args()
    candidates = parse_candidates(args.candidates)

    episodes = discover_episodes(args.dataset)
    terminal_positions = _load_terminal_positions(episodes, args.dataset)
    opportunity_episodes = [
        {
            "path": episode["path"],
            "outcome": episode["outcome"],
            "target": episode["target"],
        }
        for episode in episodes
    ]
    for episode in episodes:
        episode["events"] = (
            detect_source_events(episode["positions"])
            if episode["positions"] is not None
            else []
        )
        episode.pop("positions", None)
    successes = [episode for episode in episodes if episode["outcome"] == "success"]
    results = [analyze_config(successes, *config) for config in candidates]
    for result in results:
        result.update(
            analyze_transformer_opportunity(
                opportunity_episodes,
                terminal_positions,
                result["history_length"],
                result["hist_stride"],
            )
        )

    payload = {
        "dataset": str(args.dataset),
        "criterion": {
            "description": (
                "Target-3 and multi-target terminal history coverage, with "
                "target-3 weighted most heavily. This is a data-coverage "
                "proxy for reward credit assignment, not a model-performance "
                "prediction. Ordered-history diagnostics compare a simple "
                "reward-separability proxy against bagged and current-only "
                "geometry features."
            ),
            "candidates": [list(config) for config in candidates],
        },
        "episodes": {
            "total": len(episodes),
            "successes": len(successes),
            "failures": len(episodes) - len(successes),
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output_json}")
    print(
        f"episodes={len(episodes)} successes={len(successes)} "
        f"candidates={len(results)}"
    )
    for result in results:
        target3 = result["target_terminal_all_events"]["3"]
        print(
            f"H{result['history_length']}/S{result['hist_stride']}: "
            f"target3_all={target3:.1%} "
            f"multi_all={result['multi_target_terminal_all_events']:.1%} "
            f"event_recall={result['terminal_event_recall']:.1%}"
        )


if __name__ == "__main__":
    main()
