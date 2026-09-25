#!/usr/bin/env python3
"""Quantify why the hard-BC selection can provide an offline-RL advantage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INVENTORY_PATH = ROOT / "analysis/cabinet_offline_rl_curated/episode_phase_inventory.json"
SELECTION_PATH = HERE / "selection.json"
CONTEXTS = ((16, 64), (20, 50), (24, 40), (28, 36), (32, 32))


def edit_distance(a: list[int], b: list[int]) -> int:
    table = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    table[:, 0] = np.arange(len(a) + 1)
    table[0, :] = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            table[i, j] = min(
                table[i - 1, j] + 1,
                table[i, j - 1] + 1,
                table[i - 1, j - 1] + (x != y),
            )
    return int(table[-1, -1])


def context_capture(items: list[dict], length: int, stride: int) -> dict:
    prior_all = []
    terminal_all = []
    previous_counts = []
    offsets = np.arange(-length, 0, dtype=np.int64) * stride
    for item in items:
        phases = item["phases"]
        for phase_index in range(1, len(phases)):
            samples = np.maximum(phases[phase_index]["start"] + offsets, 0)
            hits = [
                np.any((samples >= p["start"]) & (samples < p["end"]))
                for p in phases[:phase_index]
            ]
            prior_all.append(all(hits))
            p = phases[phase_index - 1]
            in_previous = samples[(samples >= p["start"]) & (samples < p["end"])]
            previous_counts.append(len(np.unique(in_previous)))
        samples = np.maximum((item["length"] - 1) + offsets, 0)
        terminal_all.append(all(
            np.any((samples >= p["start"]) & (samples < p["end"]))
            for p in phases
        ))
    return {
        "history_length": length,
        "history_stride": stride,
        "span": length * stride,
        "all_prior_phases_at_decisions": float(np.mean(prior_all)),
        "all_phases_at_terminal": float(np.mean(terminal_all)),
        "mean_unique_frames_from_previous_phase": float(np.mean(previous_counts)),
    }


def shaped_return(length: int, success: bool, gamma: float, step_cost: float = 0.001) -> float:
    if gamma == 1.0:
        prefix = -step_cost * (length - 1)
    else:
        prefix = -step_cost * (1 - gamma ** (length - 1)) / (1 - gamma)
    terminal = (1.0 if success else -1.0) * gamma ** (length - 1)
    return float(prefix + terminal)


def main() -> None:
    selection = json.loads(SELECTION_PATH.read_text())
    inventory = json.loads(INVENTORY_PATH.read_text())["episodes"]
    by_name = {item["episode"]: item for item in inventory}
    efficient = [by_name[name] for name in selection["efficient_successes"]]
    inefficient = [by_name[name] for name in selection["inefficient_successes"]]
    failures = [by_name[name] for name in selection["targeted_failures"]]
    successes = efficient + inefficient
    all_items = successes + failures

    success_transitions = sum(x["length"] for x in successes)
    inefficient_transitions = sum(x["length"] for x in inefficient)
    failure_transitions = sum(x["length"] for x in failures)

    success_sequences = [x["drawer_sequence"] for x in successes]
    matched_failures = []
    for failure in failures:
        distances = [edit_distance(failure["drawer_sequence"], seq) for seq in success_sequences]
        nearest = int(np.argmin(distances))
        matched_failures.append({
            "failure": failure["episode"],
            "sequence": failure["drawer_sequence"],
            "nearest_success": successes[nearest]["episode"],
            "nearest_success_sequence": success_sequences[nearest],
            "edit_distance": distances[nearest],
        })

    success_bigrams = {
        tuple(seq[i:i + 2]) for seq in success_sequences for i in range(len(seq) - 1)
    }
    failure_bigrams = [
        tuple(seq[i:i + 2])
        for item in failures for seq in [item["drawer_sequence"]]
        for i in range(len(seq) - 1)
    ]

    reward_ranges = {}
    for gamma in (0.999, 0.9995, 1.0):
        success_returns = [shaped_return(x["length"], True, gamma) for x in successes]
        failure_returns = [shaped_return(x["length"], False, gamma) for x in failures]
        reward_ranges[str(gamma)] = {
            "success_min": min(success_returns),
            "success_max": max(success_returns),
            "failure_min": min(failure_returns),
            "failure_max": max(failure_returns),
            "worst_success_minus_best_failure": min(success_returns) - max(failure_returns),
        }

    result = {
        "composition": {
            "efficient_success_episodes": len(efficient),
            "inefficient_success_episodes": len(inefficient),
            "failure_episodes": len(failures),
            "success_transitions": success_transitions,
            "inefficient_success_transitions": inefficient_transitions,
            "inefficient_share_of_bc_transitions": inefficient_transitions / success_transitions,
            "failure_transitions": failure_transitions,
            "failure_share_of_critic_transitions": failure_transitions / (success_transitions + failure_transitions),
        },
        "success_revisit_statistics": {
            "transition_weighted_mean_revisits": sum(x["length"] * x["revisit_count"] for x in successes) / success_transitions,
            "transition_weighted_mean_excess_phases": sum(x["length"] * x["excess_success_phases"] for x in successes) / success_transitions,
        },
        "failure_phase_overlap": {
            "failure_bigrams_seen_in_successes": int(sum(x in success_bigrams for x in failure_bigrams)),
            "failure_bigrams_total": len(failure_bigrams),
            "fraction": float(np.mean([x in success_bigrams for x in failure_bigrams])),
            "nearest_sequence_matches": matched_failures,
        },
        "contexts": [context_capture(all_items, *context) for context in CONTEXTS],
        "outcome_time_reward_returns": reward_ranges,
    }
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
