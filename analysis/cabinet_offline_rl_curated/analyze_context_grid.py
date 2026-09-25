#!/usr/bin/env python3
"""Score transformer history layouts against the curated proprio phases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SELECTION = json.loads((HERE / "proposed_selection.json").read_text())
INVENTORY = json.loads((HERE / "episode_phase_inventory.json").read_text())["episodes"]
BY_NAME = {item["episode"]: item for item in INVENTORY}

# Similar ~1,000-step horizons with different temporal resolution/token cost,
# plus the old shorter-context baseline.
CONTEXTS = (
    (12, 50),
    (16, 64),
    (20, 50),
    (24, 40),
    (28, 36),
    (32, 32),
)


def captures_phase(sample_times: np.ndarray, phase: dict) -> bool:
    # Inventory phases use half-open [start, end) intervals.
    return bool(np.any((sample_times >= phase["start"]) & (sample_times < phase["end"])))


def sampled_tokens_in_phase(sample_times: np.ndarray, phase: dict) -> int:
    in_phase = sample_times[
        (sample_times >= phase["start"]) & (sample_times < phase["end"])
    ]
    # Padding clamps many early history slots to t=0. Count unique physical
    # frames so padding duplicates are not mistaken for temporal resolution.
    return int(len(np.unique(in_phase)))


def score_context(items: list[dict], length: int, stride: int) -> dict:
    previous_hit = []
    all_previous_hit = []
    previous_fraction = []
    terminal_fraction = []
    terminal_all = []

    for item in items:
        phases = item["phases"]
        for phase_index in range(1, len(phases)):
            decision_t = phases[phase_index]["start"]
            offsets = np.arange(-length, 0, dtype=np.int64) * stride
            samples = np.maximum(decision_t + offsets, 0)
            previous = phases[:phase_index]
            hits = [captures_phase(samples, phase) for phase in previous]
            previous_hit.append(float(hits[-1]))
            all_previous_hit.append(float(all(hits)))
            previous_fraction.append(float(np.mean(hits)))

        terminal_t = item["length"] - 1
        samples = np.maximum(
            terminal_t + np.arange(-length, 0, dtype=np.int64) * stride, 0
        )
        hits = [captures_phase(samples, phase) for phase in phases]
        terminal_fraction.append(float(np.mean(hits)))
        terminal_all.append(float(all(hits)))

    return {
        "history_length": length,
        "history_stride": stride,
        "oldest_offset_steps": length * stride,
        "previous_phase_capture": float(np.mean(previous_hit)),
        "all_previous_phases_capture": float(np.mean(all_previous_hit)),
        "mean_previous_phases_capture": float(np.mean(previous_fraction)),
        "terminal_mean_phase_capture": float(np.mean(terminal_fraction)),
        "terminal_all_phases_capture": float(np.mean(terminal_all)),
    }


def quantiles(values: list[float]) -> dict:
    q = np.quantile(values, [0, .1, .25, .5, .75, .9, 1])
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"), map(float, q)))


def main() -> None:
    names = sum(SELECTION.values(), [])
    items = [BY_NAME[name] for name in names]
    successes = [BY_NAME[name] for name in SELECTION["efficient_successes"] + SELECTION["inefficient_successes"]]
    failures = [BY_NAME[name] for name in SELECTION["targeted_failures"]]

    durations = [phase["length"] for item in items for phase in item["phases"]]
    phase_starts = [
        b["start"] - a["start"]
        for item in items for a, b in zip(item["phases"], item["phases"][1:])
    ]
    result = {
        "episodes": len(items),
        "episode_length_all": quantiles([item["length"] for item in items]),
        "episode_length_success": quantiles([item["length"] for item in successes]),
        "episode_length_failure": quantiles([item["length"] for item in failures]),
        "phase_duration": quantiles(durations),
        "phase_start_spacing": quantiles(phase_starts),
        "contexts": [score_context(items, *context) for context in CONTEXTS],
        "discount_terminal_weight": {
            str(discount): {
                "at_500_steps": discount ** 500,
                "at_750_steps": discount ** 750,
                "at_1000_steps": discount ** 1000,
            }
            for discount in (.99, .995, .997, .999, .9995, .9999, 1.0)
        },
    }
    output = HERE / "context_grid_analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    categories = {
        name: category for category, names_in_category in SELECTION.items()
        for name in names_in_category
    }
    rollout_map = []
    for item in items:
        context_maps = {}
        for length, stride in ((20, 50), (24, 40), (28, 36)):
            previous_counts = []
            for phase_index in range(1, len(item["phases"])):
                decision_t = item["phases"][phase_index]["start"]
                samples = np.maximum(
                    decision_t + np.arange(-length, 0, dtype=np.int64) * stride,
                    0,
                )
                previous_counts.append(sampled_tokens_in_phase(
                    samples, item["phases"][phase_index - 1]
                ))
            terminal_t = item["length"] - 1
            terminal_samples = np.maximum(
                terminal_t + np.arange(-length, 0, dtype=np.int64) * stride,
                0,
            )
            terminal_counts = [
                sampled_tokens_in_phase(terminal_samples, phase)
                for phase in item["phases"]
            ]
            context_maps[f"h{length}_s{stride}"] = {
                "previous_phase_unique_frame_counts_at_next_phase_start": previous_counts,
                "terminal_unique_frame_counts_by_phase": terminal_counts,
                "terminal_phases_captured": int(np.count_nonzero(terminal_counts)),
            }
        rollout_map.append({
            "category": categories[item["episode"]],
            "episode": item["episode"],
            "success": item["success"],
            "length": item["length"],
            "phase_count": item["phase_count"],
            "phase_durations": [phase["length"] for phase in item["phases"]],
            "drawer_sequence_cluster_ids": item["drawer_sequence"],
            "unique_drawers": item["unique_drawers"],
            "revisit_count": item["revisit_count"],
            "context_capture": context_maps,
        })
    (HERE / "selected_rollout_context_map.json").write_text(
        json.dumps(rollout_map, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
