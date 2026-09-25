#!/usr/bin/env python3
"""Measure egg-task cue retention for MTQL history configurations.

The wrist-camera card pass is identified by the first positive-Y excursion of
the end effector.  Three nested Y thresholds are evaluated so the result does
not depend on one hand-picked boundary.  Gripper-close onset is used as a
conservative end of the salt/pepper choice interval: the route starts earlier,
so retaining the cue through grasp is stricter than retaining it only through
initial route selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np


CONTROL_HZ = 10.0
CUE_THRESHOLDS = (0.20, 0.25, 0.28)
DEFAULT_CANDIDATES = (
    (20, 7),
    (20, 8),
    (18, 10),
    (20, 9),
    (20, 10),
    (22, 8),
    (24, 7),
    (25, 7),
    (30, 7),
)


def _read_episode(path_text: str, dataset_text: str) -> dict:
    path = Path(path_text)
    dataset = Path(dataset_text)
    relative = path.relative_to(dataset)
    color = "black" if "card_black" in relative.parts else "white"
    outcome = "success" if "success" in relative.parts else "failure"
    with h5py.File(path, "r") as handle:
        state = np.concatenate(
            [
                np.asarray(
                    handle["saved_observation/cartesian_position"],
                    dtype=np.float32,
                ),
                np.asarray(
                    handle["saved_observation/gripper_position"],
                    dtype=np.float32,
                )[:, None],
            ],
            axis=-1,
        )
        actions = np.concatenate(
            [
                np.asarray(
                    handle["action/cartesian_velocity"], dtype=np.float32
                ),
                np.asarray(
                    handle["action/gripper_velocity"], dtype=np.float32
                )[:, None],
            ],
            axis=-1,
        )
    if len(state) != len(actions):
        raise ValueError(f"State/action length mismatch in {path}")
    return {
        "path": str(relative),
        "color": color,
        "outcome": outcome,
        "state": state,
        "actions": actions,
    }


def discover_episodes(dataset: Path, workers: int) -> list[dict]:
    paths = sorted(
        path for path in dataset.rglob("traj.hdf5") if path.parent.name.isdigit()
    )
    episodes = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_read_episode, str(path), str(dataset))
            for path in paths
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            episodes.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"read {index}/{len(futures)} episodes", flush=True)
    episodes.sort(key=lambda item: item["path"])
    return episodes


def cue_window(y_position: np.ndarray, threshold: float) -> tuple[int, int]:
    """Return a thresholded cue window within the first card-side excursion."""
    outer = np.flatnonzero(y_position > 0.15)
    if not len(outer):
        raise ValueError("Trajectory never entered the card-side workspace")
    excursion_start = int(outer[0])
    returned = np.flatnonzero(
        (np.arange(len(y_position)) > excursion_start) & (y_position < 0.10)
    )
    excursion_end = int(returned[0]) if len(returned) else len(y_position) - 1
    indices = np.flatnonzero(
        (np.arange(len(y_position)) >= excursion_start)
        & (np.arange(len(y_position)) <= excursion_end)
        & (y_position > threshold)
    )
    if not len(indices):
        raise ValueError(
            f"First card-side excursion never exceeded y={threshold:.2f}"
        )
    return int(indices[0]), int(indices[-1])


def history_indices(current: int, length: int, stride: int) -> np.ndarray:
    return np.maximum(
        current - np.arange(length, 0, -1, dtype=np.int64) * stride,
        0,
    )


def cue_hits(
    current: int,
    cue: tuple[int, int],
    length: int,
    stride: int,
) -> int:
    indices = history_indices(current, length, stride)
    return int(np.sum((indices >= cue[0]) & (indices <= cue[1])))


def prepare_episode(episode: dict) -> dict:
    y_position = episode["state"][:, 1]
    close_indices = np.flatnonzero(np.abs(episode["actions"][:, -1]) > 0.1)
    close = int(close_indices[0]) if len(close_indices) else None
    windows = {}
    for threshold in CUE_THRESHOLDS:
        try:
            windows[str(threshold)] = cue_window(y_position, threshold)
        except ValueError:
            # Some deliberately aborted failures turn away before reaching
            # the innermost geometric threshold. They are retained in dataset
            # counts and counterfactual-route analysis, but not cue timing.
            windows[str(threshold)] = None
    return {
        "path": episode["path"],
        "color": episode["color"],
        "outcome": episode["outcome"],
        "length": len(y_position),
        "terminal": len(y_position) - 1,
        "close": close,
        "grasp_xyz": (
            episode["state"][close, :3].tolist() if close is not None else None
        ),
        "cue_windows": windows,
    }


def candidate_metrics(
    successes: list[dict], length: int, stride: int
) -> dict:
    threshold_metrics = {}
    for threshold in CUE_THRESHOLDS:
        key = str(threshold)
        grasp_hits = []
        all_decision_steps = []
        decision_step_hits = []
        for episode in successes:
            if episode["close"] is None:
                continue
            cue = episode["cue_windows"][key]
            grasp_hits.append(
                cue_hits(episode["close"], cue, length, stride)
            )
            covered = [
                cue_hits(current, cue, length, stride) > 0
                for current in range(cue[1] + 1, episode["close"] + 1)
            ]
            all_decision_steps.append(bool(covered) and all(covered))
            decision_step_hits.extend(covered)
        hits = np.asarray(grasp_hits)
        threshold_metrics[key] = {
            "grasp_any_cue": float(np.mean(hits > 0)),
            "grasp_at_least_two_cues": float(np.mean(hits >= 2)),
            "grasp_mean_cue_samples": float(np.mean(hits)),
            "episodes_with_cue_at_every_post_card_step": float(
                np.mean(all_decision_steps)
            ),
            "post_card_step_coverage": float(np.mean(decision_step_hits)),
        }

    metric_names = next(iter(threshold_metrics.values())).keys()
    robust = {
        name: min(values[name] for values in threshold_metrics.values())
        for name in metric_names
    }
    return {
        "history_length": length,
        "hist_stride": stride,
        "span_steps": length * stride,
        "span_seconds": length * stride / CONTROL_HZ,
        "sampling_seconds": stride / CONTROL_HZ,
        "actor_tokens": length + 2,
        "critic_tokens": length + 27,
        "threshold_metrics": threshold_metrics,
        "robust_minimum": robust,
    }


def percentile(values: list[float]) -> dict:
    quantiles = (0, 5, 10, 25, 50, 75, 90, 95, 100)
    result = np.percentile(np.asarray(values, dtype=np.float64), quantiles)
    return {str(q): float(value) for q, value in zip(quantiles, result)}


def bootstrap_core_coverage(
    successes: list[dict],
    length: int,
    stride: int,
    *,
    samples: int,
    seed: int,
) -> dict:
    cue_key = str(max(CUE_THRESHOLDS))
    hits = np.asarray(
        [
            cue_hits(
                episode["close"],
                episode["cue_windows"][cue_key],
                length,
                stride,
            )
            for episode in successes
            if episode["close"] is not None
        ]
    )
    rng = np.random.default_rng(seed)
    any_samples = np.empty(samples)
    two_samples = np.empty(samples)
    for index in range(samples):
        selected = hits[rng.integers(0, len(hits), len(hits))]
        any_samples[index] = np.mean(selected > 0)
        two_samples[index] = np.mean(selected >= 2)
    return {
        "core_grasp_any_point": float(np.mean(hits > 0)),
        "core_grasp_any_95pct": np.percentile(
            any_samples, [2.5, 97.5]
        ).tolist(),
        "core_grasp_two_point": float(np.mean(hits >= 2)),
        "core_grasp_two_95pct": np.percentile(
            two_samples, [2.5, 97.5]
        ).tolist(),
    }


def parse_candidates(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        length, stride = (int(field) for field in item.split(":"))
        result.append((length, stride))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/iris/u/ronpo/expo-ft-data/egg_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/egg_history_metrics.json"),
    )
    parser.add_argument(
        "--candidates",
        default=",".join(f"{h}:{s}" for h, s in DEFAULT_CANDIDATES),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    raw_episodes = discover_episodes(args.dataset, args.workers)
    episodes = [prepare_episode(episode) for episode in raw_episodes]
    successes = [item for item in episodes if item["outcome"] == "success"]
    failures = [item for item in episodes if item["outcome"] == "failure"]
    candidates = parse_candidates(args.candidates)
    results = []
    for index, (length, stride) in enumerate(candidates):
        result = candidate_metrics(successes, length, stride)
        result["bootstrap"] = bootstrap_core_coverage(
            successes,
            length,
            stride,
            samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        results.append(result)

    cue_timing = {}
    for threshold in CUE_THRESHOLDS:
        key = str(threshold)
        cue_timing[key] = {
            "duration_seconds": percentile(
                [
                    (item["cue_windows"][key][1] - item["cue_windows"][key][0] + 1)
                    / CONTROL_HZ
                    for item in successes
                ]
            ),
            "cue_exit_to_grasp_seconds": percentile(
                [
                    (item["close"] - item["cue_windows"][key][1])
                    / CONTROL_HZ
                    for item in successes
                    if item["close"] is not None
                ]
            ),
        }

    grasped_failures = [item for item in failures if item["grasp_xyz"] is not None]
    success_grasp_x = {
        color: np.asarray(
            [
                item["grasp_xyz"][0]
                for item in successes
                if item["color"] == color
            ]
        )
        for color in ("black", "white")
    }
    route_midpoint = float(
        sum(np.median(values) for values in success_grasp_x.values()) / 2
    )
    wrong_route_failures = sum(
        (
            item["color"] == "black"
            and item["grasp_xyz"][0] >= route_midpoint
        )
        or (
            item["color"] == "white"
            and item["grasp_xyz"][0] < route_midpoint
        )
        for item in grasped_failures
    )

    counts = Counter((item["color"], item["outcome"]) for item in episodes)
    payload = {
        "dataset": str(args.dataset),
        "control_hz": CONTROL_HZ,
        "method": {
            "cue_thresholds_y": CUE_THRESHOLDS,
            "history_indices": "t-H*S,...,t-S with episode-start clamping",
            "decision_endpoint": "first abs(gripper_velocity) > 0.1",
        },
        "dataset_summary": {
            "episodes": len(episodes),
            "transitions": sum(item["length"] for item in episodes),
            "counts": {
                f"{color}_{outcome}": counts[(color, outcome)]
                for color in ("black", "white")
                for outcome in ("success", "failure")
            },
            "success_length_steps": percentile(
                [item["length"] for item in successes]
            ),
            "failure_length_steps": percentile(
                [item["length"] for item in failures]
            ),
        },
        "cue_timing": cue_timing,
        "counterfactual_failures": {
            "failures_reaching_grasp": len(grasped_failures),
            "wrong_route_at_grasp": wrong_route_failures,
            "success_grasp_x_median": {
                color: float(np.median(values))
                for color, values in success_grasp_x.items()
            },
            "route_midpoint_x": route_midpoint,
        },
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    for result in results:
        robust = result["robust_minimum"]
        print(
            f"H{result['history_length']}/S{result['hist_stride']}: "
            f"span={result['span_seconds']:.1f}s "
            f"grasp_any={robust['grasp_any_cue']:.1%} "
            f"grasp_two={robust['grasp_at_least_two_cues']:.1%} "
            f"mean_hits={robust['grasp_mean_cue_samples']:.2f}"
        )


if __name__ == "__main__":
    main()
