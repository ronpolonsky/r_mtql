#!/usr/bin/env python3
"""Analyze whether egg_v2 is a useful visual-memory benchmark.

The task cue is the demonstrator's finger in the first exterior-camera
frames.  This script deliberately does not reuse ``analyze_egg_history.py``:
that older utility detects a robot-Y excursion from a different egg data
collection, not the finger cue in egg_v2.

The analysis is read-only with respect to the dataset.  It measures the cue
duration, robot/grasp timing, target-route failures, cue retention under the
production history sampler, and simple target-label leakage after the hand
has left the scene.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONTROL_HZ = 10.0
SKIN_ROI_HEIGHT = 145
SKIN_ROI_WIDTH = 235
SKIN_LOW_COUNT = 3000
SKIN_SUSTAINED_FRAMES = 3
MOVEMENT_METERS = 0.01
MOVEMENT_SUSTAINED_FRAMES = 3
DEFAULT_LENGTHS = (8, 10, 12, 14, 16, 18, 19)
DEFAULT_STRIDES = (2, 3, 4, 5, 6, 7, 8, 10)


def _percentiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = (0, 5, 10, 25, 50, 75, 90, 95, 100)
    result = np.percentile(values, quantiles)
    return {str(q): float(value) for q, value in zip(quantiles, result)}


def _first_sustained(mask: np.ndarray, run: int, start: int = 0) -> int | None:
    mask = np.asarray(mask, dtype=bool)
    if run < 1:
        raise ValueError("run must be positive")
    if len(mask) < run:
        return None
    hits = np.convolve(mask.astype(np.int8), np.ones(run, dtype=np.int8), "valid")
    indices = np.flatnonzero((hits == run) & (np.arange(len(hits)) >= start))
    return int(indices[0]) if len(indices) else None


def _skin_counts(images: np.ndarray) -> np.ndarray:
    """Count skin-colored pixels in the upper-left side-camera work area."""
    rgb = images[:, :SKIN_ROI_HEIGHT, :SKIN_ROI_WIDTH].astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    skin = (
        (y > 35.0)
        & (cr > 132.0)
        & (cr < 180.0)
        & (cb > 72.0)
        & (cb < 132.0)
    )
    return skin.sum(axis=(1, 2)).astype(np.int32)


def _grid_feature(image: np.ndarray, rows: int = 6, cols: int = 10) -> np.ndarray:
    """Return low-capacity spatial RGB means without importing a vision model."""
    height, width = image.shape[:2]
    row_edges = np.linspace(0, height, rows + 1, dtype=np.int32)
    col_edges = np.linspace(0, width, cols + 1, dtype=np.int32)
    cells = []
    for row in range(rows):
        for col in range(cols):
            patch = image[
                row_edges[row] : row_edges[row + 1],
                col_edges[col] : col_edges[col + 1],
            ]
            cells.append(patch.mean(axis=(0, 1)))
    return np.asarray(cells, dtype=np.float32).reshape(-1) / 255.0


def _cue_end(counts: np.ndarray) -> int:
    """Return the last cue frame before a sustained low-skin interval."""
    first_low = _first_sustained(
        counts < SKIN_LOW_COUNT,
        SKIN_SUSTAINED_FRAMES,
        start=2,
    )
    if first_low is None:
        return min(len(counts) - 1, 30)
    return max(0, first_low - 1)


def _episode_info(path_text: str, dataset_text: str) -> dict:
    path = Path(path_text)
    dataset = Path(dataset_text)
    relative = path.relative_to(dataset)
    parts = relative.parts
    target = "pepper" if parts[0] == "card_black" else "salt"
    outcome = parts[1]
    episode_number = int(parts[2])

    with h5py.File(path, "r") as handle:
        observations = handle["saved_observation"]
        side = np.asarray(observations["exterior_image_1_left"], dtype=np.uint8)
        wrist = np.asarray(observations["wrist_image_left"], dtype=np.uint8)
        cartesian = np.asarray(observations["cartesian_position"], dtype=np.float32)
        gripper_position = np.asarray(
            observations["gripper_position"], dtype=np.float32
        )
        actions = handle["action"]
        cartesian_velocity = np.asarray(
            actions["cartesian_velocity"], dtype=np.float32
        )
        gripper_velocity = np.asarray(
            actions["gripper_velocity"], dtype=np.float32
        )

    lengths = {
        len(side),
        len(wrist),
        len(cartesian),
        len(gripper_position),
        len(cartesian_velocity),
        len(gripper_velocity),
    }
    if len(lengths) != 1:
        raise ValueError(f"Mismatched arrays in {path}: {sorted(lengths)}")

    counts = _skin_counts(side)
    cue_end = _cue_end(counts)
    initial_xyz = np.median(cartesian[: min(3, len(cartesian)), :3], axis=0)
    displacement = np.linalg.norm(cartesian[:, :3] - initial_xyz, axis=1)
    movement = _first_sustained(
        displacement > MOVEMENT_METERS,
        MOVEMENT_SUSTAINED_FRAMES,
    )
    if movement is None:
        movement = len(cartesian) - 1

    close_indices = np.flatnonzero(np.abs(gripper_velocity) > 0.1)
    close = int(close_indices[0]) if len(close_indices) else None

    # The neutral frame is after the hand leaves.  Prefer one before route
    # motion begins, but never move backward into the detected cue interval.
    neutral = min(cue_end + 3, len(side) - 1)
    if movement > cue_end + 1:
        neutral = min(neutral, movement - 1)
    neutral = max(neutral, cue_end + 1)

    feature_indices = {
        "cue_side": 0,
        "cue_wrist": 0,
        "neutral_side": neutral,
        "neutral_wrist": neutral,
    }
    features = {
        key: _grid_feature(side[index] if key.endswith("side") else wrist[index])
        for key, index in feature_indices.items()
    }
    # Regions that exclude the pointing hand test for collection-session
    # shortcuts such as camera exposure, lighting, or background drift.
    features["initial_side_bottom"] = _grid_feature(
        side[0, -30:], rows=2, cols=4
    )
    features["initial_wrist_bottom"] = _grid_feature(
        wrist[0, -45:], rows=2, cols=4
    )

    state = np.concatenate([cartesian, gripper_position[:, None]], axis=1)
    return {
        "path": str(relative),
        "target": target,
        "outcome": outcome,
        "episode_number": episode_number,
        "length": len(side),
        "skin_counts": counts.tolist(),
        "cue_end": cue_end,
        "movement": movement,
        "close": close,
        "neutral": neutral,
        "state": state.tolist(),
        "cartesian_velocity": cartesian_velocity.tolist(),
        "features": {key: value.tolist() for key, value in features.items()},
    }


def discover(dataset: Path, workers: int) -> list[dict]:
    paths = sorted(
        path
        for path in dataset.rglob("traj.hdf5")
        if path.parent.name.isdigit()
        and path.relative_to(dataset).parts[0] in {"card_black", "card_white"}
    )
    episodes = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_episode_info, str(path), str(dataset))
            for path in paths
        ]
        for number, future in enumerate(as_completed(futures), 1):
            episodes.append(future.result())
            if number % 20 == 0 or number == len(futures):
                print(f"Analyzed {number}/{len(futures)} episodes", flush=True)
    episodes.sort(key=lambda item: item["path"])
    return episodes


def history_indices(current: int, length: int, stride: int) -> np.ndarray:
    return np.maximum(
        current - np.arange(length, 0, -1, dtype=np.int64) * stride,
        0,
    )


def candidate_metrics(episodes: list[dict], length: int, stride: int) -> dict:
    usable = [episode for episode in episodes if episode["close"] is not None]
    cue_tokens = []
    cue_unique = []
    padding = []
    all_steps = []
    step_coverage = []
    for episode in usable:
        indices = history_indices(episode["close"], length, stride)
        hits = indices <= episode["cue_end"]
        cue_tokens.append(int(hits.sum()))
        cue_unique.append(len(np.unique(indices[hits])))
        padding.append(float(np.mean(indices == 0)))

        covered = []
        for current in range(episode["cue_end"] + 1, episode["close"] + 1):
            decision_indices = history_indices(current, length, stride)
            covered.append(bool(np.any(decision_indices <= episode["cue_end"])))
        all_steps.append(all(covered))
        step_coverage.extend(covered)

    cue_tokens = np.asarray(cue_tokens)
    cue_unique = np.asarray(cue_unique)
    return {
        "history_length": length,
        "hist_stride": stride,
        "span_seconds": length * stride / CONTROL_HZ,
        "sample_interval_seconds": stride / CONTROL_HZ,
        "actor_tokens_including_cls": length + 2,
        "critic_tokens_including_cls": length + 27,
        "history_image_mib_batch16_current_and_next": (
            2 * 16 * length * 2 * 224 * 224 * 3 / 1024**2
        ),
        "grasp_any_cue": float(np.mean(cue_tokens > 0)),
        "grasp_two_unique_cue_frames": float(np.mean(cue_unique >= 2)),
        "grasp_mean_cue_tokens": float(np.mean(cue_tokens)),
        "grasp_mean_unique_cue_frames": float(np.mean(cue_unique)),
        "grasp_mean_start_padding": float(np.mean(padding)),
        "episodes_covered_every_post_cue_step": float(np.mean(all_steps)),
        "post_cue_step_coverage": float(np.mean(step_coverage)),
    }


def _cross_validated_classifier(
    episodes: list[dict], feature_key: str
) -> dict[str, float]:
    # Successes are exactly target-balanced. Pair black/white episode number
    # in the same fold so each held-out set represents both targets.
    selected = [episode for episode in episodes if episode["outcome"] == "success"]
    x = np.asarray([episode["features"][feature_key] for episode in selected])
    y = np.asarray([episode["target"] == "salt" for episode in selected], dtype=int)
    groups = np.asarray([episode["episode_number"] for episode in selected])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    predictions = np.empty_like(y)
    for train, test in splitter.split(x, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=5000, random_state=0),
        )
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "n": int(len(y)),
    }


def _outcome_shortcut_classifiers(episodes: list[dict]) -> dict[str, dict]:
    output = {}
    for target in ("pepper", "salt"):
        selected = [episode for episode in episodes if episode["target"] == target]
        y = np.asarray(
            [episode["outcome"] == "success" for episode in selected], dtype=int
        )
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        output[target] = {}
        for feature_key in ("initial_side_bottom", "initial_wrist_bottom"):
            x = np.asarray(
                [episode["features"][feature_key] for episode in selected]
            )
            predictions = np.empty_like(y)
            for train, test in splitter.split(x, y):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.03,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=0,
                    ),
                )
                model.fit(x[train], y[train])
                predictions[test] = model.predict(x[test])
            output[target][feature_key] = {
                "accuracy": float(accuracy_score(y, predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y, predictions)
                ),
                "n": int(len(y)),
            }
    return output


def _route_analysis(episodes: list[dict]) -> dict:
    success = [
        episode
        for episode in episodes
        if episode["outcome"] == "success" and episode["close"] is not None
    ]
    close_xyz = {
        target: np.asarray(
            [
                episode["state"][episode["close"]][:3]
                for episode in success
                if episode["target"] == target
            ],
            dtype=np.float32,
        )
        for target in ("pepper", "salt")
    }
    centroids = {target: values.mean(axis=0) for target, values in close_xyz.items()}

    def predicted_route(episode: dict) -> str | None:
        if episode["close"] is None:
            return None
        xyz = np.asarray(episode["state"][episode["close"]][:3])
        return min(centroids, key=lambda target: np.linalg.norm(xyz - centroids[target]))

    success_correct = [predicted_route(episode) == episode["target"] for episode in success]
    failure = [episode for episode in episodes if episode["outcome"] == "failure"]
    failure_routes = [predicted_route(episode) for episode in failure]
    wrong = [
        predicted_route(episode) != episode["target"]
        for episode in failure
        if episode["close"] is not None
    ]
    wrong_episodes = [
        episode
        for episode in failure
        if episode["close"] is not None
        and predicted_route(episode) != episode["target"]
    ]
    same_route_episodes = [
        episode
        for episode in failure
        if episode["close"] is not None
        and predicted_route(episode) == episode["target"]
    ]
    return {
        "grasp_centroid_xyz": {
            target: value.tolist() for target, value in centroids.items()
        },
        "success_route_separation_accuracy": float(np.mean(success_correct)),
        "failures": len(failure),
        "failures_reaching_gripper_close": int(
            sum(episode["close"] is not None for episode in failure)
        ),
        "failure_predicted_routes": {
            (route if route is not None else "no_gripper_close"): int(count)
            for route, count in Counter(failure_routes).items()
        },
        "wrong_target_route_after_close": int(sum(wrong)),
        "wrong_target_route_fraction_after_close": float(np.mean(wrong)),
        "wrong_target_route_by_intended_target": dict(
            Counter(episode["target"] for episode in wrong_episodes)
        ),
        "wrong_target_route_transitions": int(
            sum(episode["length"] for episode in wrong_episodes)
        ),
        "same_target_route_failure_transitions": int(
            sum(episode["length"] for episode in same_route_episodes)
        ),
    }


def _state_target_accuracy(episodes: list[dict]) -> dict[str, dict]:
    selected = [
        episode
        for episode in episodes
        if episode["outcome"] == "success"
    ]
    output = {}
    for offset in (0, 2, 5, 10, 15, 20, 30, 40):
        usable = [
            episode
            for episode in selected
            if episode["movement"] + offset < episode["length"]
        ]
        x = np.asarray(
            [episode["state"][episode["movement"] + offset] for episode in usable]
        )
        y = np.asarray([episode["target"] == "salt" for episode in usable], dtype=int)
        groups = np.asarray([episode["episode_number"] for episode in usable])
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        predictions = np.empty_like(y)
        for train, test in splitter.split(x, y, groups):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=5000, random_state=0),
            )
            model.fit(x[train], y[train])
            predictions[test] = model.predict(x[test])
        output[str(offset)] = {
            "seconds_after_movement": offset / CONTROL_HZ,
            "accuracy": float(accuracy_score(y, predictions)),
            "n": int(len(y)),
        }
    return output


def _absolute_state_target_accuracy(episodes: list[dict]) -> dict[str, dict]:
    selected = [episode for episode in episodes if episode["outcome"] == "success"]
    output = {}
    for step in (0, 5, 10, 15, 20, 25, 30, 40):
        usable = [episode for episode in selected if step < episode["length"]]
        x = np.asarray([episode["state"][step] for episode in usable])
        y = np.asarray([episode["target"] == "salt" for episode in usable], dtype=int)
        groups = np.asarray([episode["episode_number"] for episode in usable])
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        predictions = np.empty_like(y)
        for train, test in splitter.split(x, y, groups):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=5000, random_state=0),
            )
            model.fit(x[train], y[train])
            predictions[test] = model.predict(x[test])
        output[str(step)] = {
            "seconds_from_start": step / CONTROL_HZ,
            "accuracy": float(accuracy_score(y, predictions)),
            "n": int(len(y)),
        }
    return output


def summarize(episodes: list[dict], lengths, strides) -> dict:
    successes = [episode for episode in episodes if episode["outcome"] == "success"]
    failures = [episode for episode in episodes if episode["outcome"] == "failure"]
    close_episodes = [episode for episode in episodes if episode["close"] is not None]
    timing = {
        "episode_length_steps": _percentiles([episode["length"] for episode in episodes]),
        "cue_end_steps": _percentiles([episode["cue_end"] for episode in episodes]),
        "movement_onset_steps": _percentiles([episode["movement"] for episode in episodes]),
        "cue_to_movement_steps": _percentiles(
            [episode["movement"] - episode["cue_end"] for episode in episodes]
        ),
        "cue_to_close_steps": _percentiles(
            [episode["close"] - episode["cue_end"] for episode in close_episodes]
        ),
        "start_to_close_steps": _percentiles(
            [episode["close"] for episode in close_episodes]
        ),
    }
    candidates = [
        candidate_metrics(successes, length, stride)
        for length in lengths
        for stride in strides
    ]
    return {
        "dataset": {
            "episodes": len(episodes),
            "successes": len(successes),
            "failures": len(failures),
            "transitions": int(sum(episode["length"] for episode in episodes)),
            "success_transitions": int(sum(episode["length"] for episode in successes)),
            "failure_transitions": int(sum(episode["length"] for episode in failures)),
            "by_target_outcome": {
                f"{target}_{outcome}": int(count)
                for (target, outcome), count in Counter(
                    (episode["target"], episode["outcome"])
                    for episode in episodes
                ).items()
            },
        },
        "detector": {
            "skin_low_count": SKIN_LOW_COUNT,
            "skin_sustained_frames": SKIN_SUSTAINED_FRAMES,
            "initial_skin_counts": _percentiles(
                [episode["skin_counts"][0] for episode in episodes]
            ),
            "skin_count_at_neutral": _percentiles(
                [episode["skin_counts"][episode["neutral"]] for episode in episodes]
            ),
        },
        "timing": timing,
        "route_analysis": _route_analysis(episodes),
        "visual_target_classification": {
            key: _cross_validated_classifier(episodes, key)
            for key in (
                "cue_side",
                "cue_wrist",
                "neutral_side",
                "neutral_wrist",
                "initial_side_bottom",
                "initial_wrist_bottom",
            )
        },
        "initial_image_outcome_shortcuts": _outcome_shortcut_classifiers(episodes),
        "proprio_target_classification_from_start": (
            _absolute_state_target_accuracy(episodes)
        ),
        "proprio_target_classification_after_movement": _state_target_accuracy(episodes),
        "history_candidates": candidates,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/iris/u/ronpo/expo-ft-data/egg_v2"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    episodes = discover(args.dataset, args.workers)
    report = summarize(episodes, DEFAULT_LENGTHS, DEFAULT_STRIDES)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.output}", flush=True)

    compact = {key: value for key, value in report.items() if key != "episodes"}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
