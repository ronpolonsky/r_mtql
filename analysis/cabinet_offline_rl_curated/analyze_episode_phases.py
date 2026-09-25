#!/usr/bin/env python3
"""Inventory cabinet trajectories using proprio-defined drawer phases.

The motion-planning policy returns close to its initial arm configuration
between drawer operations.  We use those sustained returns to segment each
episode, then cluster the non-home joint configurations into four drawer
identities.  The output is a candidate-selection aid; final selections must
still be checked visually.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "analysis/cabinet_25_mix_analysis/train_episode_inventory.csv"
OUTPUT = Path(__file__).resolve().parent


def sustained_home_runs(q: np.ndarray, threshold: float = 0.15, min_run: int = 5):
    distance = np.linalg.norm(q - q[0], axis=1)
    near = distance < threshold
    starts = np.flatnonzero(near & ~np.r_[False, near[:-1]])
    ends = np.flatnonzero(near & ~np.r_[near[1:], False])
    return distance, [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_run]


def episode_phases(q: np.ndarray):
    distance, home_runs = sustained_home_runs(q)
    starts = [a for a, _ in home_runs]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    boundaries = starts + [len(q)]
    phases = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 30:
            continue
        local_dist = distance[start:end]
        if local_dist.max(initial=0.0) < 0.5:
            continue
        peak = start + int(np.argmax(local_dist))
        active = np.flatnonzero(local_dist >= max(0.5, float(np.percentile(local_dist, 60)))) + start
        if not len(active):
            active = np.array([peak])
        # Joint configurations at the furthest point and across the active
        # part of the excursion distinguish the four physical drawers.
        feature = np.concatenate([
            q[peak],
            q[active].mean(axis=0),
            q[active].std(axis=0),
        ])
        phases.append({
            "start": start,
            "end": end,
            "length": end - start,
            "peak": peak,
            "max_home_distance": float(local_dist.max()),
            "feature": feature,
        })
    return phases, home_runs


def kmeans(x: np.ndarray, k: int = 4, attempts: int = 40, iterations: int = 200):
    scale = np.std(x, axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - np.mean(x, axis=0)) / scale
    rng = np.random.default_rng(20260901)
    best = None
    for _ in range(attempts):
        centers = z[rng.choice(len(z), k, replace=False)].copy()
        labels = np.zeros(len(z), dtype=np.int64)
        for _ in range(iterations):
            distances = ((z[:, None] - centers[None]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                labels = new_labels
                break
            labels = new_labels
            new_centers = []
            for cluster in range(k):
                members = z[labels == cluster]
                new_centers.append(members.mean(axis=0) if len(members) else z[rng.integers(len(z))])
            centers = np.stack(new_centers)
        inertia = float(((z - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy(), centers.copy(), scale.copy())
    return best


def collapse(sequence):
    out = []
    for item in sequence:
        if not out or item != out[-1]:
            out.append(item)
    return out


def main():
    rows = list(csv.DictReader(INVENTORY.open()))
    episodes = []
    all_features = []
    phase_owners = []
    for episode_index, row in enumerate(rows):
        with np.load(row["path"], allow_pickle=False) as data:
            q = data["obs_proprio"][:, :7].astype(np.float64)
        phases, home_runs = episode_phases(q)
        episodes.append({
            "row": row,
            "phases": phases,
            "home_runs": home_runs,
        })
        for phase_index, phase in enumerate(phases):
            all_features.append(phase["feature"])
            phase_owners.append((episode_index, phase_index))

    features = np.stack(all_features)
    inertia, labels, centers, scale = kmeans(features)
    for label, (episode_index, phase_index) in zip(labels, phase_owners):
        episodes[episode_index]["phases"][phase_index]["cluster"] = int(label)

    summaries = []
    for episode in episodes:
        row = episode["row"]
        sequence = [p["cluster"] for p in episode["phases"]]
        compressed = collapse(sequence)
        revisits = len(compressed) - len(set(compressed))
        run_lengths = []
        for cluster in compressed:
            # Count the corresponding contiguous run in the original sequence.
            pass
        cursor = 0
        while cursor < len(sequence):
            end = cursor + 1
            while end < len(sequence) and sequence[end] == sequence[cursor]:
                end += 1
            run_lengths.append(end - cursor)
            cursor = end
        unique = len(set(sequence))
        expected_success_phases = max(0, 2 * unique - 1)
        excess_phases = max(0, len(sequence) - expected_success_phases)
        summaries.append({
            "order": int(row["order"]),
            "episode": row["episode"],
            "path": row["path"],
            "success": row["success"] == "True",
            "length": int(row["length"]),
            "seed": int(row["seed"]),
            "noise_scale": float(row["noise_scale"]),
            "phase_count": len(sequence),
            "drawer_sequence": sequence,
            "compressed_drawer_sequence": compressed,
            "run_lengths": run_lengths,
            "unique_drawers": unique,
            "revisit_count": revisits,
            "excess_success_phases": excess_phases,
            "phases": [
                {k: v for k, v in phase.items() if k != "feature"}
                for phase in episode["phases"]
            ],
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": {
            "home_threshold": 0.15,
            "minimum_home_run": 5,
            "minimum_phase_length": 30,
            "minimum_phase_excursion": 0.5,
            "kmeans_inertia": inertia,
            "note": "Cluster numbers are arbitrary drawer identities pending frame inspection.",
        },
        "episodes": summaries,
    }
    (OUTPUT / "episode_phase_inventory.json").write_text(json.dumps(payload, indent=2) + "\n")

    fields = [
        "order", "episode", "path", "success", "length", "seed", "noise_scale",
        "phase_count", "drawer_sequence", "compressed_drawer_sequence", "run_lengths",
        "unique_drawers", "revisit_count", "excess_success_phases",
    ]
    with (OUTPUT / "episode_phase_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({k: summary[k] for k in fields})

    for success in (True, False):
        candidates = [s for s in summaries if s["success"] == success]
        candidates.sort(
            key=lambda s: (s["revisit_count"], s["excess_success_phases"], s["length"]),
            reverse=True,
        )
        print("SUCCESS" if success else "FAILURE")
        for item in candidates[:30]:
            print(
                f"{item['episode']} len={item['length']:4d} phases={item['phase_count']:2d} "
                f"seq={item['drawer_sequence']} compressed={item['compressed_drawer_sequence']} "
                f"revisits={item['revisit_count']} excess={item['excess_success_phases']}"
            )


if __name__ == "__main__":
    main()
