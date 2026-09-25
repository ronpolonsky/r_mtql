#!/usr/bin/env python3
"""Analyze and visualize the cabinet dataset used by the 25-success RL runs.

The reconstructed merged split was produced by shuffling ep_*.npz with
random.Random(42), reserving the last 25 episodes for validation, and merging
the remaining 225.  Recreating that ordering here makes all reported prefixes
match ``_select_episode_mix`` exactly.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = (
    ROOT / "cabinet-memory-sim" / "ManiSkill" /
    "cabinet_dataset_two_cams_more_rand"
)
EPISODE_DIR = DATASET_DIR / "episodes"
OUTPUT_DIR = Path(__file__).resolve().parent

HISTORY_SETTINGS = (
    (12, 7),
    (20, 7),
    (20, 10),
    (20, 25),
    (20, 40),
    (20, 50),
    (24, 40),
)


def train_paths() -> list[Path]:
    paths = sorted(EPISODE_DIR.glob("ep_*.npz"))
    random.Random(42).shuffle(paths)
    return paths[:-25]


def episode_info(path: Path, order: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        rewards = data["rewards"]
        actions = data["actions"]
        proprio = data["obs_proprio"]
        length = len(rewards)
        success = bool(np.any(rewards > 0))
        # Mean physical motion is useful for separating a short, static abort
        # from a trajectory that attempted a full drawer operation.
        joint_motion = (
            float(np.linalg.norm(np.diff(proprio[:, :7], axis=0), axis=1).sum())
            if length > 1 else 0.0
        )
        action_tv = (
            float(np.linalg.norm(np.diff(actions, axis=0), axis=1).sum())
            if length > 1 else 0.0
        )
        return {
            "order": order,
            "episode": path.stem,
            "path": str(path),
            "length": length,
            "success": success,
            "seed": int(data["meta_seed"]),
            "noise_scale": float(data["meta_noise_scale"]),
            "joint_motion": joint_motion,
            "action_total_variation": action_tv,
        }


def assign_length_band(length: int) -> str:
    # These empirical bands align with the strong modes in failed-episode
    # lengths. They describe coverage, not semantic ground-truth labels.
    if length < 180:
        return "short (<180)"
    if length < 330:
        return "medium-short (180-329)"
    if length < 600:
        return "medium (330-599)"
    return "long (>=600)"


def save_inventory(rows: list[dict]) -> None:
    fields = list(rows[0].keys()) + ["length_band", "success_rank", "failure_rank"]
    success_rank = 0
    failure_rank = 0
    with (OUTPUT_DIR / "train_episode_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["length_band"] = assign_length_band(row["length"])
            if row["success"]:
                out["success_rank"] = success_rank
                out["failure_rank"] = ""
                success_rank += 1
            else:
                out["success_rank"] = ""
                out["failure_rank"] = failure_rank
                failure_rank += 1
            writer.writerow(out)


def candidate_mixtures(rows: list[dict]) -> list[dict]:
    successes = [r for r in rows if r["success"]][:25]
    failures = [r for r in rows if not r["success"]]
    success_transitions = sum(r["length"] for r in successes)
    candidates = []
    for n in (0, 1, 3, 5, 8, 10, 12, 15, 20, 25, 40, 60, 82):
        selected = failures[:n]
        failure_transitions = sum(r["length"] for r in selected)
        bands = {}
        for row in selected:
            band = assign_length_band(row["length"])
            bands[band] = bands.get(band, 0) + 1
        candidates.append({
            "num_success": 25,
            "num_failure": n,
            "success_transitions": success_transitions,
            "failure_transitions": failure_transitions,
            "total_transitions": success_transitions + failure_transitions,
            "failure_episode_fraction": n / (25 + n) if n else 0.0,
            "failure_transition_fraction": (
                failure_transitions / (success_transitions + failure_transitions)
                if failure_transitions else 0.0
            ),
            "failure_length_bands": bands,
            "failure_episodes": [r["episode"] for r in selected],
        })
    return candidates


def image_change_curve(images: np.ndarray) -> np.ndarray:
    x = images.astype(np.float32)
    return np.mean(np.abs(x[1:] - x[:-1]), axis=(1, 2, 3))


def representative_times(path: Path, max_times: int = 24) -> list[int]:
    """Combine uniform coverage and high visual-change moments."""
    with np.load(path, allow_pickle=False) as data:
        length = len(data["rewards"])
        agent_change = image_change_curve(data["obs_image"])
        wrist_change = image_change_curve(data["obs_wrist_image"])
    uniform_n = min(16, max_times)
    uniform = np.linspace(0, length - 1, uniform_n, dtype=int)
    change = agent_change + wrist_change
    peak_n = max_times - uniform_n
    if peak_n:
        # Keep peaks separated so a single fast motion does not consume every slot.
        order = np.argsort(change)[::-1]
        peaks = []
        for idx in order:
            t = int(idx + 1)
            if all(abs(t - prior) >= 12 for prior in peaks):
                peaks.append(t)
            if len(peaks) == peak_n:
                break
    else:
        peaks = []
    return sorted(set(int(t) for t in uniform) | set(peaks))


def add_image(ax, image: np.ndarray, title: str = "") -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def trajectory_page(path: Path, info: dict):
    times = representative_times(path)
    cols = 8
    blocks = math.ceil(len(times) / cols)
    fig = plt.figure(figsize=(16, 2.75 * blocks + 1.2), facecolor="white")
    grid = fig.add_gridspec(
        blocks * 2, cols, left=0.045, right=0.99, bottom=0.04,
        top=0.88, wspace=0.06, hspace=0.18,
    )
    outcome = "SUCCESS" if info["success"] else "FAILURE"
    fig.suptitle(
        f"{info['episode']} · {outcome} · {info['length']} transitions · seed {info['seed']}",
        x=0.045, y=0.965, ha="left", fontsize=17, fontweight="bold",
    )
    fig.text(
        0.045, 0.91,
        "Representative physical states (uniform samples plus largest visual changes); "
        "agent camera above wrist camera.",
        fontsize=10, color="#374151",
    )
    with np.load(path, allow_pickle=False) as data:
        for i, t in enumerate(times):
            block, col = divmod(i, cols)
            a = fig.add_subplot(grid[2 * block, col])
            w = fig.add_subplot(grid[2 * block + 1, col])
            add_image(a, data["obs_image"][t], f"t={t}")
            add_image(w, data["obs_wrist_image"][t])
            if col == 0:
                a.set_ylabel("agent", fontsize=8)
                w.set_ylabel("wrist", fontsize=8)
    return fig


def history_indices(decision_t: int, length: int, stride: int) -> np.ndarray:
    raw = decision_t - stride * np.arange(length, 0, -1)
    return np.maximum(raw, 0)


def history_page(path: Path, decision_t: int):
    """Show exactly which stored states each history setting presents."""
    max_h = max(h for h, _ in HISTORY_SETTINGS)
    cols = 10
    rows_per_setting = 2 * math.ceil(max_h / cols)
    fig = plt.figure(
        figsize=(18, 1.85 * rows_per_setting * len(HISTORY_SETTINGS) + 1.35),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        len(HISTORY_SETTINGS) * rows_per_setting,
        cols + 1,
        width_ratios=[0.85] + [1] * cols,
        left=0.025, right=0.995, bottom=0.025, top=0.93,
        wspace=0.04, hspace=0.14,
    )
    fig.suptitle(
        f"History-window comparison · {path.stem} · decision t={decision_t}",
        x=0.025, y=0.985, ha="left", fontsize=19, fontweight="bold",
    )
    fig.text(
        0.025, 0.952,
        "Each setting shows the exact agent/wrist frames fed as historical states. "
        "Repeated t=0 cells are episode-start padding.",
        fontsize=10.5, color="#374151",
    )
    with np.load(path, allow_pickle=False) as data:
        for setting_i, (hist_length, stride) in enumerate(HISTORY_SETTINGS):
            idxs = history_indices(decision_t, hist_length, stride)
            base = setting_i * rows_per_setting
            label_ax = fig.add_subplot(grid[base:base + rows_per_setting, 0])
            label_ax.axis("off")
            label_ax.text(
                0.5, 0.5,
                f"H={hist_length}\nS={stride}\nspan={stride * hist_length}",
                ha="center", va="center", fontsize=11, fontweight="bold",
            )
            # Right-align H=12 within the same 20-slot visual footprint.
            start_slot = max_h - hist_length
            for local, t in enumerate(idxs):
                slot = start_slot + local
                block, col = divmod(slot, cols)
                a = fig.add_subplot(grid[base + block * 2, col + 1])
                w = fig.add_subplot(grid[base + block * 2 + 1, col + 1])
                add_image(a, data["obs_image"][t], f"{t}")
                add_image(w, data["obs_wrist_image"][t])
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = train_paths()
    rows = [episode_info(path, order) for order, path in enumerate(paths)]
    save_inventory(rows)

    candidates = candidate_mixtures(rows)
    (OUTPUT_DIR / "candidate_mixtures.json").write_text(
        json.dumps(candidates, indent=2) + "\n"
    )

    by_name = {row["episode"]: row for row in rows}
    representative = [
        # Four selected successes spanning search lengths.
        "ep_00044", "ep_00167", "ep_00123", "ep_00003",
        # The exact first five failures selected by n=5.
        "ep_00098", "ep_00184", "ep_00176", "ep_00197", "ep_00015",
        # Failures introduced as n grows from 5 to 10.
        "ep_00209", "ep_00085", "ep_00103", "ep_00243", "ep_00061",
    ]
    with PdfPages(OUTPUT_DIR / "representative_trajectories.pdf") as pdf:
        for name in representative:
            row = by_name[name]
            fig = trajectory_page(Path(row["path"]), row)
            pdf.savefig(fig)
            plt.close(fig)

    # Long successful trajectory: compare early/middle/late policy inputs.
    history_episode = EPISODE_DIR / "ep_00003.npz"
    with PdfPages(OUTPUT_DIR / "history_window_comparison_ep00003.pdf") as pdf:
        for decision_t in (200, 450, 750, 1000):
            fig = history_page(history_episode, decision_t)
            pdf.savefig(fig)
            plt.close(fig)

    # Export first pages as PNGs for quick inspection in tools that do not show PDFs.
    for name in ("ep_00003", "ep_00098", "ep_00184", "ep_00176", "ep_00197"):
        row = by_name[name]
        fig = trajectory_page(Path(row["path"]), row)
        fig.savefig(OUTPUT_DIR / f"trajectory_{name}.png", dpi=130)
        plt.close(fig)
    for decision_t in (450, 750, 1000):
        fig = history_page(history_episode, decision_t)
        fig.savefig(OUTPUT_DIR / f"history_ep00003_t{decision_t}.png", dpi=115)
        plt.close(fig)

    print(f"Wrote analysis artifacts to {OUTPUT_DIR}")
    print(f"Train episodes: {sum(r['success'] for r in rows)} success, "
          f"{sum(not r['success'] for r in rows)} failure")
    for candidate in candidates:
        if candidate["num_failure"] in (0, 5, 10, 15, 20, 25):
            print(
                f"25+{candidate['num_failure']:02d}: "
                f"{candidate['total_transitions']} transitions, "
                f"failure transition share="
                f"{candidate['failure_transition_fraction']:.1%}, "
                f"bands={candidate['failure_length_bands']}"
            )


if __name__ == "__main__":
    main()
