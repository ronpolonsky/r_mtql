#!/usr/bin/env python3
"""Render two-camera timelines and proprio traces for proposed curation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SELECTION = json.loads((HERE / "proposed_selection.json").read_text())
INVENTORY = json.loads((HERE / "episode_phase_inventory.json").read_text())["episodes"]
BY_EPISODE = {item["episode"]: item for item in INVENTORY}
OUT = HERE / "candidate_reports"


def make_report(category: str, episode_name: str):
    item = BY_EPISODE[episode_name]
    path = Path(item["path"])
    with np.load(path, allow_pickle=False) as data:
        agent = data["obs_image"]
        wrist = data["obs_wrist_image"]
        proprio = data["obs_proprio"][:, :7]
        rewards = data["rewards"]

    times = np.linspace(0, len(rewards) - 1, 25, dtype=int)
    fig = plt.figure(figsize=(15, 17), facecolor="white")
    grid = fig.add_gridspec(
        6, 5, height_ratios=[1, 1, 1, 1, 1, 1.35],
        left=0.025, right=0.99, top=0.93, bottom=0.045,
        wspace=0.04, hspace=0.13,
    )
    outcome = "SUCCESS" if item["success"] else "FAILURE"
    fig.suptitle(
        f"{episode_name} | {category} | {outcome} | {item['length']} steps | "
        f"{item['phase_count']} proprio phases",
        x=0.025, y=0.975, ha="left", fontsize=18, fontweight="bold",
    )
    fig.text(
        0.025, 0.945,
        f"phase clusters={item['drawer_sequence']}  compressed={item['compressed_drawer_sequence']}  "
        f"revisits={item['revisit_count']} (cluster identities are only a screening heuristic)",
        fontsize=10, color="#374151",
    )

    for slot, t in enumerate(times):
        row, col = divmod(slot, 5)
        ax = fig.add_subplot(grid[row, col])
        tile = np.concatenate([agent[t], wrist[t]], axis=0)
        ax.imshow(tile)
        ax.set_title(f"t={t}", fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("agent\n\nwrist", fontsize=8)

    ax = fig.add_subplot(grid[5, :])
    for joint in range(7):
        ax.plot(proprio[:, joint], linewidth=0.85, label=f"q{joint}")
    q_distance = np.linalg.norm(proprio - proprio[0], axis=1)
    ax.plot(q_distance, color="black", linewidth=1.5, alpha=0.8, label="distance to home")
    for phase in item["phases"]:
        ax.axvline(phase["start"], color=f"C{phase['cluster']}", alpha=0.5, linewidth=1)
        ax.text(
            phase["start"], ax.get_ylim()[1], f"p{phase['cluster']}",
            color=f"C{phase['cluster']}", fontsize=7, va="top", ha="left",
        )
    positive = np.flatnonzero(rewards > 0)
    if len(positive):
        ax.axvline(int(positive[0]), color="limegreen", linewidth=2.2, label="first positive reward")
    ax.set_xlim(0, len(rewards) - 1)
    ax.set_xlabel("physical environment step")
    ax.set_ylabel("joint position / home distance")
    ax.grid(alpha=0.15)
    ax.legend(ncol=9, loc="upper center", fontsize=7)

    category_dir = OUT / category
    category_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(category_dir / f"{episode_name}.png", dpi=125)
    plt.close(fig)


def main():
    for category, episodes in SELECTION.items():
        print(category, len(episodes), flush=True)
        for episode in episodes:
            print(" ", episode, flush=True)
            make_report(category, episode)


if __name__ == "__main__":
    main()
