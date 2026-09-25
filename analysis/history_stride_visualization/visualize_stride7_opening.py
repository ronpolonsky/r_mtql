#!/usr/bin/env python3
"""Focused visualization of drawer opening for history length 12, stride 7."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPISODE_PATH = (
    PROJECT_ROOT
    / "cabinet-memory-sim"
    / "ManiSkill"
    / "cabinet_dataset_two_cams_more_rand"
    / "episodes"
    / "ep_00100.npz"
)
OUTPUT_DIR = Path(__file__).resolve().parent

HIST_LENGTH = 12
HIST_STRIDE = 7
POLICY_TIMES = (375, 400, 425, 450)
CONTACT_TIMES = tuple(range(350, 468, 5))

HISTORY_COLOR = "#2563eb"
CURRENT_COLOR = "#f59e0b"
SUCCESS_COLOR = "#16a34a"


def add_border(ax, color: str, width: float = 3.0) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(width)


def show_image(ax, image, color, title=None):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    add_border(ax, color)
    if title:
        ax.set_title(title, fontsize=10, pad=4, color="#111827")


def make_contact_sheet(agent_images, wrist_images, success_t):
    cols = 8
    groups = int(np.ceil(len(CONTACT_TIMES) / cols))
    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
    grid = fig.add_gridspec(
        groups * 2,
        cols,
        left=0.055,
        right=0.985,
        bottom=0.065,
        top=0.84,
        wspace=0.08,
        hspace=0.28,
    )
    fig.suptitle(
        "Physical-frame sequence around drawer opening",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.055,
        0.915,
        "Successful dataset episode ep_00100 · frames sampled every 5 physical steps · "
        "agent view above wrist view",
        fontsize=12.5,
        color="#374151",
    )

    for i, t in enumerate(CONTACT_TIMES):
        group = i // cols
        col = i % cols
        color = SUCCESS_COLOR if t == success_t else CURRENT_COLOR
        agent_ax = fig.add_subplot(grid[group * 2, col])
        wrist_ax = fig.add_subplot(grid[group * 2 + 1, col])
        show_image(agent_ax, agent_images[t], color, f"t={t}")
        show_image(wrist_ax, wrist_images[t], color)
        if col == 0:
            agent_ax.set_ylabel("agent", rotation=0, ha="right", va="center", labelpad=10)
            wrist_ax.set_ylabel("wrist", rotation=0, ha="right", va="center", labelpad=10)

    fig.text(
        0.985,
        0.025,
        "This page shows the stored trajectory continuously; blue history selection is shown on following pages.",
        ha="right",
        fontsize=10.5,
        color="#4b5563",
    )
    return fig


def make_policy_page(agent_images, wrist_images, decision_t, success_t, is_policy_query=True):
    raw = decision_t - HIST_STRIDE * np.arange(HIST_LENGTH, 0, -1)
    history = np.maximum(raw, 0)

    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
    grid = fig.add_gridspec(
        4,
        7,
        left=0.06,
        right=0.98,
        bottom=0.10,
        top=0.83,
        wspace=0.12,
        hspace=0.26,
        width_ratios=[1] * 6 + [1.12],
    )
    page_kind = "POLICY QUERY" if is_policy_query else "DIAGNOSTIC SNAPSHOT"
    fig.suptitle(
        f"History during drawer opening — {page_kind} t={decision_t}",
        x=0.06,
        y=0.96,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#111827",
    )
    timing_note = (
        "A new 25-action chunk is generated from this input."
        if is_policy_query
        else "The policy is not queried here; this state occurs inside a previously generated 25-action chunk."
    )
    fig.text(
        0.06,
        0.908,
        f"hist_length=12, stride=7 → frames {history[0]}–{history[-1]} sampled every 7 steps; "
        f"current observation t={decision_t}; success at t={success_t}. {timing_note}",
        fontsize=12.5,
        color="#374151",
    )

    for slot, t in enumerate(history):
        block = slot // 6
        col = slot % 6
        agent_ax = fig.add_subplot(grid[block * 2, col])
        wrist_ax = fig.add_subplot(grid[block * 2 + 1, col])
        show_image(agent_ax, agent_images[t], HISTORY_COLOR, f"history t={t}")
        show_image(wrist_ax, wrist_images[t], HISTORY_COLOR)
        if col == 0:
            agent_ax.set_ylabel("agent", rotation=0, ha="right", va="center", labelpad=10)
            wrist_ax.set_ylabel("wrist", rotation=0, ha="right", va="center", labelpad=10)

    current_agent = fig.add_subplot(grid[0:2, 6])
    current_wrist = fig.add_subplot(grid[2:4, 6])
    show_image(current_agent, agent_images[decision_t], CURRENT_COLOR, f"CURRENT t={decision_t}")
    show_image(current_wrist, wrist_images[decision_t], CURRENT_COLOR, "current wrist")

    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor=HISTORY_COLOR, linewidth=3, label="History observation"),
            Patch(facecolor="white", edgecolor=CURRENT_COLOR, linewidth=3, label="Current observation"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.06, 0.025),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.98,
        0.045,
        "Policy queries occur at t=0,25,50,…,450.",
        ha="right",
        fontsize=10.5,
        color="#4b5563",
    )
    return fig


def make_terminal_page(agent_images, wrist_images, next_agent, next_wrist, success_t):
    history = success_t - HIST_STRIDE * np.arange(HIST_LENGTH, 0, -1)
    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
    grid = fig.add_gridspec(
        4,
        8,
        left=0.055,
        right=0.985,
        bottom=0.10,
        top=0.83,
        wspace=0.11,
        hspace=0.26,
        width_ratios=[1] * 6 + [1.08, 1.08],
    )
    fig.suptitle(
        f"Terminal success transition — t={success_t}",
        x=0.055,
        y=0.96,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.055,
        0.908,
        "Diagnostic reconstruction with hist_length=12, stride=7. The policy was last queried at t=450; "
        "t=467 is inside that open-loop action chunk.",
        fontsize=12,
        color="#374151",
    )

    for slot, t in enumerate(history):
        block = slot // 6
        col = slot % 6
        a = fig.add_subplot(grid[block * 2, col])
        w = fig.add_subplot(grid[block * 2 + 1, col])
        show_image(a, agent_images[t], HISTORY_COLOR, f"history t={t}")
        show_image(w, wrist_images[t], HISTORY_COLOR)

    current_a = fig.add_subplot(grid[0:2, 6])
    current_w = fig.add_subplot(grid[2:4, 6])
    outcome_a = fig.add_subplot(grid[0:2, 7])
    outcome_w = fig.add_subplot(grid[2:4, 7])
    show_image(current_a, agent_images[success_t], CURRENT_COLOR, "PRE-ACTION t=467")
    show_image(current_w, wrist_images[success_t], CURRENT_COLOR, "pre-action wrist")
    show_image(outcome_a, next_agent, SUCCESS_COLOR, "SUCCESS OUTCOME")
    show_image(outcome_w, next_wrist, SUCCESS_COLOR, "reward +1")

    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor=HISTORY_COLOR, linewidth=3, label="History observation"),
            Patch(facecolor="white", edgecolor=CURRENT_COLOR, linewidth=3, label="Pre-action observation"),
            Patch(facecolor="white", edgecolor=SUCCESS_COLOR, linewidth=3, label="Post-action success"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.055, 0.025),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    return fig


def main():
    with np.load(EPISODE_PATH) as episode:
        agent = episode["obs_image"]
        wrist = episode["obs_wrist_image"]
        success_t = int(np.flatnonzero(episode["rewards"] > 0)[0])
        next_agent = episode["next_obs_image"][success_t]
        next_wrist = episode["next_obs_wrist_image"][success_t]

        with PdfPages(OUTPUT_DIR / "stride7_drawer_opening_ep00100.pdf") as pdf:
            page_builders = [
                lambda: make_contact_sheet(agent, wrist, success_t),
                *(lambda t=t: make_policy_page(agent, wrist, t, success_t) for t in POLICY_TIMES),
                lambda: make_terminal_page(agent, wrist, next_agent, next_wrist, success_t),
            ]
            for page, build_page in enumerate(page_builders, start=1):
                fig = build_page()
                pdf.savefig(fig)
                fig.savefig(OUTPUT_DIR / f"stride7_opening_{page:02d}.png", dpi=180, facecolor="white")
                plt.close(fig)

    print(f"Wrote {len(page_builders)} pages to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
