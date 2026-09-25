#!/usr/bin/env python3
"""Visualize the exact observation history seen at cabinet policy decisions.

This reconstructs the inputs that a chunked policy would receive while
following one stored dataset trajectory. It does not claim that the learned
policy generated the stored trajectory.
"""

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
HIST_STRIDES = (3, 5, 7)
ACTION_EXEC_HORIZON = 25
SUMMARY_DECISIONS = (25, 225, 450)

PAD_COLOR = "#9ca3af"
VALID_COLOR = "#2563eb"
CURRENT_COLOR = "#f59e0b"
SUCCESS_COLOR = "#16a34a"


def history_indices(t: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return clamped frame indices and a mask for pre-episode padding."""
    raw = t - stride * np.arange(HIST_LENGTH, 0, -1)
    return np.maximum(raw, 0), raw < 0


def add_border(ax, color: str, width: float = 2.5) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(width)


def draw_observation(
    ax,
    image: np.ndarray,
    *,
    border_color: str,
    label: str | None = None,
) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    add_border(ax, border_color)
    if label is not None:
        ax.set_title(label, fontsize=8, pad=3, color="#111827")


def make_decision_figure(
    agent_images: np.ndarray,
    wrist_images: np.ndarray,
    *,
    decision_t: int,
    success_t: int,
    success_outcome: tuple[np.ndarray, np.ndarray] | None = None,
) -> plt.Figure:
    terminal_page = success_outcome is not None
    num_columns = 14 if terminal_page else 13
    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white", constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=6,
        ncols=num_columns,
        left=0.055,
        right=0.985,
        bottom=0.095,
        top=0.84,
        wspace=0.055,
        hspace=0.19,
        width_ratios=[1] * 12 + [1.08] + ([1.08] if terminal_page else []),
    )

    remaining = success_t - decision_t
    page_title = (
        f"Successful cabinet rollout ep_00100 — terminal transition t={decision_t}"
        if terminal_page
        else f"Successful cabinet rollout ep_00100 — policy execution t={decision_t}"
    )
    fig.suptitle(
        page_title,
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#111827",
    )
    page_subtitle = (
        "Diagnostic view only: obs t=467 is before the terminal action; the green column is its "
        "post-action next observation with reward +1. The deployed policy was last queried at t=450."
        if terminal_page
        else f"12 history observations; policy executes 25 actions open-loop; "
        f"terminal success at t={success_t} ({remaining} steps from this decision)"
    )
    fig.text(
        0.055,
        0.915,
        page_subtitle,
        ha="left",
        va="center",
        fontsize=11.5,
        color="#374151",
    )

    for stride_row, stride in enumerate(HIST_STRIDES):
        idxs, padded = history_indices(decision_t, stride)
        row_agent = stride_row * 2
        row_wrist = row_agent + 1
        first_agent_ax = None
        first_wrist_ax = None

        for slot, (frame_idx, is_padding) in enumerate(zip(idxs, padded)):
            border = PAD_COLOR if is_padding else VALID_COLOR
            suffix = " · PAD" if is_padding else ""
            agent_ax = fig.add_subplot(grid[row_agent, slot])
            wrist_ax = fig.add_subplot(grid[row_wrist, slot])
            if slot == 0:
                first_agent_ax = agent_ax
                first_wrist_ax = wrist_ax
            draw_observation(
                agent_ax,
                agent_images[frame_idx],
                border_color=border,
                label=f"t={frame_idx}{suffix}",
            )
            draw_observation(
                wrist_ax,
                wrist_images[frame_idx],
                border_color=border,
            )

        current_agent_ax = fig.add_subplot(grid[row_agent, 12])
        current_wrist_ax = fig.add_subplot(grid[row_wrist, 12])
        draw_observation(
            current_agent_ax,
            agent_images[decision_t],
            border_color=CURRENT_COLOR,
            label=f"CURRENT t={decision_t}",
        )
        draw_observation(
            current_wrist_ax,
            wrist_images[decision_t],
            border_color=CURRENT_COLOR,
        )

        if terminal_page:
            outcome_agent_ax = fig.add_subplot(grid[row_agent, 13])
            outcome_wrist_ax = fig.add_subplot(grid[row_wrist, 13])
            draw_observation(
                outcome_agent_ax,
                success_outcome[0],
                border_color=SUCCESS_COLOR,
                label="SUCCESS OUTCOME",
            )
            draw_observation(
                outcome_wrist_ax,
                success_outcome[1],
                border_color=SUCCESS_COLOR,
            )

        first_agent_ax.set_ylabel(
            f"stride {stride}\nagent",
            rotation=0,
            ha="right",
            va="center",
            labelpad=9,
            fontsize=10,
            fontweight="bold",
            color="#111827",
        )
        first_wrist_ax.set_ylabel(
            "wrist",
            rotation=0,
            ha="right",
            va="center",
            labelpad=9,
            fontsize=9,
            color="#4b5563",
        )

    legend = [
        Patch(facecolor="white", edgecolor=PAD_COLOR, linewidth=2.5, label="Initial-frame padding"),
        Patch(facecolor="white", edgecolor=VALID_COLOR, linewidth=2.5, label="Real past observation"),
        Patch(facecolor="white", edgecolor=CURRENT_COLOR, linewidth=2.5, label="Current observation"),
    ]
    if terminal_page:
        legend.append(
            Patch(facecolor="white", edgecolor=SUCCESS_COLOR, linewidth=2.5, label="Post-action success")
        )
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.018),
        ncol=len(legend),
        frameon=False,
        fontsize=10.5,
    )
    fig.text(
        0.985,
        0.038,
        "Each column is one observation: agent view (top) + wrist view (bottom).",
        ha="right",
        va="center",
        fontsize=10.5,
        color="#4b5563",
    )
    return fig


def make_overview_figure(
    agent_images: np.ndarray,
    wrist_images: np.ndarray,
    *,
    success_t: int,
) -> plt.Figure:
    decisions = np.arange(0, success_t + 1, ACTION_EXEC_HORIZON)
    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
    grid = fig.add_gridspec(
        4,
        1,
        left=0.07,
        right=0.97,
        bottom=0.09,
        top=0.84,
        height_ratios=(1.0, 1.0, 1.35, 2.0),
        hspace=0.52,
    )
    fig.suptitle(
        "What image history does the policy see?",
        x=0.07,
        y=0.955,
        ha="left",
        fontsize=25,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.07,
        0.905,
        "Representative successful training episode ep_00100: 468 transitions, terminal success at t=467",
        ha="left",
        fontsize=13,
        color="#374151",
    )

    timeline_ax = fig.add_subplot(grid[0])
    timeline_ax.hlines(0, 0, success_t, color="#94a3b8", linewidth=4)
    timeline_ax.scatter(decisions, np.zeros_like(decisions), s=55, color=VALID_COLOR, zorder=3)
    timeline_ax.scatter([success_t], [0], s=120, marker="*", color=SUCCESS_COLOR, zorder=4)
    for t in decisions:
        label_y = -0.2 if t >= success_t - ACTION_EXEC_HORIZON else (0.17 if (t // 25) % 2 == 0 else -0.2)
        timeline_ax.text(t, label_y, str(t), ha="center", fontsize=8)
    timeline_ax.text(success_t, 0.22, "success 467", ha="right", fontsize=10, color=SUCCESS_COLOR)
    timeline_ax.set_xlim(-5, success_t + 7)
    timeline_ax.set_ylim(-0.35, 0.35)
    timeline_ax.axis("off")
    timeline_ax.set_title(
        "Policy executions (blue): a new 25-action chunk is generated only at t=0,25,50,…,450",
        loc="left",
        fontsize=12,
        pad=8,
        fontweight="bold",
    )

    coverage_ax = fig.add_subplot(grid[1])
    y_positions = np.arange(len(HIST_STRIDES))[::-1]
    colors = ("#38bdf8", "#2563eb", "#4338ca")
    for y, stride, color in zip(y_positions, HIST_STRIDES, colors):
        coverage_ax.barh(y, HIST_LENGTH * stride, color=color, height=0.56)
        coverage_ax.text(
            HIST_LENGTH * stride + 2,
            y,
            f"{HIST_LENGTH} × {stride} = {HIST_LENGTH * stride} physical steps",
            va="center",
            fontsize=11,
        )
    coverage_ax.set_yticks(y_positions, [f"stride {s}" for s in HIST_STRIDES])
    coverage_ax.set_xlim(0, 125)
    coverage_ax.set_xlabel("Maximum history horizon")
    coverage_ax.spines[["top", "right"]].set_visible(False)
    coverage_ax.set_title(
        "Stride changes temporal coverage while keeping 12 history slots fixed",
        loc="left",
        fontsize=12,
        pad=8,
        fontweight="bold",
    )

    explanation_ax = fig.add_subplot(grid[2])
    explanation_ax.axis("off")
    explanation = (
        r"At policy execution $t$, stride $s$ supplies:  "
        r"$[\mathrm{obs}_{t-12s},\mathrm{obs}_{t-11s},\ldots,\mathrm{obs}_{t-s}]$"
        "\n\n"
        "Negative indices are replaced by the initial observation (gray padding).  "
        "The current observation is supplied separately (orange).\n"
        "Every observation contains one agent-view image, one wrist-view image, and 9-D proprio."
    )
    explanation_ax.text(
        0,
        0.92,
        explanation,
        ha="left",
        va="top",
        fontsize=14,
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    snapshots = [0, 225, 450, success_t]
    snapshot_grid = grid[3].subgridspec(2, len(snapshots), wspace=0.11, hspace=0.12)
    for col, t in enumerate(snapshots):
        agent_ax = fig.add_subplot(snapshot_grid[0, col])
        wrist_ax = fig.add_subplot(snapshot_grid[1, col])
        border = SUCCESS_COLOR if t == success_t else CURRENT_COLOR
        draw_observation(agent_ax, agent_images[t], border_color=border, label=f"agent · t={t}")
        draw_observation(wrist_ax, wrist_images[t], border_color=border, label=f"wrist · t={t}")
    fig.text(
        0.07,
        0.315,
        "Episode landmarks (these are stored dataset images, not a learned-policy rollout)",
        fontsize=12,
        fontweight="bold",
        color="#111827",
    )
    return fig


def save_png(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=180, facecolor="white")


def main() -> None:
    with np.load(EPISODE_PATH) as episode:
        agent_images = episode["obs_image"]
        wrist_images = episode["obs_wrist_image"]
        next_agent_images = episode["next_obs_image"]
        next_wrist_images = episode["next_obs_wrist_image"]
        rewards = episode["rewards"]

        success_indices = np.flatnonzero(rewards > 0)
        if len(success_indices) != 1:
            raise ValueError(f"Expected exactly one success reward, found {success_indices}")
        success_t = int(success_indices[0])
        if success_t != len(rewards) - 1:
            raise ValueError("Expected reconstructed episode to end at first success")

        overview = make_overview_figure(agent_images, wrist_images, success_t=success_t)
        save_png(overview, "01_overview.png")

        summary_figures = []
        for page_number, decision_t in enumerate(SUMMARY_DECISIONS, start=2):
            fig = make_decision_figure(
                agent_images,
                wrist_images,
                decision_t=decision_t,
                success_t=success_t,
            )
            save_png(fig, f"{page_number:02d}_decision_t{decision_t:03d}.png")
            summary_figures.append(fig)

        terminal_figure = make_decision_figure(
            agent_images,
            wrist_images,
            decision_t=success_t,
            success_t=success_t,
            success_outcome=(next_agent_images[success_t], next_wrist_images[success_t]),
        )
        save_png(terminal_figure, "05_terminal_success_t467.png")

        with PdfPages(OUTPUT_DIR / "mentor_summary_success_ep00100.pdf") as pdf:
            for fig in summary_figures:
                pdf.savefig(fig)
            pdf.savefig(terminal_figure)

        for fig in summary_figures:
            plt.close(fig)
        plt.close(terminal_figure)

        with PdfPages(OUTPUT_DIR / "full_success_ep00100_all_policy_executions.pdf") as pdf:
            for decision_t in range(0, success_t + 1, ACTION_EXEC_HORIZON):
                fig = make_decision_figure(
                    agent_images,
                    wrist_images,
                    decision_t=decision_t,
                    success_t=success_t,
                )
                pdf.savefig(fig)
                plt.close(fig)
            terminal_figure = make_decision_figure(
                agent_images,
                wrist_images,
                decision_t=success_t,
                success_t=success_t,
                success_outcome=(next_agent_images[success_t], next_wrist_images[success_t]),
            )
            pdf.savefig(terminal_figure)
            plt.close(terminal_figure)

        plt.close(overview)

    print(f"Wrote visualization artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
