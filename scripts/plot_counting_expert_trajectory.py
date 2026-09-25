#!/usr/bin/env python3
"""Plot every state from a successful counting expert trajectory.

Each PDF page is chronological from left to right.  The upper row contains
``obs_image`` (the overview camera) and the lower row contains
``obs_wrist_image``.  The terminal ``next_obs_*`` state is appended so a
trajectory with T transitions is plotted as T + 1 states.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_DATASET = Path(
    "cabinet-memory-sim/ManiSkill/counting_dataset_scooping_v4"
)


def _scalar(episode: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in episode:
        return default
    return episode[key].item()


def select_expert_episode(dataset_dir: Path, target_count: int | None) -> Path:
    """Select the first pure successful expert episode at the requested target."""

    candidates: list[tuple[int, Path]] = []
    for path in sorted((dataset_dir / "episodes").glob("ep_*.npz")):
        with np.load(path) as episode:
            target = int(_scalar(episode, "meta_target_count", -1))
            success = bool(_scalar(episode, "meta_success", False))
            policy = str(_scalar(episode, "meta_policy", ""))
            prefix = int(_scalar(episode, "meta_planned_prefix_count", 0))
            executed = int(_scalar(episode, "meta_executed_policy_actions", 0))
        if success and policy == "success" and prefix == 0 and executed == 0:
            candidates.append((target, path))

    if not candidates:
        raise FileNotFoundError(
            f"No pure successful expert episodes found under {dataset_dir / 'episodes'}"
        )

    selected_target = (
        max(target for target, _ in candidates)
        if target_count is None
        else target_count
    )
    for target, path in candidates:
        if target == selected_target:
            return path
    available = sorted({target for target, _ in candidates})
    raise ValueError(
        f"No successful expert episode has target {selected_target}; "
        f"available targets are {available}"
    )


def load_states(episode_path: Path):
    with np.load(episode_path) as episode:
        overview = np.concatenate(
            [episode["obs_image"], episode["next_obs_image"][-1:]], axis=0
        ).astype(np.uint8)
        wrist = np.concatenate(
            [episode["obs_wrist_image"], episode["next_obs_wrist_image"][-1:]],
            axis=0,
        ).astype(np.uint8)
        metadata = {
            "target": int(_scalar(episode, "meta_target_count", -1)),
            "completed": int(_scalar(episode, "meta_completed_count", -1)),
            "success": bool(_scalar(episode, "meta_success", False)),
            "seed": int(_scalar(episode, "meta_seed", -1)),
            "cycle_boundaries": tuple(
                int(x)
                for x in episode.get(
                    "meta_cycle_boundaries", np.empty(0, dtype=np.int32)
                )
            ),
        }
    if overview.shape != wrist.shape:
        raise ValueError(
            f"Camera streams do not match: {overview.shape} versus {wrist.shape}"
        )
    return overview, wrist, metadata


def plot_trajectory(
    episode_path: Path,
    output_path: Path,
    frames_per_page: int,
    dpi: int,
) -> None:
    overview, wrist, metadata = load_states(episode_path)
    num_states = len(overview)
    boundary_states = set(metadata["cycle_boundaries"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for start in range(0, num_states, frames_per_page):
            stop = min(start + frames_per_page, num_states)
            count = stop - start
            fig, axes = plt.subplots(
                2,
                frames_per_page,
                figsize=(1.15 * frames_per_page, 2.85),
                squeeze=False,
            )
            fig.subplots_adjust(
                left=0.025,
                right=0.995,
                bottom=0.035,
                top=0.82,
                wspace=0.025,
                hspace=0.08,
            )
            page = start // frames_per_page + 1
            num_pages = (num_states + frames_per_page - 1) // frames_per_page
            fig.suptitle(
                f"{episode_path.name} | target={metadata['target']} | "
                f"states {start}–{stop - 1} of {num_states - 1} | "
                f"page {page}/{num_pages}",
                fontsize=13,
                y=0.965,
            )

            for column in range(frames_per_page):
                for row in range(2):
                    axes[row, column].axis("off")
                if column >= count:
                    continue

                state = start + column
                axes[0, column].imshow(overview[state], interpolation="nearest")
                axes[1, column].imshow(wrist[state], interpolation="nearest")
                title = f"t={state}"
                if state in boundary_states:
                    title += "  checkpoint"
                    title_color = "tab:green"
                    for axis in axes[:, column]:
                        for spine in axis.spines.values():
                            spine.set_visible(True)
                            spine.set_color("tab:green")
                            spine.set_linewidth(2.0)
                else:
                    title_color = "black"
                axes[0, column].set_title(title, fontsize=7, color=title_color, pad=2)

            axes[0, 0].set_ylabel("overview", fontsize=9)
            axes[1, 0].set_ylabel("wrist", fontsize=9)
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    print(f"episode: {episode_path}")
    print(
        f"target={metadata['target']} completed={metadata['completed']} "
        f"success={metadata['success']} seed={metadata['seed']}"
    )
    print(f"transitions={num_states - 1} plotted_states={num_states}")
    print(f"cycle_boundaries={metadata['cycle_boundaries']}")
    print(f"wrote: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--episode",
        type=Path,
        help="Explicit episode NPZ; otherwise select a pure successful expert.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Target to select; default is the largest target in the dataset.",
    )
    parser.add_argument(
        "--frames-per-page", type=int, default=20, help="Chronological columns per page."
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectory_plots/counting_v4_expert_max_target.pdf"),
    )
    args = parser.parse_args()

    if args.frames_per_page < 1:
        parser.error("--frames-per-page must be positive")
    episode = args.episode or select_expert_episode(
        args.dataset_dir, args.target_count
    )
    plot_trajectory(episode, args.output, args.frames_per_page, args.dpi)


if __name__ == "__main__":
    main()
