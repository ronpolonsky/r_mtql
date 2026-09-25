#!/usr/bin/env python3
"""Render every policy input in one successful rollout for H=12, stride=7."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from visualize_stride7_opening import make_policy_page, make_terminal_page


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPISODE_PATH = (
    PROJECT_ROOT
    / "cabinet-memory-sim"
    / "ManiSkill"
    / "cabinet_dataset_two_cams_more_rand"
    / "episodes"
    / "ep_00100.npz"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "full_success_ep00100_hist12_stride7.pdf"
ACTION_EXEC_HORIZON = 25
DENSE_OPENING_TIMES = tuple(range(425, 466, 5))


def main() -> None:
    with np.load(EPISODE_PATH) as episode:
        agent = episode["obs_image"]
        wrist = episode["obs_wrist_image"]
        rewards = episode["rewards"]
        success_t = int(np.flatnonzero(rewards > 0)[0])

        # Actual chunk boundaries plus denser diagnostic snapshots near opening.
        policy_times = tuple(range(0, success_t + 1, ACTION_EXEC_HORIZON))
        page_times = sorted(set(policy_times) | set(DENSE_OPENING_TIMES))

        with PdfPages(OUTPUT_PATH) as pdf:
            for decision_t in page_times:
                fig = make_policy_page(
                    agent,
                    wrist,
                    decision_t,
                    success_t,
                    is_policy_query=decision_t in policy_times,
                )
                pdf.savefig(fig)
                plt.close(fig)

            # This final diagnostic page is not another policy query. It shows
            # the terminal observation reached inside the chunk emitted at t=450.
            fig = make_terminal_page(
                agent,
                wrist,
                episode["next_obs_image"][success_t],
                episode["next_obs_wrist_image"][success_t],
                success_t,
            )
            pdf.savefig(fig)
            plt.close(fig)

    diagnostic_count = len(set(DENSE_OPENING_TIMES) - set(policy_times))
    print(
        f"Wrote {len(policy_times)} policy-input pages + {diagnostic_count} dense opening pages "
        f"+ 1 terminal page to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
