#!/usr/bin/env python3
"""Summarize local MTQL cabinet runs for alpha/sweep selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOTS = (
    ROOT / "exp" / "cabinet-v2-25-success-5-failure",
    ROOT / "exp" / "cabinet-original-s25-f5",
)


def read_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({key: float(value) for key, value in row.items() if value != ""})
            except ValueError:
                continue
    return rows


def avg(rows: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if key in row]
    return mean(values) if values else None


def main() -> None:
    results = []
    for root in RUN_ROOTS:
        if not root.is_dir():
            continue
        for run_dir in sorted(root.iterdir()):
            flags_path = run_dir / "flags.json"
            if not flags_path.is_file():
                continue
            flags = json.loads(flags_path.read_text(encoding="utf-8"))
            train = read_rows(run_dir / "train.csv")
            evaluation = read_rows(run_dir / "eval.csv")
            if not train:
                continue
            train.sort(key=lambda row: row.get("step", -1))
            evaluation.sort(key=lambda row: row.get("step", -1))
            max_step = int(train[-1]["step"])
            # Use up to the final 100k steps to estimate the converged loss scale.
            tail = [row for row in train if row.get("step", 0) > max_step - 100_000]
            alpha = float(flags.get("agent", {}).get("alpha", float("nan")))
            normalize_q = bool(flags.get("agent", {}).get("normalize_q_loss", False))
            distill = avg(tail, "training/actor/distill_loss")
            q_actor = avg(tail, "training/actor/q_loss")
            successes = [row["evaluation/success"] for row in evaluation
                         if "evaluation/success" in row]
            results.append({
                "run": run_dir.name,
                "root": root.name,
                "seed": flags.get("seed"),
                "history": flags.get("hist_length"),
                "stride": flags.get("hist_stride"),
                "alpha": alpha,
                "normalize_q_loss": normalize_q,
                "max_step": max_step,
                "tail_bc_flow": avg(tail, "training/actor/bc_flow_loss"),
                "tail_distill": distill,
                "tail_alpha_distill": alpha * distill if distill is not None else None,
                "tail_actor_q_loss": q_actor,
                "tail_actor_loss": avg(tail, "training/actor/actor_loss"),
                "tail_actor_batch_fraction": avg(
                    tail, "training/actor/actor_batch_fraction"
                ),
                "tail_critic_q_loss": avg(tail, "training/critic/q_loss"),
                "tail_q": avg(tail, "training/actor/q"),
                "eval_points": len(successes),
                "eval_best": max(successes) if successes else None,
                "eval_last": successes[-1] if successes else None,
                "eval_tail5": mean(successes[-5:]) if successes else None,
            })

    complete = [row for row in results if row["max_step"] >= 1_400_000]
    h20s50_all = [row for row in results if row["history"] == 20 and row["stride"] == 50]
    h20s50_all.sort(key=lambda row: row["max_step"], reverse=True)
    h20s50 = [row for row in complete if row["history"] == 20 and row["stride"] == 50]
    h20s50.sort(key=lambda row: (row["eval_best"] or -1), reverse=True)

    output = {
        "all_runs_with_train_csv": len(results),
        "complete_runs": len(complete),
        "complete_h20_s50_runs": len(h20s50),
        "complete_h20_s50": h20s50,
        "most_progressed_h20_s50": h20s50_all[:20],
    }
    destination = Path(__file__).with_name("previous_loss_summary.json")
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
