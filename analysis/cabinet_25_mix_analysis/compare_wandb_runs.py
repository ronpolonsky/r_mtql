#!/usr/bin/env python3
"""Download compact BC/MTQL cabinet run comparisons from W&B."""

from __future__ import annotations

import json
from pathlib import Path

import wandb


PROJECTS = (
    "new_mtql/bc-cabinet",
    "new_mtql/new-offline-cabinet_v2",
)
METRIC = "evaluation/success"
OUT = Path(__file__).resolve().parent / "wandb_run_comparison.json"


def scalar_config(config: dict) -> dict:
    agent = config.get("agent", {})
    return {
        "seed": config.get("seed"),
        "env_name": config.get("env_name"),
        "max_demos": config.get("max_demos"),
        "successful_demos_only": config.get("successful_demos_only"),
        "num_success_demos": config.get("num_success_demos"),
        "num_failure_demos": config.get("num_failure_demos"),
        "hist_length": config.get("hist_length"),
        "hist_stride": config.get("hist_stride"),
        "eval_episodes": config.get("eval_episodes"),
        "eval_interval": config.get("eval_interval"),
        "train_steps": config.get("train_steps"),
        "online_warmup_steps": config.get("online_warmup_steps"),
        "wandb_run_group": config.get("wandb_run_group"),
        "agent_name": agent.get("agent_name"),
        "alpha": agent.get("alpha"),
        "attention_entropy_target": agent.get("attention_entropy_target"),
        "normalize_q_loss": agent.get("normalize_q_loss"),
        "hidden_dim": agent.get("hidden_dim"),
        "num_heads": agent.get("num_heads"),
    }


def metric_history(run) -> list[dict]:
    rows = []
    # Evaluations are logged only every 50k updates. ``history`` asks W&B for
    # a compact sampled series and returns all ~20-30 evaluation points here;
    # ``scan_history`` would scan every training row and is unnecessarily slow.
    for row in run.history(keys=[METRIC], samples=10_000, pandas=False):
        value = row.get(METRIC)
        if value is not None:
            rows.append({"step": int(row["_step"]), "success": float(value)})
    return rows


def summarize_history(history: list[dict]) -> dict:
    if not history:
        return {
            "num_evals": 0,
            "last_step": None,
            "last_success": None,
            "best_step": None,
            "best_success": None,
            "mean_last_5": None,
            "mean_last_10": None,
        }
    best = max(history, key=lambda row: row["success"])
    return {
        "num_evals": len(history),
        "last_step": history[-1]["step"],
        "last_success": history[-1]["success"],
        "best_step": best["step"],
        "best_success": best["success"],
        "mean_last_5": sum(r["success"] for r in history[-5:]) / min(5, len(history)),
        "mean_last_10": sum(r["success"] for r in history[-10:]) / min(10, len(history)),
    }


def main() -> None:
    api = wandb.Api(timeout=120)
    output = []
    for project in PROJECTS:
        for listed in api.runs(project):
            # Fetch individually so W&B materializes the complete config.
            run = api.run(f"{project}/{listed.id}")
            history = metric_history(run)
            output.append({
                "project": project,
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": str(run.created_at),
                "config": scalar_config(dict(run.config)),
                "metrics": summarize_history(history),
                "history": history,
            })
            print(
                project.split("/")[-1], run.id, run.state,
                scalar_config(dict(run.config)), summarize_history(history),
                flush=True,
            )
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {len(output)} runs to {OUT}")


if __name__ == "__main__":
    main()
