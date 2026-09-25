#!/usr/bin/env python3
"""Verify that a real MTQL checkpoint round-trips the complete training state.

The test performs one update, saves an Orbax checkpoint, restores it into a
fresh agent skeleton, compares model/optimizer/RNG state and metadata, then
checks that the next update matches an uninterrupted continuation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import jax
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPO_ROOT = PROJECT_ROOT.parent / "expo-ft"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPO_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPO_ROOT))

from agents import agents  # noqa: E402
from configs.task import candy_scoop  # noqa: E402
from utils.mtql_droid import (  # noqa: E402
    OpenPINormalizer,
    add_droid_target_cues,
    create_droid_history_dataset,
    process_droid_dataset,
)
from utils.mtql_orbax import (  # noqa: E402
    initialize_checkpoint_dir,
    restore_checkpoint,
    save_checkpoint,
)


def _load_agent_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("checkpoint_smoke_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def _device_sync(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value: (
            value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value
        ),
        tree,
    )


def _compare_trees(
    expected: Any,
    actual: Any,
    label: str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> float:
    expected_leaves, expected_def = jax.tree_util.tree_flatten(expected)
    actual_leaves, actual_def = jax.tree_util.tree_flatten(actual)
    if expected_def != actual_def:
        raise AssertionError(f"{label}: pytree structures differ")
    if len(expected_leaves) != len(actual_leaves):
        raise AssertionError(
            f"{label}: leaf counts differ: "
            f"{len(expected_leaves)} != {len(actual_leaves)}"
        )

    max_abs_diff = 0.0
    for index, (expected_leaf, actual_leaf) in enumerate(
        zip(expected_leaves, actual_leaves)
    ):
        expected_array = np.asarray(expected_leaf)
        actual_array = np.asarray(actual_leaf)
        if expected_array.shape != actual_array.shape:
            raise AssertionError(
                f"{label}[{index}]: shapes differ: "
                f"{expected_array.shape} != {actual_array.shape}"
            )
        if expected_array.dtype.kind in "biufc" and actual_array.dtype.kind in "biufc":
            if expected_array.size:
                max_abs_diff = max(
                    max_abs_diff,
                    float(
                        np.max(
                            np.abs(
                                expected_array.astype(np.float64)
                                - actual_array.astype(np.float64)
                            )
                        )
                    ),
                )
            np.testing.assert_allclose(
                expected_array,
                actual_array,
                rtol=rtol,
                atol=atol,
                err_msg=f"{label}[{index}] differs",
            )
        elif not np.array_equal(expected_array, actual_array):
            raise AssertionError(f"{label}[{index}] differs")
    print(
        f"[checkpoint-smoke] {label}: OK "
        f"({len(expected_leaves)} leaves, max_abs_diff={max_abs_diff:.3g})",
        flush=True,
    )
    return max_abs_diff


def _compare_metadata(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if expected.keys() != actual.keys():
        raise AssertionError(
            f"metadata keys differ: {sorted(expected)} != {sorted(actual)}"
        )
    if expected["smoke_marker"] != actual["smoke_marker"]:
        raise AssertionError("metadata smoke marker was not restored")
    if expected["seed"] != actual["seed"]:
        raise AssertionError("metadata seed was not restored")

    def canonical(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return ("ndarray", value.dtype.str, value.shape, tuple(value.tolist()))
        if isinstance(value, (tuple, list)):
            return tuple(canonical(item) for item in value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    if canonical(expected["numpy_random_state"]) != canonical(
        actual["numpy_random_state"]
    ):
        raise AssertionError("NumPy RNG metadata was not restored")
    if canonical(expected["python_random_state"]) != canonical(
        actual["python_random_state"]
    ):
        raise AssertionError("Python RNG metadata was not restored")
    print("[checkpoint-smoke] metadata: OK", flush=True)


def _training_batch(dataset: Any, indices: np.ndarray) -> dict[str, Any]:
    batch = dataset.sample(len(indices), idxs=indices)
    cue_targets = np.asarray(batch.pop("cue_targets"))
    batch.pop("episode_success")
    return add_droid_target_cues(batch, cue_targets)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="/iris/u/ronpo/expo-ft-data/candy_scoop",
    )
    parser.add_argument(
        "--norm-stats-path",
        default="/iris/u/ronpo/expo-ft-data/candy_scoop_norm_stats",
    )
    parser.add_argument(
        "--agent-config",
        default=str(PROJECT_ROOT / "agents/mtql_transformer_real.py"),
    )
    parser.add_argument("--hist-length", type=int, default=14)
    parser.add_argument("--hist-stride", type=int, default=30)
    parser.add_argument("--action-chunk-size", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-succ", type=int, default=3)
    parser.add_argument("--n-fails", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        default="/iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/checkpoint_smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agent_config_path = Path(args.agent_config)
    if not agent_config_path.is_absolute():
        agent_config_path = PROJECT_ROOT / agent_config_path

    print(f"[checkpoint-smoke] JAX devices: {jax.devices()}", flush=True)
    print(f"[checkpoint-smoke] agent_config={agent_config_path}", flush=True)
    print(
        f"[checkpoint-smoke] H={args.hist_length} S={args.hist_stride} "
        f"batch={args.batch_size} episodes={args.n_succ}+{args.n_fails}",
        flush=True,
    )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "checkpoint smoke test requires a GPU; "
            f"JAX selected backend {jax.default_backend()!r}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_config = candy_scoop.get_config()
    transitions = process_droid_dataset(
        args.dataset_path,
        task_config,
        n_success=args.n_succ,
        n_failure=args.n_fails,
    )
    normalizer = OpenPINormalizer.from_path(args.norm_stats_path)
    dataset = create_droid_history_dataset(
        transitions,
        normalizer,
        hist_length=args.hist_length,
        hist_stride=args.hist_stride,
        action_chunk_size=args.action_chunk_size,
        discount=0.99,
        image_size=args.image_size,
    )

    config = _load_agent_config(agent_config_path)
    config["train_steps"] = 2
    config["batch_size"] = args.batch_size
    config["action_chunk_size"] = args.action_chunk_size
    if "normalize_q_loss" in config:
        config["normalize_q_loss"] = False
    if "attention_entropy_target" in config:
        config["attention_entropy_target"] = ((3.5, 3.5), (3.0, 3.0))
    if config.get("hidden_dim") and "num_heads" in config:
        config["num_heads"] = int(config["hidden_dim"]) // 32

    agent_name = config["agent_name"]
    agent_class = agents[agent_name]
    example_batch = _training_batch(dataset, np.array([0], dtype=np.int64))
    agent0 = agent_class.create(args.seed, example_batch, config)

    first_indices = np.arange(args.batch_size, dtype=np.int64)
    second_indices = np.arange(
        args.batch_size,
        2 * args.batch_size,
        dtype=np.int64,
    )
    first_batch = _training_batch(dataset, first_indices)
    second_batch_a = _training_batch(dataset, second_indices)
    second_batch_b = _training_batch(dataset, second_indices)

    print("[checkpoint-smoke] running update before save", flush=True)
    agent_after_update, info_before_save = agent0.update(first_batch, step=1)
    agent_after_update = _device_sync(agent_after_update)
    _device_sync(info_before_save)

    metadata = {
        "smoke_marker": "complete_agent_roundtrip_v1",
        "seed": args.seed,
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
    }

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix="roundtrip_", dir=str(output_root))
    )
    checkpoint_dir = run_dir / "checkpoints"
    print(f"[checkpoint-smoke] checkpoint_dir={checkpoint_dir}", flush=True)

    manager, _ = initialize_checkpoint_dir(
        checkpoint_dir,
        overwrite=True,
    )
    save_checkpoint(manager, agent_after_update, 1, metadata=metadata)
    manager.wait_until_finished()
    steps = tuple(manager.all_steps())
    if steps != (1,):
        raise AssertionError(f"expected checkpoint step (1,), found {steps}")
    step_dir = checkpoint_dir / "1"
    missing_items = [
        item for item in ("agent", "params", "metadata")
        if not (step_dir / item).exists()
    ]
    if missing_items:
        raise AssertionError(
            f"checkpoint is missing Orbax items: {missing_items}"
        )
    manager.close()
    print(
        "[checkpoint-smoke] checkpoint contains agent, params, metadata",
        flush=True,
    )

    fresh_agent = agent_class.create(args.seed, example_batch, config)
    restore_manager, _ = initialize_checkpoint_dir(
        checkpoint_dir,
        resume=True,
    )
    restored_agent, restored_metadata = restore_checkpoint(
        restore_manager,
        fresh_agent,
        1,
        return_metadata=True,
    )
    restored_agent = _device_sync(restored_agent)
    restore_manager.close()
    # The fresh skeleton is no longer needed after restore. Keeping it alive
    # would unnecessarily consume another full copy of the 127M-parameter
    # agent on memory-constrained GPUs.
    del fresh_agent

    _compare_trees(
        agent_after_update.rng,
        restored_agent.rng,
        "agent RNG",
        rtol=0.0,
        atol=0.0,
    )
    _compare_trees(
        agent_after_update.network.params,
        restored_agent.network.params,
        "model and target parameters",
    )
    _compare_trees(
        agent_after_update.network.opt_state,
        restored_agent.network.opt_state,
        "optimizer state",
    )
    if agent_after_update.network.step != restored_agent.network.step:
        raise AssertionError(
            "optimizer step differs: "
            f"{agent_after_update.network.step} != "
            f"{restored_agent.network.step}"
        )
    print(
        f"[checkpoint-smoke] optimizer step: OK "
        f"({restored_agent.network.step})",
        flush=True,
    )
    if dict(agent_after_update.config) != dict(restored_agent.config):
        raise AssertionError("agent configuration differs after restore")
    print("[checkpoint-smoke] agent configuration: OK", flush=True)
    _compare_metadata(metadata, restored_metadata)

    print("[checkpoint-smoke] comparing next uninterrupted update", flush=True)
    # Run the restored branch first, copy its result to host memory, and free
    # its device buffers before running the uninterrupted branch. Otherwise
    # this test briefly holds multiple complete agents plus two compiled JAX
    # update programs, which can OOM on a 46-GiB L40S even though real training
    # keeps only one agent resident.
    resumed_agent, resumed_info = restored_agent.update(
        second_batch_b,
        step=2,
    )
    resumed_agent = _device_sync(resumed_agent)
    resumed_agent_host = jax.device_get(resumed_agent)
    resumed_info_host = jax.device_get(resumed_info)
    del restored_agent, resumed_agent, resumed_info
    gc.collect()

    continued_agent, continued_info = agent_after_update.update(
        second_batch_a,
        step=2,
    )
    continued_agent = _device_sync(continued_agent)
    continued_info = _device_sync(continued_info)
    _compare_trees(
        continued_agent,
        resumed_agent_host,
        "next agent state after restore",
        rtol=1e-5,
        atol=1e-5,
    )
    _compare_trees(
        continued_info,
        resumed_info_host,
        "next update metrics",
        rtol=1e-5,
        atol=1e-5,
    )
    print(
        "[checkpoint-smoke] PASS: checkpoint round-trip preserved the "
        "complete state and exact continuation",
        flush=True,
    )


if __name__ == "__main__":
    main()
