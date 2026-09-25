"""Orbax checkpoint helpers for native MTQL agents.

These helpers use the ``agent`` and ``params`` items configured by EXPO's
``initialize_checkpoint_dir`` without changing MTQL's agent interface.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np


def initialize_checkpoint_dir(
    checkpoint_dir: Any,
    *,
    max_to_keep: int = 100,
    keep_period: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> tuple[Any, bool]:
    """Create the same Orbax checkpoint manager used by EXPO-FT."""
    # Keep these imports lazy so native MTQL checkpoint loading does not
    # require EXPO's newer Orbax installation.
    import etils.epath as epath
    import orbax.checkpoint as ocp

    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. "
                "Use --overwrite or --resume to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    item_handlers = {
        "agent": ocp.PyTreeCheckpointHandler(),
        "params": ocp.PyTreeCheckpointHandler(),
        "metadata": ocp.PyTreeCheckpointHandler(),
    }
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers=item_handlers,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            create=False,
            # etils' local filesystem backend rejects ``mode=None`` in
            # Orbax 0.11.x; explicitly use its supported default mode.
            file_options=ocp.options.FileOptions(path_permission_mode=0o777),
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    if resuming and tuple(checkpoint_manager.all_steps()) in [(), (0,)]:
        logging.info(
            "Checkpoint directory exists, but does not contain any "
            "checkpoints. Aborting resume."
        )
        resuming = False

    return checkpoint_manager, resuming


def _split_params(agent: Any) -> tuple[Any, dict[str, Any]]:
    """Separate network parameters from the remaining agent state."""
    params = {
        "network_params": agent.network.params,
    }
    agent_without_params = agent.replace(
        network=agent.network.replace(params={}),
    )
    return agent_without_params, params


def _merge_params(agent: Any, params: dict[str, Any]) -> Any:
    """Restore network parameters to an MTQL agent checkpoint skeleton."""
    if "network_params" not in params:
        raise KeyError("Orbax checkpoint is missing 'network_params'.")
    return agent.replace(
        network=agent.network.replace(
            params=params["network_params"],
        ),
    )


def save_checkpoint(
    checkpoint_manager: Any,
    agent: Any,
    step: int,
    metadata: dict[str, Any] | None = None,
):
    """Save the complete MTQL state through an EXPO-style Orbax manager."""
    agent_state, params = _split_params(agent)
    items = {
        "agent": agent_state,
        "params": params,
        "metadata": {} if metadata is None else metadata,
    }
    return checkpoint_manager.save(step, items)


def restore_checkpoint(
    checkpoint_manager: Any,
    agent: Any,
    step: int | None = None,
    *,
    return_metadata: bool = False,
    restore_optimizer: bool = True,
) -> Any:
    """Restore MTQL from an EXPO-style Orbax checkpoint.

    Evaluation can set ``restore_optimizer=False`` because it only needs the
    saved network parameters. This avoids requiring the freshly initialized
    optimizer pytree to have the exact serialized structure used by training.
    """
    agent_state, params = _split_params(agent)
    if restore_optimizer:
        restored = checkpoint_manager.restore(
            step,
            items={
                "agent": agent_state,
                "params": params,
            },
        )
    else:
        restored = checkpoint_manager.restore(
            step,
            items={"params": params},
        )
    restored_agent = _merge_params(
        agent if not restore_optimizer else restored["agent"],
        restored["params"],
    )
    if return_metadata:
        # Metadata has no fixed restore structure, so ask Orbax to restore the
        # complete composite item when the caller explicitly requests it.
        all_items = checkpoint_manager.restore(step)
        return restored_agent, all_items.get("metadata", {})
    return restored_agent


def restore_rng_metadata(metadata: dict[str, Any]) -> None:
    """Restore Python and NumPy RNG states after Orbax tuple serialization.

    Orbax may deserialize tuples as lists. NumPy accepts that representation,
    but ``random.setstate`` requires the nested internal state to be tuples.
    Normalize both states here so checkpoint resume is independent of the
    serializer's sequence representation.
    """

    def as_tuple_tree(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return tuple(as_tuple_tree(item) for item in value)
        return value

    if "numpy_random_state" in metadata:
        state = as_tuple_tree(metadata["numpy_random_state"])
        if len(state) != 5:
            raise ValueError("Invalid NumPy RNG state in checkpoint metadata")
        np.random.set_state(
            (
                state[0],
                np.asarray(state[1], dtype=np.uint32),
                int(state[2]),
                int(state[3]),
                float(state[4]),
            )
        )
    if "python_random_state" in metadata:
        random.setstate(as_tuple_tree(metadata["python_random_state"]))
