"""Device-resident observation cache for fixed-history Egg experiments.

This module is deliberately opt-in.  It wraps an already validated
``DroidHistoryDataset`` without changing the standard dataset or trainer.
Compact per-transition observations are copied to the active JAX device once,
while episode-safe history and action-chunk index tables are precomputed once.
Each training batch is then gathered directly on the accelerator.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from utils.mtql_droid import MTQL_IMAGE_KEYS, _get_jitted_augmenter


def _tree_nbytes(tree: Any) -> int:
    return sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree_util.tree_leaves(tree))


@jax.jit
def _gather_critic_batch(
    observations,
    actions,
    terminals,
    actor_mask,
    history_indices,
    history_padding,
    action_chunk_indices,
    next_indices,
    chunk_returns,
    chunk_masks,
    idxs,
):
    """Gather one complete critic transition batch on the active device."""
    batch_history_indices = history_indices[idxs]
    batch_next_indices = next_indices[idxs]
    batch_history_padding = history_padding[idxs]

    history_actions = actions[batch_history_indices]
    history_actions = jnp.where(
        batch_history_padding[..., None],
        jnp.asarray(-1, dtype=history_actions.dtype),
        history_actions,
    )

    return {
        # Match DroidHistoryDataset's exact PyTree contract: current and next
        # observations are mutable dictionaries, while history observations
        # retain the source FrozenDict structure.
        "observations": {
            key: array[idxs] for key, array in observations.items()
        },
        "next_observations": {
            key: array[batch_next_indices]
            for key, array in observations.items()
        },
        "history_observations": FrozenDict(
            {
                key: array[batch_history_indices]
                for key, array in observations.items()
            }
        ),
        "next_history_observations": FrozenDict(
            {
                key: array[history_indices[batch_next_indices]]
                for key, array in observations.items()
            }
        ),
        "history_actions": history_actions,
        "actions": actions[action_chunk_indices[idxs]],
        "rewards": chunk_returns[idxs],
        "masks": chunk_masks[idxs],
        "terminals": terminals[idxs],
        "actor_mask": actor_mask[idxs],
    }


@jax.jit
def _gather_actor_batch(
    observations,
    actions,
    history_indices,
    action_chunk_indices,
    idxs,
):
    """Gather only fields consumed by the success-only actor loss."""
    return {
        "observations": FrozenDict(
            {key: array[idxs] for key, array in observations.items()}
        ),
        "history_observations": FrozenDict(
            {
                key: array[history_indices[idxs]]
                for key, array in observations.items()
            }
        ),
        "actions": actions[action_chunk_indices[idxs]],
    }


class DeviceCachedDroidHistoryDataset:
    """Read-only GPU/accelerator-backed view of a DroidHistoryDataset.

    The wrapper preserves the standard trainer-facing sampling contract.  The
    compact observation stream is resident on device; histories remain views
    expressed by small integer index tables and are materialized only for the
    sampled batch.
    """

    def __init__(self, host_dataset: Any):
        if host_dataset.frame_stack is not None:
            raise ValueError("Device cache does not support frame_stack.")
        if int(host_dataset.hist_length) <= 0:
            raise ValueError("Device cache requires a positive history length.")

        self.size = int(host_dataset.size)
        self.hist_length = int(host_dataset.hist_length)
        self.hist_stride = int(host_dataset.hist_stride)
        self.action_chunk_size = int(host_dataset.action_chunk_size or 1)
        self.discount = float(host_dataset.discount)
        self.frame_stack = None
        self.p_aug = None
        self.sampling_idxs = (
            None
            if host_dataset.sampling_idxs is None
            else np.asarray(host_dataset.sampling_idxs, dtype=np.int64)
        )
        self.initial_locs = np.asarray(host_dataset.initial_locs, dtype=np.int64)
        self.terminal_locs = np.asarray(host_dataset.terminal_locs, dtype=np.int64)

        if not len(self.terminal_locs) or self.terminal_locs[-1] != self.size - 1:
            raise ValueError(
                "Device cache requires every episode, including the last, to terminate."
            )

        # Keep the trainer's small host-side diagnostics synchronous.  In
        # particular, episode_success is converted to NumPy every step for a
        # contract assertion, so returning it from device would force a GPU
        # synchronization and device-to-host transfer on every update.
        self._host_fields = {
            key: np.asarray(host_dataset[key])
            for key in (
                "actions",
                "rewards",
                "terminals",
                "masks",
                "episode_success",
                "actor_mask",
            )
        }

        anchors = np.arange(self.size, dtype=np.int64)
        episode_starts = self.initial_locs[
            np.searchsorted(self.initial_locs, anchors, side="right") - 1
        ]
        episode_ends = self.terminal_locs[
            np.searchsorted(self.terminal_locs, anchors, side="left")
        ]

        history_offsets = (
            np.arange(self.hist_length, dtype=np.int64) - self.hist_length
        ) * self.hist_stride
        raw_history_indices = anchors[:, None] + history_offsets[None, :]
        history_padding = raw_history_indices < episode_starts[:, None]
        history_indices = np.maximum(
            raw_history_indices,
            episode_starts[:, None],
        )

        action_offsets = np.arange(self.action_chunk_size, dtype=np.int64)
        raw_action_indices = anchors[:, None] + action_offsets[None, :]
        action_chunk_indices = np.minimum(
            raw_action_indices,
            episode_ends[:, None],
        )
        valid_action_steps = raw_action_indices <= episode_ends[:, None]
        next_indices = np.minimum(
            anchors + self.action_chunk_size,
            episode_ends,
        )

        rewards = self._host_fields["rewards"]
        discount_weights = self.discount ** action_offsets.astype(np.float32)
        chunk_returns = np.sum(
            rewards[action_chunk_indices]
            * valid_action_steps
            * discount_weights[None, :],
            axis=1,
        ).astype(np.float32)
        chunk_hits_terminal = (
            anchors + self.action_chunk_size - 1 >= episode_ends
        )
        chunk_masks = np.where(
            chunk_hits_terminal,
            0.0,
            self._host_fields["masks"],
        ).astype(np.float32)

        device = jax.devices()[0]
        transfer_start = time.time()
        self._observations = jax.device_put(
            jax.tree_util.tree_map(np.asarray, host_dataset["observations"]),
            device,
        )
        self._actions = jax.device_put(self._host_fields["actions"], device)
        self._terminals = jax.device_put(self._host_fields["terminals"], device)
        self._actor_mask = jax.device_put(self._host_fields["actor_mask"], device)
        self._history_indices = jax.device_put(
            history_indices.astype(np.int32), device
        )
        self._history_padding = jax.device_put(history_padding, device)
        self._action_chunk_indices = jax.device_put(
            action_chunk_indices.astype(np.int32), device
        )
        self._next_indices = jax.device_put(next_indices.astype(np.int32), device)
        self._chunk_returns = jax.device_put(chunk_returns, device)
        self._chunk_masks = jax.device_put(chunk_masks, device)

        # device_put is asynchronous.  Finish the one-time transfer now so the
        # first training-step timing cannot hide cache population work.
        for leaf in jax.tree_util.tree_leaves(
            (
                self._observations,
                self._actions,
                self._history_indices,
                self._action_chunk_indices,
                self._next_indices,
            )
        ):
            leaf.block_until_ready()

        observation_gib = _tree_nbytes(self._observations) / 1024**3
        total_gib = _tree_nbytes(
            (
                self._observations,
                self._actions,
                self._terminals,
                self._actor_mask,
                self._history_indices,
                self._history_padding,
                self._action_chunk_indices,
                self._next_indices,
                self._chunk_returns,
                self._chunk_masks,
            )
        ) / 1024**3
        print(
            "[device-cache] ready: "
            f"device={device} transitions={self.size} "
            f"history=H{self.hist_length}/S{self.hist_stride} "
            f"observations={observation_gib:.2f} GiB "
            f"total={total_gib:.2f} GiB "
            f"transfer_seconds={time.time() - transfer_start:.2f}",
            flush=True,
        )

    def __contains__(self, key: str) -> bool:
        return key == "observations" or key in self._host_fields

    def __getitem__(self, key: str):
        if key == "observations":
            return self._observations
        return self._host_fields[key]

    def get_random_idxs(self, num_idxs: int) -> np.ndarray:
        if self.sampling_idxs is None:
            return np.random.randint(self.size, size=num_idxs)
        pool_positions = np.random.randint(len(self.sampling_idxs), size=num_idxs)
        return self.sampling_idxs[pool_positions]

    def sample(self, batch_size: int, idxs=None) -> dict[str, Any]:
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        idxs = np.asarray(idxs, dtype=np.int32)
        if idxs.ndim != 1 or len(idxs) != batch_size:
            raise ValueError(
                f"Sample indices must have shape ({batch_size},), got {idxs.shape}."
            )
        batch = _gather_critic_batch(
            self._observations,
            self._actions,
            self._terminals,
            self._actor_mask,
            self._history_indices,
            self._history_padding,
            self._action_chunk_indices,
            self._next_indices,
            self._chunk_returns,
            self._chunk_masks,
            jax.device_put(idxs),
        )
        batch["episode_success"] = self._host_fields["episode_success"][idxs]
        return batch

    def sample_actor_batch(self, batch_size: int, idxs=None) -> dict[str, Any]:
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        idxs = np.asarray(idxs, dtype=np.int32)
        if idxs.ndim != 1 or len(idxs) != batch_size:
            raise ValueError(
                "Actor sample indices must have shape "
                f"({batch_size},), got {idxs.shape}."
            )
        batch = _gather_actor_batch(
            self._observations,
            self._actions,
            self._history_indices,
            self._action_chunk_indices,
            jax.device_put(idxs),
        )
        batch["episode_success"] = self._host_fields["episode_success"][idxs]
        return batch


def augment_droid_batch_device(
    batch: Mapping[str, Any], rng: Any
) -> dict[str, Any]:
    """Device-only equivalent of the standard fused Egg augmentation path."""
    result = dict(batch)
    sequence_fields = (
        "history_observations",
        "observations",
        "next_history_observations",
        "next_observations",
    )
    for field in sequence_fields:
        if field in result:
            result[field] = dict(result[field])

    groups = (
        ("history_observations", "observations"),
        ("next_history_observations", "next_observations"),
    )
    active_groups = [
        tuple(field for field in fields if field in result)
        for fields in groups
        if any(field in result for field in fields)
    ]
    if not active_groups:
        raise KeyError("Batch has no MTQL observation fields to augment.")

    camera_keys = [
        key
        for key in MTQL_IMAGE_KEYS
        if any(key in result[field] for fields in active_groups for field in fields)
    ]
    if not camera_keys:
        raise KeyError(
            f"Batch contains none of the expected image keys: {MTQL_IMAGE_KEYS}."
        )

    packed_sequences = []
    packed_specs = []
    for camera_key in camera_keys:
        for fields in active_groups:
            group_fields = [
                field for field in fields if camera_key in result[field]
            ]
            if not group_fields:
                continue
            pieces = []
            field_specs = []
            for field in group_fields:
                original = jnp.asarray(result[field][camera_key])
                if original.ndim == 4:
                    images = original[:, None]
                elif original.ndim == 5:
                    images = original
                else:
                    raise ValueError(
                        f"Expected batched images in {field}/{camera_key}, "
                        f"got {original.shape}."
                    )
                pieces.append(images)
                field_specs.append((field, images.shape[1], original.ndim))
            packed_sequences.append(jnp.concatenate(pieces, axis=1))
            packed_specs.append((camera_key, field_specs))

    packed_shapes = {sequence.shape[1:] for sequence in packed_sequences}
    if len(packed_shapes) != 1:
        raise ValueError(
            "Device-cache augmentation requires equal packed temporal shapes; "
            f"got {sorted(map(str, packed_shapes))}."
        )

    packed = jnp.stack(packed_sequences, axis=0)
    image_height, image_width = packed.shape[-3:-1]
    group_rngs = jax.random.split(rng, len(packed_sequences))
    augmented = _get_jitted_augmenter(image_height, image_width)(
        packed, group_rngs
    )

    for group_index, (camera_key, field_specs) in enumerate(packed_specs):
        offset = 0
        for field, length, original_ndim in field_specs:
            images = augmented[group_index, :, offset : offset + length]
            result[field][camera_key] = (
                images[:, 0] if original_ndim == 4 else images
            )
            offset += length
    return result
