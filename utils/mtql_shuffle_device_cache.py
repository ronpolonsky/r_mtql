"""Device-resident batch sampler for shuffle trajectories.

The first ``cue_frames`` observations are expanded only in the *index space*:
each occupies ``hist_stride`` virtual positions. The rest of the episode keeps
one virtual position per observation. Ordinary fixed-length, fixed-stride
history sampling is then applied to this virtual trajectory. No images or
trajectories are duplicated; the cache stores only the resulting integer index
table and gathers the requested real frames on-device.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from utils.mtql_device_cache import DeviceCachedDroidHistoryDataset


def build_shuffle_history_indices(
    initial_locs: np.ndarray,
    terminal_locs: np.ndarray,
    *,
    size: int,
    hist_length: int,
    hist_stride: int,
    cue_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ordinary histories after virtual cue-frame expansion.

    For an original local index ``r``, its virtual position is ``r*S`` for
    ``r < cue_frames`` and ``cue_frames*S + (r-cue_frames)`` afterwards. This
    is exactly the sequence obtained by repeating each initial cue frame ``S``
    times and appending the remainder of the episode unchanged. For each real
    anchor we sample the usual ``H`` positions at offsets ``H*S, ..., S`` and
    map those virtual positions back to real observations. Negative virtual
    positions use the normal first-frame padding. Thus the cue frames are
    dropped naturally when the rolling virtual context moves past them.
    """
    initial_locs = np.asarray(initial_locs, dtype=np.int64)
    terminal_locs = np.asarray(terminal_locs, dtype=np.int64)
    if size <= 0 or hist_length < cue_frames or cue_frames < 1 or hist_stride < 1:
        raise ValueError(
            "Expected size>0, hist_length>=cue_frames>=1, and hist_stride>=1; "
            f"got size={size}, hist_length={hist_length}, "
            f"cue_frames={cue_frames}, hist_stride={hist_stride}."
        )
    if len(initial_locs) == 0 or len(initial_locs) != len(terminal_locs):
        raise ValueError("Episode start/terminal arrays must be nonempty and aligned.")
    if initial_locs[0] != 0 or terminal_locs[-1] != size - 1:
        raise ValueError("Episode bounds must cover the complete transition stream.")
    if np.any(terminal_locs < initial_locs) or np.any(
        initial_locs[1:] != terminal_locs[:-1] + 1
    ):
        raise ValueError("Episode bounds must be contiguous and non-overlapping.")
    history_indices = np.empty((size, hist_length), dtype=np.int64)
    history_padding = np.zeros((size, hist_length), dtype=bool)
    stride = int(hist_stride)
    cue_span = int(cue_frames) * stride

    def virtual_position(local_index: int) -> int:
        if local_index < cue_frames:
            return local_index * stride
        return cue_span + (local_index - cue_frames)

    def real_position(virtual_index: int) -> int:
        if virtual_index < cue_span:
            return virtual_index // stride
        return cue_frames + (virtual_index - cue_span)

    for episode_start, episode_end in zip(initial_locs, terminal_locs):
        for anchor in range(int(episode_start), int(episode_end) + 1):
            local_anchor = anchor - int(episode_start)
            virtual_anchor = virtual_position(local_anchor)
            virtual_history = virtual_anchor - (
                np.arange(hist_length, 0, -1, dtype=np.int64) * stride
            )
            padding = virtual_history < 0
            local_history = np.maximum(
                np.asarray(
                    [real_position(int(index)) for index in virtual_history],
                    dtype=np.int64,
                ),
                0,
            )
            history_indices[anchor] = int(episode_start) + local_history
            history_padding[anchor] = padding

    return history_indices, history_padding


class DeviceCachedShuffleHistoryDataset(DeviceCachedDroidHistoryDataset):
    """Device cache with virtual expansion of initial cue frames."""

    def __init__(self, host_dataset: Any, *, cue_frames: int = 5):
        super().__init__(host_dataset)
        history_indices, history_padding = build_shuffle_history_indices(
            self.initial_locs,
            self.terminal_locs,
            size=self.size,
            hist_length=self.hist_length,
            hist_stride=self.hist_stride,
            cue_frames=cue_frames,
        )
        device = jax.devices()[0]
        self._history_indices = jax.device_put(
            history_indices.astype(np.int32), device
        )
        self._history_padding = jax.device_put(history_padding, device)
        self.cue_frames = int(cue_frames)
        print(
            "[shuffle-history] virtual cue expansion enabled: "
            f"cue_frames={self.cue_frames} history=H{self.hist_length}/S{self.hist_stride}; "
            f"virtual cue span={self.cue_frames * self.hist_stride}; "
            "cue frames are repeated only in the integer index map; image "
            "storage is not expanded.",
            flush=True,
        )

    def set_sampling_idxs(self, sampling_idxs: Any | None) -> None:
        """Restrict random anchors, preserving the standard dataset API."""
        if sampling_idxs is None:
            self.sampling_idxs = None
            return
        sampling_idxs = np.asarray(sampling_idxs, dtype=np.int64)
        if sampling_idxs.ndim != 1 or sampling_idxs.size == 0:
            raise ValueError("sampling_idxs must be a nonempty 1D array.")
        if np.any(sampling_idxs < 0) or np.any(sampling_idxs >= self.size):
            raise ValueError("sampling_idxs contains an out-of-range transition.")
        self.sampling_idxs = sampling_idxs
