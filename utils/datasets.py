import collections
from collections import OrderedDict
from functools import partial
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        self.action_chunk_size = None  # Action chunk size for chunked action prediction; set outside the class.
        self.discount = 1.0  # Discount factor for cumulative reward over chunk; set outside the class.
        # Optional pool used for outcome-filtered training.  Keeping the full
        # arrays intact is important for episode boundaries and history
        # indexing; only the anchor transition is sampled from this pool.
        self.sampling_idxs = None

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if self.sampling_idxs is None:
            return np.random.randint(self.size, size=num_idxs)
        pool_positions = np.random.randint(len(self.sampling_idxs), size=num_idxs)
        return self.sampling_idxs[pool_positions]

    def set_sampling_idxs(self, idxs=None):
        """Restrict random sampling to explicit indices without slicing data.

        History and action chunks still index the original full episodic
        arrays, so filtering anchors does not break trajectory boundaries.
        Passing ``None`` restores sampling from every transition.
        """
        if idxs is None:
            self.sampling_idxs = None
            return

        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.ndim != 1:
            raise ValueError("sampling indices must be a one-dimensional array")
        if len(idxs) == 0:
            raise ValueError("sampling indices cannot be empty")
        if np.any(idxs < 0) or np.any(idxs >= self.size):
            raise ValueError(
                f"sampling indices must be in [0, {self.size}), got "
                f"min={idxs.min()} max={idxs.max()}"
            )
        idxs.setflags(write=False)
        self.sampling_idxs = idxs

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        if self.action_chunk_size is not None and self.action_chunk_size > 1:
            ACS = self.action_chunk_size
            # Episode end for each sampled index.
            if len(self.terminal_locs) > 0:
                search_idxs = np.minimum(
                    np.searchsorted(self.terminal_locs, idxs),
                    len(self.terminal_locs) - 1,
                )
                ep_end_locs = self.terminal_locs[search_idxs]
            else:
                ep_end_locs = np.full(len(idxs), self.size - 1)

            offsets = np.arange(ACS)[None, :]                          # (1, ACS)
            chunk_idxs = np.minimum(idxs[:, None] + offsets, ep_end_locs[:, None])  # (B, ACS)

            # Actions: (B, ACS, act_dim)
            result['actions'] = self._dict['actions'][chunk_idxs]

            # Next observations: observation at t + ACS, clamped to episode end.
            next_obs_idxs = np.minimum(idxs + ACS, ep_end_locs)
            result['next_observations'] = jax.tree_util.tree_map(
                lambda arr: arr[next_obs_idxs], self._dict['observations']
            )

            # Discounted cumulative reward over the chunk, masked to valid steps.
            valid = (idxs[:, None] + offsets) <= ep_end_locs[:, None]  # (B, ACS)
            disc_weights = self.discount ** offsets                      # (1, ACS)
            raw_rewards = self._dict['rewards'][chunk_idxs]             # (B, ACS)
            result['rewards'] = (raw_rewards * valid * disc_weights).sum(axis=1)  # (B,)

            # If the chunk reaches or passes the episode terminal, don't bootstrap.
            chunk_hits_terminal = (idxs + ACS - 1) >= ep_end_locs
            result['masks'] = np.where(chunk_hits_terminal, 0.0, result['masks']).astype(np.float32)
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class HistoryDataset(Dataset):
    """Dataset that includes a fixed-length history of past observations and actions.

    When ``hist_length`` is zero, history is disabled and history fields are
    omitted from sampled batches; the current observation remains present.

    Each sampled transition at index t gets:
      batch['history_observations']  shape: (B, hist_length, *obs_shape)
      batch['history_actions']       shape: (B, hist_length, *act_shape)

    """

    def __init__(self, *args, hist_length: int = 4, hist_stride: int = 1, **kwargs):
        if hist_length < 0:
            raise ValueError(f'hist_length must be >= 0, got {hist_length}')
        if hist_stride < 1:
            raise ValueError(f'hist_stride must be >= 1, got {hist_stride}')
        super().__init__(*args, **kwargs)
        self.hist_length = hist_length
        self.hist_stride = hist_stride

    @classmethod
    def create(cls, hist_length: int = 4, hist_stride: int = 1, freeze: bool = True, **fields):
        assert 'actions' in fields, "HistoryDataset requires an 'actions' field."
        if hist_stride < 1:
            raise ValueError(f'hist_stride must be >= 1, got {hist_stride}')
        data = fields
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data, hist_length=hist_length, hist_stride=hist_stride)

    def _get_history(self, idxs, ep_starts): # TODO decide what is the best whether to return next_obs: t+1 or t+k where k is action chunk size
        """Build raw (pre-augmentation) history arrays.

        Returns:
            hist_obs:  list of length hist_length, each a pytree with arrays (B, *obs_shape)
            hist_acts: list of length hist_length, each array (B, *act_shape)
            pad_masks: bool array (B, hist_length), True where index was before episode start
        """
        B = len(idxs)
        hist_obs  = []
        hist_acts = []
        pad_masks = np.zeros((B, self.hist_length), dtype=bool)

        for k in range(self.hist_length):
            obs_offset = (k - self.hist_length) * self.hist_stride
            act_offset = (k - self.hist_length) * self.hist_stride

            obs_raw = idxs + obs_offset
            act_raw = idxs + act_offset

            before_start_obs = obs_raw < ep_starts
            before_start_act = act_raw < ep_starts

            pad_masks[:, k] = before_start_act         # only actions get padded

            hist_obs.append(
                jax.tree_util.tree_map(lambda arr: arr[np.maximum(obs_raw, ep_starts)], self['observations'])
            )
            hist_acts.append(
                jax.tree_util.tree_map(lambda arr: arr[np.maximum(act_raw, ep_starts)], self['actions'])
            )

        return hist_obs, hist_acts, pad_masks

    def _get_next_history_indices(self, idxs, action_chunk_size):
        """Return history anchors for the action-chunk bootstrap state.

        The generic dataset has no stronger episode-aware contract than its
        global extent. Episodic adapters can override this hook to clamp the
        bootstrap state to the current episode terminal before any history is
        gathered.
        """
        return np.minimum(idxs + action_chunk_size, self.size - 1)

    def _apply_padding(self, hist_acts, pad_masks, pad_value=-1):
        """Zero out (replace with pad_value) action entries that are before episode start."""
        B = len(pad_masks)
        padded = []
        for k, acts in enumerate(hist_acts):
            mask = pad_masks[:, k]  # (B,) bool
            if not mask.any():
                padded.append(acts)
                continue
            def _pad(arr):
                arr = arr.copy()
                arr[mask] = pad_value
                return arr
            padded.append(jax.tree_util.tree_map(_pad, acts))
        return padded

    def _stack_history(self, hist_list):
        """Stack list of hist_length pytrees (each B, ...) -> (B, hist_length, ...)."""
        return jax.tree_util.tree_map(
            lambda *xs: np.stack(xs, axis=1), *hist_list
        )

    def _augment_history(self, hist_obs_stacked):
        """Apply the same random-crop augmentation as the parent class.

        hist_obs_stacked: (B, hist_length, H, W, C) or non-image — augmented in place.
        Returns augmented array.
        """
        if self.p_aug is None or np.random.rand() >= self.p_aug:
            return hist_obs_stacked

        def _aug(arr):
            if arr.ndim != 5:   # only (B, T, H, W, C)
                return arr
            B, T, H, W, C = arr.shape
            padding = 3
            crop_froms = np.random.randint(0, 2 * padding + 1, (B, 2))
            crop_froms = np.concatenate(
                [crop_froms, np.zeros((B, 1), dtype=np.int64)], axis=1
            )
            # Merge time into batch, crop, then restore shape.
            flat = arr.reshape(B * T, H, W, C)
            crop_froms_tiled = np.tile(crop_froms, (T, 1))[:B*T]  # naive tile won't do
            # Each (b, t) pair should share the same crop as timestep b — repeat B times.
            crop_froms_rep = np.repeat(crop_froms, T, axis=0)     # (B*T, 3)
            cropped = np.array(batched_random_crop(flat, crop_froms_rep, padding))
            return cropped.reshape(B, T, H, W, C)

        return jax.tree_util.tree_map(_aug, hist_obs_stacked)

    def _apply_frame_stack(self, hist_obs, idxs, ep_starts):
        """Re-stack frames when self.frame_stack is set.

        For each history slot k and each frame f in the frame stack, we look up
        obs[t + offset_k - f], clamped to the episode start.  The frames are
        concatenated on the last axis, matching the parent class behaviour.
        """
        if self.frame_stack is None:
            return hist_obs   # already correct

        B = len(idxs)
        stacked_slots = []
        for k in range(self.hist_length):
            offset = k - (self.hist_length - 1)
            base_idxs = idxs + offset              # timestep for this history slot

            frames = []
            for f in reversed(range(self.frame_stack)):
                frame_idxs = np.maximum(base_idxs - f, ep_starts)
                frames.append(
                    jax.tree_util.tree_map(lambda arr: arr[frame_idxs], self['observations'])
                )
            # Concatenate frames on last axis -> (B, *obs_shape_stacked)
            stacked_slots.append(
                jax.tree_util.tree_map(lambda *fs: np.concatenate(fs, axis=-1), *frames)
            )
        return stacked_slots

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)

        # Episode-start index for each sampled transition.
        ep_starts = self.initial_locs[
            np.searchsorted(self.initial_locs, idxs, side='right') - 1
        ]

        # Build base batch (handles frame_stack + p_aug for obs/next_obs)
        batch = self.get_subset(idxs)

        # Manually replicate parent's frame_stack logic for current obs/next_obs.
        if self.frame_stack is not None:
            obs, next_obs = [], []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, ep_starts)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))
            batch['observations'] = jax.tree_util.tree_map(lambda *a: np.concatenate(a, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *a: np.concatenate(a, axis=-1), *next_obs)

        # Augment current obs/next_obs.
        if self.p_aug is not None and np.random.rand() < self.p_aug:
            self.augment(batch, ['observations', 'next_observations'])

        # The current observation is already present in ``batch``.  A zero
        # history length therefore means that no temporal-history fields should
        # be constructed or passed to the agent.
        if self.hist_length == 0:
            return batch

        # --- History ---
        raw_hist_obs, hist_acts, pad_masks = self._get_history(idxs, ep_starts)

        # Apply frame_stack to each history slot if needed.
        raw_hist_obs = self._apply_frame_stack(raw_hist_obs, idxs, ep_starts)

        # Pad out-of-episode actions.
        hist_acts = self._apply_padding(hist_acts, pad_masks, pad_value=-1)

        # Stack into (B, hist_length, ...).
        hist_obs_stacked  = self._stack_history(raw_hist_obs)
        hist_acts_stacked = self._stack_history(hist_acts)

        # Augment history observations with the same crop family.
        hist_obs_stacked = self._augment_history(hist_obs_stacked)

        batch['history_observations'] = hist_obs_stacked
        batch['history_actions'] = hist_acts_stacked

        # --- Next-step history (for Bellman target) ---
        # The target critic evaluates Q(s_{t+ACS}, ...) so the history must be
        # anchored at t+ACS, not t+1.
        acs = self.action_chunk_size if self.action_chunk_size is not None else 1
        next_idxs = self._get_next_history_indices(idxs, acs)
        next_ep_starts = self.initial_locs[
            np.searchsorted(self.initial_locs, next_idxs, side='right') - 1
        ]
        raw_next_hist_obs, next_hist_acts, next_pad_masks = self._get_history(next_idxs, next_ep_starts)
        raw_next_hist_obs = self._apply_frame_stack(raw_next_hist_obs, next_idxs, next_ep_starts)
        next_hist_acts = self._apply_padding(next_hist_acts, next_pad_masks, pad_value=-1)
        batch['next_history_observations'] = self._stack_history(raw_next_hist_obs)
        batch['next_history_actions']      = self._stack_history(next_hist_acts)

        del batch["next_history_actions"]
        return batch


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

class HistoryReplayBuffer(HistoryDataset):
    """Pre-allocated replay buffer with O(1) inserts and history-aware sampling.

    Grows linearly up to `max_size`; never overwrites existing entries so
    episode-boundary tracking is always consistent.  Mirrors the ReplayBuffer
    API (add_transition / create_from_initial_dataset) but returns history-
    augmented batches just like HistoryDataset.sample().
    """

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size, hist_length=4, hist_stride=1):
        def make_buf(arr):
            buf = np.zeros((size, *arr.shape[1:]), dtype=arr.dtype)
            buf[:len(arr)] = arr
            return buf

        buf = jax.tree_util.tree_map(make_buf, init_dataset)
        obj = cls(buf, hist_length=hist_length, hist_stride=hist_stride)
        obj.max_size = size
        obj.size = get_size(init_dataset)
        obj._recompute_locs()
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = get_size(self._dict)
        self.size = 0   # overrides Dataset.__init__ so buffer starts empty

    def _recompute_locs(self):
        terminals = self._dict['terminals'][:self.size]
        self.terminal_locs = np.nonzero(terminals > 0)[0]
        if len(self.terminal_locs) == 0:
            self.initial_locs = np.array([0])
        else:
            all_starts = np.concatenate([[0], self.terminal_locs + 1])
            self.initial_locs = all_starts[all_starts < self.size]

    def add_transition(self, transition):
        if self.size >= self.max_size:
            raise RuntimeError(
                f'HistoryReplayBuffer is full ({self.max_size} transitions). '
                'Increase --online_buf_size.'
            )

        # Drop keys not present in the buffer (e.g. next_observations when
        # the buffer was created without it — get_subset reconstructs it).
        filtered = {k: v for k, v in transition.items() if k in self._dict}

        # Ensure matching pytree structure: FrozenDict (from buffer) vs plain
        # dict (from env transition) causes a tree_map mismatch.  Unfreeze
        # the buffer side so both are plain dicts.
        def _unfreeze(x):
            if isinstance(x, FrozenDict):
                return {k: _unfreeze(v) for k, v in x.items()}
            return x
        buf_view = _unfreeze(self._dict)

        def set_idx(buf, val):
            buf[self.size] = val

        jax.tree_util.tree_map(set_idx, buf_view, filtered)
        self.size += 1
        self._recompute_locs()

    def _compute_chunk_valid_idxs(self, acs):
        """Indices that can serve as the start of a full ACS-length action chunk."""
        if len(self.terminal_locs) == 0:
            max_valid = max(0, self.size - acs + 1)
            return np.arange(max_valid, dtype=np.int32)
        ep_starts = np.concatenate([[0], self.terminal_locs + 1]).astype(np.int32)
        ep_ends   = self.terminal_locs.astype(np.int32)
        parts = []
        for start, end in zip(ep_starts, ep_ends):
            valid_end = int(end) - acs + 2  # end - (ACS-1) inclusive → end - ACS + 2 exclusive
            if valid_end > start:
                parts.append(np.arange(start, valid_end, dtype=np.int32))
        return np.concatenate(parts) if parts else np.arange(self.size, dtype=np.int32)

    def get_random_idxs(self, num_idxs):
        acs = self.action_chunk_size
        if acs is None or acs <= 1:
            return np.random.randint(self.size, size=num_idxs)
        # Rebuild valid-index cache only when a new episode boundary is added.
        cache_key = (len(self.terminal_locs), acs)
        if getattr(self, '_chunk_cache_key', None) != cache_key:
            self._chunk_valid_idxs = self._compute_chunk_valid_idxs(acs)
            self._chunk_cache_key = cache_key
        return self._chunk_valid_idxs[np.random.randint(len(self._chunk_valid_idxs), size=num_idxs)]

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


class LazyEpisodeReplayBuffer:
    """Replay buffer that loads offline episodes lazily from per-episode NPZ files.

    Offline data is never fully materialised in RAM: episodes are loaded on
    demand from disk and kept in an LRU cache.  Online transitions collected
    during single-step-online RL are stored in a small in-memory ring buffer.

    Implements the same external API as HistoryReplayBuffer so m_main.py can
    use it transparently once the 'already a replay buffer' check is in place:
        sample(batch_size) → dict with history fields when hist_length > 0
        add_transition(transition)
        size  (int)  total offline transitions
        ['rewards'], ['actions']  for min/max stats in m_main.py

    Args:
        episode_paths:  sorted list of per-episode NPZ file paths.
        hist_length:    history window length.
        hist_stride:    stride between history steps.
        online_buf_size: max number of online transitions kept in RAM.
        ep_cache_size:  max number of episodes kept in the LRU RAM cache.
    """

    def __init__(self, episode_paths, hist_length=4, hist_stride=1,
                 online_buf_size=20_000, ep_cache_size=40, chunk_reload_interval=1000,
                 action_chunk_size=1, discount=1.0):
        if hist_length < 0:
            raise ValueError(f'hist_length must be >= 0, got {hist_length}')
        if hist_stride < 1:
            raise ValueError(f'hist_stride must be >= 1, got {hist_stride}')
        self._paths = sorted(episode_paths)
        self.hist_length = hist_length
        self.hist_stride = hist_stride
        self._ep_cache_size = min(ep_cache_size, len(self._paths))
        self._chunk_reload_interval = chunk_reload_interval
        self._action_chunk_size = action_chunk_size
        self._discount = discount

        # --- Scan episode sizes (cheap: only reads array metadata) ---
        self._ep_sizes = []
        for p in self._paths:
            with np.load(p) as d:
                self._ep_sizes.append(int(d['rewards'].shape[0]))
        self._ep_starts = np.concatenate([[0], np.cumsum(self._ep_sizes[:-1])])
        self.size = int(sum(self._ep_sizes))

        # Offline stats needed by m_main.py (load once, cheap)

        self._rewards_all  = None  # loaded lazily
        self._actions_all  = None  # loaded lazily

        # Chunk cache: load ep_cache_size random episodes into RAM, train on them
        # for chunk_reload_interval sample() calls, then rotate to a new random subset.
        self._chunk: dict = {}           # ep_idx -> dict of arrays (currently loaded)
        self._chunk_ep_idxs    = None    # flat array: chunk transition → ep_idx
        self._chunk_local_idxs = None    # flat array: chunk transition → local_idx
        self._chunk_size       = 0
        self._samples_since_reload = 0
        self._load_chunk()

        # Timing accumulators (instance variables so there's no class/instance shadowing)
        self._t_index   = 0.0
        self._t_load    = 0.0
        self._t_history = 0.0
        self._t_calls   = 0

        # Online flat buffer (pre-allocated on first add_transition call).
        # Episode boundaries are tracked via the 'terminals' field, mirroring HistoryReplayBuffer.
        self._online_buf: dict | None = None
        self._online_size  = 0
        self._online_max   = online_buf_size
        self._online_initial_locs: np.ndarray = np.array([0], dtype=np.int32)

        # Compatibility attributes expected by m_main.py
        self.p_aug = None
        self.frame_stack = None
        self.return_next_actions = False

    # ------------------------------------------------------------------
    # FrozenDict-compatible key access  (m_main.py does train_dataset['rewards'])
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        if key == 'rewards':
            return self._get_all_rewards()
        if key == 'actions':
            return self._get_all_actions()
        raise KeyError(key)

    def __contains__(self, key):
        return key in ('rewards', 'actions')

    def _get_all_rewards(self):
        if self._rewards_all is None:
            parts = []
            for p in self._paths:
                with np.load(p) as d:
                    parts.append(d['rewards'].astype(np.float32))
            self._rewards_all = np.concatenate(parts)
        return self._rewards_all

    def _get_all_actions(self):
        if self._actions_all is None:
            parts = []
            for p in self._paths:
                with np.load(p) as d:
                    parts.append(d['actions'].astype(np.float32))
            self._actions_all = np.concatenate(parts)
        return self._actions_all

    # ------------------------------------------------------------------
    # Chunk loading
    # ------------------------------------------------------------------

    def _load_chunk(self):
        """Load a random subset of ep_cache_size episodes fully into RAM."""
        selected = np.random.choice(len(self._paths), size=self._ep_cache_size, replace=False)
        print(f'[LazyEpisodeReplayBuffer] loading chunk of {self._ep_cache_size}/{len(self._paths)} episodes...')
        self._chunk = {}
        for ep_i in selected:
            with np.load(self._paths[ep_i]) as d:
                has_proprio = 'obs_proprio' in d
                has_wrist   = 'obs_wrist_image' in d
                obs_img      = d['obs_image'].copy()
                next_obs_img = d['next_obs_image'].copy()
                if has_wrist:
                    obs      = {'agent_view': obs_img,
                                'wrist_view': d['obs_wrist_image'].copy()}
                    next_obs = {'agent_view': next_obs_img,
                                'wrist_view': d['next_obs_wrist_image'].copy()}
                else:
                    obs      = {'agent_view': obs_img}
                    next_obs = {'agent_view': next_obs_img}
                if has_proprio:
                    obs['proprio']      = d['obs_proprio'].astype(np.float32)
                    next_obs['proprio'] = d['next_obs_proprio'].astype(np.float32)
                # Metadata for agent variants that optionally mask failed
                # rollouts from behavior losses. The base MTQL agent ignores
                # it; older datasets default to actor_mask=1.
                actor_weight = np.float32(
                    bool(d['meta_success'].item()) if 'meta_success' in d else True
                )
                self._chunk[int(ep_i)] = dict(
                    observations      = obs,
                    next_observations = next_obs,
                    actions           = d['actions'].astype(np.float32),
                    rewards           = d['rewards'].astype(np.float32),
                    terminals         = d['terminals'].astype(np.float32),
                    masks             = d['masks'].astype(np.float32),
                    actor_mask        = np.full(
                        d['rewards'].shape, actor_weight, dtype=np.float32
                    ),
                )
        # Build flat index arrays for this chunk (no disk I/O at sample time).
        # Exclude last ACS-1 steps per episode: those can't form a full action chunk.
        ACS = self._action_chunk_size
        ep_idx_parts    = []
        local_idx_parts = []
        for ep_i in selected:
            sz = self._ep_sizes[int(ep_i)]
            valid_sz = max(0, sz - (ACS - 1))
            if valid_sz > 0:
                ep_idx_parts.append(np.full(valid_sz, ep_i, dtype=np.int32))
                local_idx_parts.append(np.arange(valid_sz, dtype=np.int32))
        self._chunk_ep_idxs    = np.concatenate(ep_idx_parts)
        self._chunk_local_idxs = np.concatenate(local_idx_parts)
        self._chunk_size       = len(self._chunk_ep_idxs)
        self._samples_since_reload = 0
        print(f'[LazyEpisodeReplayBuffer] chunk ready ({self._chunk_size} transitions).')

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def reset_timing(self):
        self._t_index = self._t_load = self._t_history = 0.0
        self._t_calls = 0

    def log_timing(self, prefix='[LazyEpisodeReplayBuffer]'):
        if self._t_calls == 0:
            print(f'{prefix} no samples yet')
            return
        total = self._t_index + self._t_load + self._t_history
        def pct(x): return 100.0 * x / total if total > 0 else 0.0
        print(
            f'{prefix} over {self._t_calls} sample() calls — '
            f'total={1000*total:.1f}ms  '
            f'index={1000*self._t_index:.1f}ms({pct(self._t_index):.0f}%)  '
            f'load={1000*self._t_load:.1f}ms({pct(self._t_load):.0f}%)  '
            f'history={1000*self._t_history:.1f}ms({pct(self._t_history):.0f}%)'
        )

    def _sample_offline(self, n: int) -> dict:
        import time

        # Rotate to a new random chunk periodically
        self._samples_since_reload += 1
        if self._samples_since_reload >= self._chunk_reload_interval:
            self._load_chunk()

        t0 = time.perf_counter()
        chunk_idxs = np.random.randint(0, self._chunk_size, size=n)
        ep_idxs    = self._chunk_ep_idxs[chunk_idxs]
        local_idxs = self._chunk_local_idxs[chunk_idxs]
        # Stable sort groups same-episode samples together — no extra shuffle needed
        order = np.argsort(ep_idxs, kind='stable')
        self._t_index += time.perf_counter() - t0

        # Pre-allocate output arrays — avoids np.stack allocation + copy at the end
        ep0     = next(iter(self._chunk.values()))
        act_dim = ep0['actions'].shape[-1]
        HL      = self.hist_length
        ACS     = self._action_chunk_size
        is_dict_obs = isinstance(ep0['observations'], dict)

        def _alloc_obs(sample_obs, size):
            if isinstance(sample_obs, dict):
                return {k: np.empty((size, *v.shape[1:]), dtype=v.dtype)
                        for k, v in sample_obs.items()}
            return np.empty((size, *sample_obs.shape[1:]), dtype=sample_obs.dtype)

        def _alloc_hist_obs(sample_obs, size, hl):
            if isinstance(sample_obs, dict):
                return {k: np.empty((size, hl, *v.shape[1:]), dtype=v.dtype)
                        for k, v in sample_obs.items()}
            return np.empty((size, hl, *sample_obs.shape[1:]), dtype=sample_obs.dtype)

        def _index_obs(obs, idx):
            if isinstance(obs, dict):
                return {k: v[idx] for k, v in obs.items()}
            return obs[idx]

        def _assign_obs(out, idx, val):
            if isinstance(out, dict):
                for k in out:
                    out[k][idx] = val[k]
            else:
                out[idx] = val

        obs_out  = _alloc_obs(ep0['observations'], n)
        nobs_out = _alloc_obs(ep0['observations'], n)
        act_out  = np.empty((n, ACS, act_dim), dtype=np.float32)
        rew_out  = np.empty(n,                  dtype=np.float32)
        term_out = np.empty(n,                  dtype=np.float32)
        mask_out = np.empty(n,                  dtype=np.float32)
        actor_mask_out = np.empty(n,             dtype=np.float32)
        if HL > 0:
            ho_out   = _alloc_hist_obs(ep0['observations'], n, HL)
            ha_out   = np.empty((n, HL, act_dim), dtype=np.float32)
            nho_out  = _alloc_hist_obs(ep0['observations'], n, HL)
            nha_out  = np.empty((n, HL, act_dim), dtype=np.float32)

        t_load = t_hist = 0.0
        i = 0
        while i < n:
            # Identify the contiguous group of samples from the same episode
            ep_idx = int(ep_idxs[order[i]])
            j = i + 1
            while j < n and int(ep_idxs[order[j]]) == ep_idx:
                j += 1
            grp = order[i:j]          # indices into the output arrays
            g_local = local_idxs[grp] # local indices within this episode

            tl = time.perf_counter()
            ep = self._chunk[ep_idx]
            t_load += time.perf_counter() - tl

            T = self._ep_sizes[ep_idx]
            _assign_obs(obs_out,  grp, _index_obs(ep['observations'], g_local))
            # Next obs: ACS steps ahead (clamped to episode end)
            next_local = np.minimum(g_local + ACS, T - 1)
            _assign_obs(nobs_out, grp, _index_obs(ep['observations'], next_local))
            # Action chunk: gather ACS consecutive actions, flatten to (M, ACS*act_dim)
            chunk_idxs = g_local[:, None] + np.arange(ACS, dtype=np.int32)[None, :]
            chunk_idxs = np.clip(chunk_idxs, 0, T - 1)
            act_out[grp] = ep['actions'][chunk_idxs]
            # Discounted cumulative reward over the chunk, masked to valid steps.
            valid = (g_local[:, None] + np.arange(ACS, dtype=np.int32)[None, :]) < T  # (M, ACS)
            disc_weights = self._discount ** np.arange(ACS, dtype=np.float32)         # (ACS,)
            raw_rewards = ep['rewards'][chunk_idxs]                                    # (M, ACS)
            rew_out[grp] = (raw_rewards * valid * disc_weights[None, :]).sum(axis=1)
            term_out[grp] = ep['terminals'][g_local]
            mask_out[grp] = ep['masks'][g_local]
            actor_mask_out[grp] = ep['actor_mask'][g_local]
            # If the chunk reaches or passes the episode terminal, don't bootstrap.
            chunk_hits_terminal = (g_local + ACS - 1) >= (T - 1)
            mask_out[grp] = np.where(chunk_hits_terminal, 0.0, mask_out[grp]).astype(np.float32)

            if HL > 0:
                th = time.perf_counter()
                ho, ha, nho, nha = self._build_history_batch(ep, g_local, T)
                t_hist += time.perf_counter() - th

                _assign_obs(ho_out,  grp, ho)
                _assign_obs(nho_out, grp, nho)
                ha_out[grp]  = ha
                nha_out[grp] = nha

            i = j

        self._t_load    += t_load
        self._t_history += t_hist
        self._t_calls   += 1

        result = dict(
            observations              = obs_out,
            next_observations         = nobs_out,
            actions                   = act_out,
            rewards                   = rew_out,
            terminals                 = term_out,
            masks                     = mask_out,
            actor_mask                = actor_mask_out,
        )
        if HL > 0:
            result.update(
                history_observations      = ho_out,
                history_actions           = ha_out,
                next_history_observations = nho_out,
                next_history_actions      = nha_out,
            )
        return result

    def _build_history_batch(self, ep: dict, local_idxs: np.ndarray, T: int):
        """Vectorized history build for M samples from the same episode.

        local_idxs: (M,) int array of positions within the episode.
        Returns (ho, ha, nho, nha) each of shape (M, HL, ...).
        """
        HL, HS = self.hist_length, self.hist_stride
        offsets = np.arange(-HL, 0, dtype=np.int32) * HS  # (HL,)

        def _hist(base_idxs):
            # base_idxs: (M,)  →  ts: (M, HL)
            ts      = base_idxs[:, None] + offsets[None, :]
            clamped = np.clip(ts, 0, T - 1)
            obs = ep['observations']
            if isinstance(obs, dict):
                ho = {k: v[clamped] for k, v in obs.items()}  # (M, HL, ...)
            else:
                ho = obs[clamped]                              # (M, HL, *obs_shape)
            ha      = ep['actions'][clamped].copy()            # (M, HL, act_dim)
            ha[ts < 0] = -1.0
            return ho, ha

        next_idxs = np.minimum(local_idxs + self._action_chunk_size, T - 1)
        ho,  ha  = _hist(local_idxs)
        nho, nha = _hist(next_idxs)
        return ho, ha, nho, nha

    def _build_history(self, ep: dict, local_idx: int, T: int):
        """Single-sample wrapper (used by tests)."""
        ho, ha, nho, nha = self._build_history_batch(
            ep, np.array([local_idx], dtype=np.int32), T
        )
        return ho[0], ha[0], nho[0], nha[0]

    def _online_recompute_locs(self):
        """Recompute episode start indices from terminals, mirroring HistoryReplayBuffer."""
        terminals = self._online_buf['terminals'][:self._online_size]
        terminal_locs = np.nonzero(terminals > 0)[0]
        if len(terminal_locs) == 0:
            all_starts = np.array([0], dtype=np.int32)
        else:
            all_starts = np.concatenate([[0], terminal_locs + 1]).astype(np.int32)
        self._online_initial_locs = all_starts[all_starts < self._online_size]

    def _get_online_valid_idxs(self):
        """Return flat indices into the online buffer excluding last ACS-1 steps per episode."""
        ACS = self._action_chunk_size
        if ACS <= 1:
            return np.arange(self._online_size, dtype=np.int32)
        # Episode ends are one step before each episode start (except the first)
        ep_ends = np.concatenate([self._online_initial_locs[1:] - 1, [self._online_size - 1]])
        valid = []
        for start, end in zip(self._online_initial_locs, ep_ends):
            valid_end = end - (ACS - 1) + 1  # exclude last ACS-1 steps
            if valid_end > start:
                valid.append(np.arange(start, valid_end, dtype=np.int32))
        return np.concatenate(valid) if valid else np.array([], dtype=np.int32)

    def _sample_online(self, n: int) -> dict:
        """Sample n transitions from the flat online buffer with history and action chunks."""
        valid_idxs = self._get_online_valid_idxs()
        idxs = valid_idxs[np.random.randint(0, len(valid_idxs), size=n)]

        # Episode start for each sampled index (for history clamping)
        ep_starts = self._online_initial_locs[
            np.searchsorted(self._online_initial_locs, idxs, side='right') - 1
        ]
        # Episode end for each sampled index (for chunk clamping)
        ep_end_idxs = np.searchsorted(self._online_initial_locs, idxs, side='right')
        ep_ends = np.where(
            ep_end_idxs < len(self._online_initial_locs),
            np.concatenate([self._online_initial_locs[1:], [self._online_size]])[ep_end_idxs - 1],
            self._online_size,
        ) - 1

        buf = self._online_buf
        ACS = self._action_chunk_size
        HL  = self.hist_length
        act_dim = buf['actions'].shape[-1]

        def _alloc_obs(size):
            obs = buf['observations']
            if isinstance(obs, dict):
                return {k: np.empty((size, *v.shape[1:]), dtype=v.dtype) for k, v in obs.items()}
            return np.empty((size, *obs.shape[1:]), dtype=obs.dtype)

        def _alloc_hist_obs(size, hl):
            obs = buf['observations']
            if isinstance(obs, dict):
                return {k: np.empty((size, hl, *v.shape[1:]), dtype=v.dtype) for k, v in obs.items()}
            return np.empty((size, hl, *obs.shape[1:]), dtype=obs.dtype)

        def _index_obs(idx):
            obs = buf['observations']
            if isinstance(obs, dict):
                return {k: v[idx] for k, v in obs.items()}
            return obs[idx]

        # Current obs
        obs_out = _index_obs(idxs)

        # Next obs: ACS steps ahead, clamped to episode end
        next_idxs = np.minimum(idxs + ACS, ep_ends)
        nobs_out = _index_obs(next_idxs)

        # Action chunk: ACS consecutive actions clamped to episode end
        chunk_idxs = idxs[:, None] + np.arange(ACS, dtype=np.int32)[None, :]
        chunk_idxs = np.minimum(chunk_idxs, ep_ends[:, None])
        act_out = buf['actions'][chunk_idxs]

        rew_out  = buf['rewards'][idxs]
        term_out = buf['terminals'][idxs]
        mask_out = buf['masks'][idxs]
        actor_mask_out = (
            buf['actor_mask'][idxs]
            if 'actor_mask' in buf
            else np.ones(n, dtype=np.float32)
        )

        # If the chunk reaches or passes the episode terminal, don't bootstrap.
        chunk_hits_terminal = (idxs + ACS - 1) >= ep_ends
        mask_out = np.where(chunk_hits_terminal, 0.0, mask_out).astype(np.float32)

        result = dict(
            observations              = obs_out,
            next_observations         = nobs_out,
            actions                   = act_out,
            rewards                   = rew_out,
            terminals                 = term_out,
            masks                     = mask_out,
            actor_mask                = actor_mask_out,
        )

        if HL > 0:
            # History: look back HL*stride steps, clamped to episode start.
            HS = self.hist_stride
            offsets = np.arange(-HL, 0, dtype=np.int32) * HS  # (HL,)
            hist_ts      = idxs[:, None] + offsets[None, :]          # (n, HL)
            hist_ts_next = (idxs + ACS)[:, None] + offsets[None, :]  # anchored at t+ACS
            hist_ts_clamp      = np.maximum(hist_ts,      ep_starts[:, None])
            hist_ts_next_clamp = np.maximum(hist_ts_next, ep_starts[:, None])

            ho_out  = _alloc_hist_obs(n, HL)
            nho_out = _alloc_hist_obs(n, HL)
            ha_out  = np.empty((n, HL, act_dim), dtype=np.float32)
            nha_out = np.empty((n, HL, act_dim), dtype=np.float32)

            obs = buf['observations']
            acts = buf['actions']
            for k in range(HL):
                flat      = hist_ts_clamp[:, k]
                flat_next = hist_ts_next_clamp[:, k]
                if isinstance(obs, dict):
                    for key in obs:
                        ho_out[key][:, k]  = obs[key][flat]
                        nho_out[key][:, k] = obs[key][flat_next]
                else:
                    ho_out[:, k]  = obs[flat]
                    nho_out[:, k] = obs[flat_next]
                ha_chunk      = acts[flat].copy()
                nha_chunk     = acts[flat_next].copy()
                # Pad actions before episode start with -1.
                before      = hist_ts[:, k] < ep_starts
                before_next = hist_ts_next[:, k] < ep_starts
                ha_chunk[before]       = -1.0
                nha_chunk[before_next] = -1.0
                ha_out[:, k]  = ha_chunk
                nha_out[:, k] = nha_chunk

            result.update(
                history_observations      = ho_out,
                history_actions           = ha_out,
                next_history_observations = nho_out,
                next_history_actions      = nha_out,
            )

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict:
        valid_online = len(self._get_online_valid_idxs()) if self._online_size > 0 else 0
        if valid_online == 0:
            return self._sample_offline(batch_size)

        # Mix: ~67 % offline, 33 % online (capped so we don't oversample sparse online data)
        n_online  = min(batch_size // 3, valid_online)
        n_offline = batch_size - n_online

        offline = self._sample_offline(n_offline)
        online  = self._sample_online(n_online)

        def _concat(a, b):
            if isinstance(a, dict):
                return {k: np.concatenate([a[k], b[k]], axis=0) for k in a}
            return np.concatenate([a, b], axis=0)

        return {k: _concat(offline[k], online[k]) for k in offline}

    def add_transition(self, transition: dict):
        if self._online_size >= self._online_max:
            if not hasattr(self, '_online_full_warned'):
                self._online_full_warned = 0
            self._online_full_warned += 1
            if self._online_full_warned % 1000 == 1:
                print(f'[LazyEpisodeReplayBuffer] online buffer full ({self._online_max} transitions); '
                      f'new transitions are being dropped. Consider increasing --online_buf_size.')
            return
        if self._online_buf is None:
            def _alloc(v, max_size):
                if isinstance(v, dict):
                    return {k: np.zeros((max_size, *np.asarray(vv).shape), dtype=np.asarray(vv).dtype)
                            for k, vv in v.items()}
                v = np.asarray(v)
                return np.zeros((max_size, *v.shape), dtype=v.dtype)
            self._online_buf = {k: _alloc(v, self._online_max) for k, v in transition.items()}

        ptr = self._online_size
        for k, v in transition.items():
            buf = self._online_buf[k]
            if isinstance(buf, dict):
                for kk in buf:
                    buf[kk][ptr] = v[kk]
            else:
                buf[ptr] = v
        self._online_size += 1
        self._online_recompute_locs()


def load_visual_ogbench_npz(npz_path, max_demos=0):
    """Load an OGBench visual NPZ fully into RAM as a dict compatible with
    HistoryDataset.create().

    Returns dict with keys: observations, actions, rewards, terminals, masks.
    observations is a dict {'image': uint8, 'proprio': float32} if qpos/qvel
    are present, otherwise just a plain uint8 image array.
    """
    with np.load(npz_path) as d:
        observations = d['observations'].copy()
        actions   = d['actions'].astype(np.float32)
        terminals = d['terminals'].astype(np.float32)
        rewards   = d.get('rewards')
        masks     = d.get('masks')
        qpos = d.get('qpos')
        qvel = d.get('qvel')

    if rewards is None:
        rewards = np.zeros(len(terminals), dtype=np.float32)
    else:
        rewards = rewards.astype(np.float32)
    if masks is None:
        masks = (1.0 - terminals).astype(np.float32)
    else:
        masks = masks.astype(np.float32)

    if qpos is not None and qvel is not None:
        proprio = np.concatenate([qpos, qvel], axis=-1).astype(np.float32)
        obs = {'image': observations, 'proprio': proprio}
    else:
        obs = {'image': observations}

    # Truncate to max_demos complete episodes if requested.
    if max_demos > 0:
        terminal_idxs = np.nonzero(terminals > 0)[0]
        if len(terminal_idxs) > max_demos:
            cutoff = int(terminal_idxs[max_demos - 1]) + 1
            obs = {k: v[:cutoff] for k, v in obs.items()}
            actions   = actions[:cutoff]
            rewards   = rewards[:cutoff]
            terminals = terminals[:cutoff]
            masks     = masks[:cutoff]
            print(f'[load_visual_ogbench_npz] Truncated to {max_demos} demos ({cutoff} transitions)')

    print(f'[load_visual_ogbench_npz] Loaded {len(terminals)} transitions from {os.path.basename(npz_path)}')
    return dict(observations=obs, actions=actions, rewards=rewards, terminals=terminals, masks=masks)


def split_ogbench_npz_into_episodes(npz_path, cache_dir):
    """Stream through an OGBench NPZ and write one small NPZ per episode.

    Only one episode's worth of image observations is held in RAM at a time,
    so this works even when the full dataset does not fit in memory.

    OGBench multi-task datasets do not include rewards/masks; this function
    synthesises them (rewards=0, masks=1-terminals) as placeholders.

    Args:
        npz_path: Path to the monolithic OGBench .npz file.
        cache_dir: Directory to write per-episode NPZ files into.

    Returns:
        Sorted list of episode file paths.
    """
    import struct
    import zipfile

    existing = sorted(glob.glob(os.path.join(cache_dir, 'ep_*.npz')))
    if existing:
        print(f'[split_ogbench] Found {len(existing)} cached episode files in {cache_dir}')
        return existing

    os.makedirs(cache_dir, exist_ok=True)

    def _read_npy_header(f):
        """Return (shape, dtype, data_start_offset) for an open .npy file handle."""
        magic = f.read(6)
        if magic != b'\x93NUMPY':
            raise ValueError('Not a valid .npy file')
        major = f.read(1)[0]
        f.read(1)  # minor version
        hlen = struct.unpack('<H', f.read(2))[0] if major == 1 else struct.unpack('<I', f.read(4))[0]
        header = eval(f.read(hlen).decode('latin1'))
        shape = tuple(header['shape'])
        dtype = np.dtype(header['descr'])
        data_offset = 6 + 2 + (2 if major == 1 else 4) + hlen
        return shape, dtype, data_offset

    with zipfile.ZipFile(npz_path, 'r') as zf:
        names = set(zf.namelist())

        def _load_full(key):
            npy = key + '.npy'
            if npy not in names:
                return None
            with zf.open(npy) as f:
                return np.lib.format.read_array(f)

        # Load small 1-D arrays fully (they fit in RAM easily).
        terminals = _load_full('terminals')
        actions   = _load_full('actions')
        rewards   = _load_full('rewards')
        masks     = _load_full('masks')

        # Synthesise rewards/masks if absent (multi-task goal-conditioned envs).
        if rewards is None:
            rewards = np.zeros(len(terminals), dtype=np.float32)
        if masks is None:
            masks = (1.0 - terminals).astype(np.float32)

        ep_end_locs   = np.nonzero(terminals > 0)[0]
        ep_start_locs = np.concatenate([[0], ep_end_locs[:-1] + 1])

        print(f'[split_ogbench] Splitting {len(ep_end_locs)} episodes from '
              f'{os.path.basename(npz_path)} into {cache_dir} ...')

        # Load proprio (qpos + qvel) if available in the NPZ.
        qpos = _load_full('qpos')
        qvel = _load_full('qvel')
        if qpos is not None and qvel is not None:
            proprio_all = np.concatenate([qpos, qvel], axis=-1).astype(np.float32)
        else:
            print(
                f'[split_ogbench] WARNING: qpos/qvel not found in {os.path.basename(npz_path)}. '
                'Proprio will NOT be included in the dataset observations. '
                'The agent will only receive image observations.'
            )
            proprio_all = None

        paths = []
        # Stream observations sequentially — only one episode in RAM at a time.
        with zf.open('observations.npy') as obs_f:
            obs_shape, obs_dtype, _ = _read_npy_header(obs_f)
            row_bytes = int(np.prod(obs_shape[1:])) * obs_dtype.itemsize

            for i, (start, end) in enumerate(zip(ep_start_locs, ep_end_locs + 1)):
                n = int(end - start)
                raw = obs_f.read(n * row_bytes)
                img_ep = np.frombuffer(raw, dtype=obs_dtype).reshape((n,) + obs_shape[1:]).copy()

                ep_path = os.path.join(cache_dir, f'ep_{i:06d}.npz')
                save_kwargs = dict(
                    observations_image=img_ep,
                    actions=actions[start:end],
                    rewards=rewards[start:end],
                    terminals=terminals[start:end],
                    masks=masks[start:end],
                )
                if proprio_all is not None:
                    save_kwargs['observations_proprio'] = proprio_all[start:end]
                np.savez(ep_path, **save_kwargs)
                paths.append(ep_path)
                del img_ep

    print(f'[split_ogbench] Done. {len(paths)} episode files written.')
    return paths


class LazyOGBenchEpisodeReplayBuffer(LazyEpisodeReplayBuffer):
    """LazyEpisodeReplayBuffer for OGBench visual datasets.

    Overrides _load_chunk to handle OGBench's flat observation format
    (key 'observations' contains uint8 images) instead of the search_cabinet
    per-episode format.
    """

    def _load_chunk(self):
        selected = np.random.choice(len(self._paths), size=self._ep_cache_size, replace=False)
        print(f'[LazyOGBenchEpisodeReplayBuffer] loading chunk of {self._ep_cache_size}/{len(self._paths)} episodes...')
        self._chunk = {}
        for ep_i in selected:
            with np.load(self._paths[ep_i]) as d:
                keys = set(d.files)
                has_proprio = 'observations_proprio' in keys
                if has_proprio:
                    obs = {
                        'image': d['observations_image'].copy(),
                        'proprio': d['observations_proprio'].astype(np.float32),
                    }
                else:
                    obs = d['observations_image'].copy()
                self._chunk[int(ep_i)] = dict(
                    observations=obs,
                    actions=d['actions'].astype(np.float32),
                    rewards=d['rewards'].astype(np.float32),
                    terminals=d['terminals'].astype(np.float32),
                    masks=d['masks'].astype(np.float32),
                )
        ACS = self._action_chunk_size
        ep_idx_parts = []
        local_idx_parts = []
        for ep_i in selected:
            sz = self._ep_sizes[int(ep_i)]
            valid_sz = max(0, sz - (ACS - 1))
            if valid_sz > 0:
                ep_idx_parts.append(np.full(valid_sz, ep_i, dtype=np.int32))
                local_idx_parts.append(np.arange(valid_sz, dtype=np.int32))
        self._chunk_ep_idxs = np.concatenate(ep_idx_parts)
        self._chunk_local_idxs = np.concatenate(local_idx_parts)
        self._chunk_size = len(self._chunk_ep_idxs)
        self._samples_since_reload = 0
        print(f'[LazyOGBenchEpisodeReplayBuffer] chunk ready ({self._chunk_size} transitions).')


def add_mc_returns(data, discount):
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=np.float32)
    terminals = np.asarray(data["terminals"], dtype=np.float32)

    mc = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros_like(rewards[0], dtype=np.float32)
    for i in range(len(rewards) - 1, -1, -1):
        bootstrap = masks[i] * (1.0 - terminals[i])
        running = rewards[i] + float(discount) * running * bootstrap
        mc[i] = running

    out = dict(data)
    out["mc_returns"] = mc
    return out
