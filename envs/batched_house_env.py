"""Vectorized HouseEnv: N environments backed by batched numpy arrays.

Holds all parallel environment states in a single set of (N, ...) arrays and
vectorises every operation — no subprocess spawning, no pickling overhead.

Exposes the stable-baselines3 VecEnv interface so it can be used as a drop-in
replacement for SubprocVecEnv / DummyVecEnv.

Optional built-in history:
    Setting hist_length > 0 maintains an (N, window, obs_dim) rolling buffer
    internally, so the HistoryWrapper is no longer needed.  Output obs shape is
    hist_length * (base_obs_dim + act_dim) + base_obs_dim, matching the
    HistoryWrapper convention exactly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from envs.house_env import (
    GRID_SIZE as G,
    PASSABLE,
    ROOM_CELLS,
    ROOM_BOUNDS,
    GRID_DIST,
    GRID_DIST_MAX,
    GOAL_ROOM_X_MIN, GOAL_ROOM_X_MAX,
    GOAL_ROOM_Y_MIN, GOAL_ROOM_Y_MAX,
    DELIVERY_ORDER,
    _parse_num_objects,
)


# Precompute cell centres once (G, G)
_IY, _IX = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
_CX = (_IX + 0.5).astype(np.float32)   # (G, G)
_CY = (_IY + 0.5).astype(np.float32)   # (G, G)

_BASE_OBS_DIM_CACHE: Dict[int, int] = {}


def _base_obs_dim(num_objects: int, small_obs: bool = False) -> int:
    vis_dim = 2 * num_objects if small_obs else G * G
    return 4 + vis_dim + 2 * num_objects + (num_objects + 1)


class BatchedHouseEnv(VecEnv):
    """Batched, subprocess-free vectorised HouseEnv.

    All N environment states live in (N, ...) numpy arrays.  Every step is a
    sequence of vectorised numpy operations; no processes or locks are needed.

    Parameters
    ----------
    num_envs:        Number of parallel environments.
    num_objects:     Number of objects to deliver.
    max_steps:       Episode length limit.
    hist_length:     Number of history steps to include in the observation
                     (0 = no history, matching plain HouseEnv).
    hist_stride:     Stride between history samples.
    fov_radius, fov_angle, max_rotation, pickup_radius, drop_goal_radius:
                     Same as HouseEnv.
    count_reward:    Dense (#delivered) vs sparse (+1/delivery) reward.
    seed:            RNG seed.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        num_envs:          int   = 4,
        num_objects:       int   = 3,
        max_steps:         int   = 100,
        hist_length:       int   = 0,
        hist_stride:       int   = 1,
        fov_radius:        float = 3.5,
        fov_angle:         float = 2.0 * np.pi / 3,
        max_rotation:      float = np.pi / 4,
        pickup_radius:     float = 0.7,
        drop_goal_radius:  float = 1.0,
        count_reward:      bool  = False,
        dense_reward:      bool  = False,
        fully_observable:  bool  = False,
        small_obs:         bool  = False,
        seed:              int   = 0,
        online_reset_seed: int | None = None,
        randomization:     str   = 'large',
        image_obs:         bool  = False,
    ):
        self._online_reset_seed = online_reset_seed
        self.randomization = randomization
        self.small_obs = small_obs
        self.image_obs = image_obs
        self._base_obs_dim = _base_obs_dim(num_objects, small_obs)
        act_dim = 4

        if image_obs:
            from envs.house_env import IMG_OBS_SIZE
            observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0, high=255,
                    shape=(IMG_OBS_SIZE, IMG_OBS_SIZE, 3), dtype=np.uint8,
                ),
                'proprio': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32,
                ),
            })
        elif hist_length > 0:
            obs_dim = hist_length * (self._base_obs_dim + act_dim) + self._base_obs_dim
            observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
        else:
            obs_dim = self._base_obs_dim
            observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32,
        )
        super().__init__(num_envs, observation_space, action_space)
        self.render_mode = 'rgb_array'

        self.num_objects      = num_objects
        self.max_steps        = max_steps
        self.hist_length      = hist_length
        self.hist_stride      = hist_stride
        self._window          = hist_length * hist_stride  # rolling buffer length
        self.fov_radius       = fov_radius
        self.fov_angle        = fov_angle
        self.half_fov         = fov_angle / 2.0
        self.max_rotation     = max_rotation
        self.pickup_radius    = pickup_radius
        self.drop_goal_radius = drop_goal_radius
        self.count_reward     = count_reward
        self.dense_reward     = dense_reward
        self.fully_observable = fully_observable
        self._max_dist        = GRID_DIST_MAX

        self.np_random = np.random.default_rng(seed)

        N = num_envs
        # Per-env physics state
        self.agent_pos   = np.zeros((N, 2),                    dtype=np.float32)
        self.agent_theta = np.zeros(N,                          dtype=np.float32)
        self.object_pos  = np.zeros((N, num_objects, 2),        dtype=np.float32)
        self.object_held      = np.full(N, -1,                   dtype=np.int32)
        self.goal_pos         = np.zeros((N, num_objects, 2),   dtype=np.float32)
        self.delivered        = np.zeros((N, num_objects),      dtype=bool)
        self.step_count       = np.zeros(N,                     dtype=np.int32)
        self.next_pickup_idx  = np.zeros(N,                     dtype=np.int32)
        self.wrong_pickup     = np.zeros(N,                     dtype=bool)

        # Episode stats — needed so SB3 can report ep_rew_mean / ep_len_mean
        self._ep_return  = np.zeros(N,                          dtype=np.float32)

        # History buffers (only allocated when hist_length > 0)
        if hist_length > 0:
            self.obs_history = np.zeros(
                (N, self._window, self._base_obs_dim), dtype=np.float32
            )
            self.act_history = np.full(
                (N, self._window, act_dim), -1.0, dtype=np.float32
            )
            # Base obs at the previous step (pushed into history on next step)
            self._prev_base_obs = np.zeros((N, self._base_obs_dim), dtype=np.float32)

        # step_async buffer
        self._actions: Optional[np.ndarray] = None

        # SB3 ≥ 1.8
        self.reset_infos: List[Dict] = [{} for _ in range(N)]

        # Initialise all environments
        self._batch_reset(np.arange(N))

    # ------------------------------------------------------------------
    # Vectorised sampling helpers
    # ------------------------------------------------------------------

    def _sample_room_batch(self, room: str, n: int) -> np.ndarray:
        """Return (n, 2) float32 [x, y] positions inside *room* at the configured randomization level."""
        row_min, col_min, row_max, col_max = ROOM_BOUNDS[room]
        cx = (col_min + col_max + 1) / 2.0
        cy = (row_min + row_max + 1) / 2.0
        if self.randomization == 'low':
            return np.stack([
                np.full(n, cx, dtype=np.float32),
                np.full(n, cy, dtype=np.float32),
            ], axis=1)
        elif self.randomization == 'medium':
            half_w = (col_max - col_min + 1) / 4.0
            half_h = (row_max - row_min + 1) / 4.0
            x = self.np_random.uniform(cx - half_w, cx + half_w, size=n).astype(np.float32)
            y = self.np_random.uniform(cy - half_h, cy + half_h, size=n).astype(np.float32)
            return np.stack([x, y], axis=1)
        else:  # 'large'
            cells = ROOM_CELLS[room]
            idx = self.np_random.integers(len(cells), size=n)
            chosen = np.array(cells, dtype=np.float32)[idx]
            offsets = self.np_random.uniform(0.2, 0.8, size=(n, 2)).astype(np.float32)
            return np.stack([
                chosen[:, 1] + offsets[:, 0],   # x = ix + offset
                chosen[:, 0] + offsets[:, 1],   # y = iy + offset
            ], axis=1)

    # ------------------------------------------------------------------
    # Core reset
    # ------------------------------------------------------------------

    def _batch_reset(self, indices: np.ndarray):
        """Reset a subset of envs; returns their (history-augmented) observations."""
        n = len(indices)
        if n == 0:
            if self.image_obs:
                from envs.house_env import IMG_OBS_SIZE
                proprio_dim = self.observation_space['proprio'].shape[0]
                return {
                    'image':  np.empty((0, IMG_OBS_SIZE, IMG_OBS_SIZE, 3), dtype=np.uint8),
                    'proprio': np.empty((0, proprio_dim), dtype=np.float32),
                }
            return np.empty((0, self.observation_space.shape[0]), dtype=np.float32)

        if self._online_reset_seed is not None:
            self.np_random = np.random.default_rng(self._online_reset_seed)

        self.step_count[indices]      = 0
        self._ep_return[indices]      = 0.0
        self.object_held[indices]     = -1
        self.delivered[indices]       = False
        self.next_pickup_idx[indices] = 0
        self.wrong_pickup[indices]    = False

        self.agent_pos[indices]   = self._sample_room_batch('kitchen', n)
        self.agent_theta[indices] = self.np_random.uniform(
            -np.pi, np.pi, size=n
        ).astype(np.float32)

        src_rooms = ['living', 'bedroom', 'dining']
        for i in range(self.num_objects):
            self.object_pos[indices, i] = self._sample_room_batch(
                src_rooms[i % len(src_rooms)], n
            )
        for i in range(self.num_objects):
            self.goal_pos[indices, i] = self._sample_room_batch('kitchen', n)

        base_obs = self._compute_base_obs(indices)     # (n, base_obs_dim)

        if self.hist_length > 0:
            # Fill entire window with the initial observation; actions padded -1
            self.obs_history[indices] = base_obs[:, None, :].repeat(self._window, axis=1)
            self.act_history[indices] = -1.0
            self._prev_base_obs[indices] = base_obs
            return self._make_hist_obs(base_obs, indices)

        return base_obs

    # ------------------------------------------------------------------
    # Observation computation
    # ------------------------------------------------------------------

    def _fov_mask(self, indices: np.ndarray) -> np.ndarray:
        """Return boolean FOV mask (n, G, G) for the given env indices."""
        n = len(indices)
        if self.fully_observable:
            return np.ones((n, G, G), dtype=bool)
        pos   = self.agent_pos[indices]      # (n, 2)
        theta = self.agent_theta[indices]    # (n,)
        dx    = _CX[None] - pos[:, 0, None, None]    # (n, G, G)
        dy    = _CY[None] - pos[:, 1, None, None]
        r_sq  = dx * dx + dy * dy
        angle = np.arctan2(dy, dx)
        diff  = (angle - theta[:, None, None] + np.pi) % (2 * np.pi) - np.pi
        return (r_sq <= self.fov_radius ** 2) & (np.abs(diff) <= self.half_fov)

    def _render_obs_images(self, indices: np.ndarray) -> np.ndarray:
        """Render (n, IMG_OBS_SIZE, IMG_OBS_SIZE, 3) uint8 images for env *indices*."""
        from envs.house_env import make_house_obs_image, IMG_OBS_SIZE
        n = len(indices)
        imgs = np.empty((n, IMG_OBS_SIZE, IMG_OBS_SIZE, 3), dtype=np.uint8)
        for k, idx in enumerate(indices):
            imgs[k] = make_house_obs_image(
                self.agent_pos[idx], self.agent_theta[idx],
                self.object_pos[idx], int(self.object_held[idx]),
                self.delivered[idx], self.goal_pos[idx],
                self.num_objects,
                fov_radius=self.fov_radius,
                fov_angle=self.fov_angle,
                fully_observable=self.fully_observable,
            )
        return imgs

    def _compute_proprio_batch(self, indices: np.ndarray) -> np.ndarray:
        """Compute proprioceptive obs (n, 4): agent position and orientation only."""
        pos   = self.agent_pos[indices]
        theta = self.agent_theta[indices]
        return np.stack([
            pos[:, 0] / G, pos[:, 1] / G,
            np.sin(theta), np.cos(theta),
        ], axis=1).astype(np.float32)

    def _compute_base_obs(self, indices: np.ndarray):
        """Compute raw (no history) observations for the given env indices."""
        if self.image_obs:
            return {
                'image':  self._render_obs_images(indices),
                'proprio': self._compute_proprio_batch(indices),
            }

        n     = len(indices)
        pos   = self.agent_pos[indices]      # (n, 2)
        theta = self.agent_theta[indices]    # (n,)
        opos  = self.object_pos[indices]     # (n, num_objects, 2)
        gpos  = self.goal_pos[indices]       # (n, num_objects, 2)
        deliv = self.delivered[indices]      # (n, num_objects)
        held  = self.object_held[indices]    # (n,)

        # Pose (n, 4)
        pose = np.stack([
            pos[:, 0] / G, pos[:, 1] / G,
            np.sin(theta), np.cos(theta),
        ], axis=1).astype(np.float32)

        # FOV mask (n, G, G)
        in_fov = self._fov_mask(indices)

        if self.small_obs:
            # Compact vis: (n, 2*num_objects) — [x/G, y/G] if visible, [-1,-1] if not.
            # When fully_observable, skip FOV check and always provide positions.
            vis = np.full((n, 2 * self.num_objects), -1.0, dtype=np.float32)
            for i in range(self.num_objects):
                oi_y = np.clip(opos[:, i, 1].astype(int), 0, G - 1)
                oi_x = np.clip(opos[:, i, 0].astype(int), 0, G - 1)
                vis_cell = in_fov[np.arange(n), oi_y, oi_x] if not self.fully_observable else np.ones(n, dtype=bool)
                active   = ~deliv[:, i] & (held != i)
                mark     = vis_cell & active
                vis[mark, 2 * i]     = opos[mark, i, 0] / G
                vis[mark, 2 * i + 1] = opos[mark, i, 1] / G
        else:
            # Full grid: (n, G²): -1=outside FOV, 0=empty, 1..N=object id
            vis = np.where(in_fov, 0.0, -1.0).reshape(n, G * G).astype(np.float32)
            for i in range(self.num_objects):
                oi_y = np.clip(opos[:, i, 1].astype(int), 0, G - 1)
                oi_x = np.clip(opos[:, i, 0].astype(int), 0, G - 1)
                flat = oi_y * G + oi_x
                vis_cell = in_fov[np.arange(n), oi_y, oi_x]
                active   = ~deliv[:, i] & (held != i)
                mark     = vis_cell & active
                vis[mark, flat[mark]] = float(i + 1)

        # Goal: (n, 2*num_objects) normalized XY, always shown for all objects
        goal = np.zeros((n, 2 * self.num_objects), dtype=np.float32)
        for i in range(self.num_objects):
            goal[:, 2 * i]     = (gpos[:, i, 0] - GOAL_ROOM_X_MIN) / (GOAL_ROOM_X_MAX - GOAL_ROOM_X_MIN)
            goal[:, 2 * i + 1] = (gpos[:, i, 1] - GOAL_ROOM_Y_MIN) / (GOAL_ROOM_Y_MAX - GOAL_ROOM_Y_MIN)

        # Inventory one-hot (n, num_objects+1)
        inv = np.zeros((n, self.num_objects + 1), dtype=np.float32)
        inv[held == -1, 0] = 1.0
        for i in range(self.num_objects):
            inv[held == i, i + 1] = 1.0

        return np.concatenate([pose, vis, goal, inv], axis=1)

    def _make_hist_obs(
        self, cur_base_obs: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Interleave rolling history with current obs → flat history obs."""
        # history sampled at stride: indices [0, stride, 2*stride, ...]
        hist_o = self.obs_history[indices][:, ::self.hist_stride, :]  # (n, hl, base)
        hist_a = self.act_history[indices][:, ::self.hist_stride, :]  # (n, hl, 4)
        parts = []
        for t in range(self.hist_length):
            parts.append(hist_o[:, t, :])
            parts.append(hist_a[:, t, :])
        parts.append(cur_base_obs)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _compute_dense_reward_batch(self, prev_held: np.ndarray) -> np.ndarray:
        """Vectorised dense reward (N,):
          - Not holding: (1 - dist_agent→nearest_obj / max_dist)
                       + (1 - dist_nearest_obj→its_goal / max_dist)
          - Pickup event (prev_held==-1, now holding): +1 bonus
          - Holding:     (1 - dist_agent→goal / max_dist)
                       + (1 - dist_held_obj→goal / max_dist)
        """
        N       = self.num_envs
        held    = self.object_held    # (N,)
        rewards = np.zeros(N, dtype=np.float32)

        # Pickup bonus
        rewards[(prev_held == -1) & (held >= 0)] += 1.0

        # +2 per already-delivered object
        rewards += 2.0 * np.sum(self.delivered, axis=1)

        # Helper: vectorised BFS distance lookup for (N,2) float positions → (N,) distances
        def _bfs_dist_vec(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
            """pos_a, pos_b: (k, 2) float [x, y]; returns (k,) BFS distances."""
            iy_a = np.clip(pos_a[:, 1].astype(np.int32), 0, G - 1)
            ix_a = np.clip(pos_a[:, 0].astype(np.int32), 0, G - 1)
            iy_b = np.clip(pos_b[:, 1].astype(np.int32), 0, G - 1)
            ix_b = np.clip(pos_b[:, 0].astype(np.int32), 0, G - 1)
            return GRID_DIST[iy_a * G + ix_a, iy_b * G + ix_b]

        # Envs holding an object: agent→goal + held_obj→goal (same distance)
        holding = held >= 0
        if np.any(holding):
            idx      = np.where(holding)[0]
            held_obj = held[idx]
            dist_to_goal = _bfs_dist_vec(
                self.agent_pos[idx], self.goal_pos[idx, held_obj]
            )
            rewards[idx] += 2.0 * (1.0 - dist_to_goal / self._max_dist)

        # Envs not holding: agent→nearest_obj + nearest_obj→its_goal
        not_holding = ~holding
        if np.any(not_holding):
            idx   = np.where(not_holding)[0]
            agent = self.agent_pos[idx]                     # (k, 2)

            # Find nearest undelivered object and its goal for each env
            min_agent_obj = np.full(len(idx), np.inf, dtype=np.float32)
            nearest_obj_pos  = np.zeros((len(idx), 2), dtype=np.float32)
            nearest_goal_pos = np.zeros((len(idx), 2), dtype=np.float32)

            for i in range(self.num_objects):
                active = ~self.delivered[idx, i]
                if not np.any(active):
                    continue
                d = _bfs_dist_vec(agent, self.object_pos[idx, i])
                closer = active & (d < min_agent_obj)
                min_agent_obj = np.where(closer, d, min_agent_obj)
                nearest_obj_pos[closer]  = self.object_pos[idx[closer], i]
                nearest_goal_pos[closer] = self.goal_pos[idx[closer], i]

            has_target = min_agent_obj < np.inf
            if np.any(has_target):
                ht = idx[has_target]
                obj_goal_dist = _bfs_dist_vec(
                    nearest_obj_pos[has_target], nearest_goal_pos[has_target]
                )
                rewards[ht] += (1.0 - min_agent_obj[has_target] / self._max_dist)
                rewards[ht] += (1.0 - obj_goal_dist / self._max_dist)

        return rewards

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> VecEnvObs:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        obs = self._batch_reset(np.arange(self.num_envs))
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return obs

    def step(self, actions: np.ndarray):
        """Convenience wrapper: step_async + step_wait in one call."""
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        N       = self.num_envs
        actions = np.clip(self._actions, -1.0, 1.0)   # (N, 4)
        dx      = actions[:, 0]
        dy      = actions[:, 1]
        dtheta  = actions[:, 2]
        hold    = actions[:, 3]

        # ── Rotate ──────────────────────────────────────────────────────
        self.agent_theta = (
            self.agent_theta + dtheta * self.max_rotation + np.pi
        ) % (2 * np.pi) - np.pi

        # ── Translate (unit-circle clip) ─────────────────────────────────
        mag   = np.sqrt(dx * dx + dy * dy)
        scale = np.where(mag > 1.0, 1.0 / np.maximum(mag, 1e-8), 1.0)
        dx    = dx * scale
        dy    = dy * scale

        new_x = np.clip(self.agent_pos[:, 0] + dx, 0.05, G - 0.05)
        new_y = np.clip(self.agent_pos[:, 1] + dy, 0.05, G - 0.05)
        cur_x = self.agent_pos[:, 0]
        cur_y = self.agent_pos[:, 1]

        def _passable(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return PASSABLE[np.clip(y.astype(int), 0, G-1),
                            np.clip(x.astype(int), 0, G-1)]

        p_both   = _passable(new_x, new_y)
        p_x_only = _passable(new_x, cur_y)
        p_y_only = _passable(cur_x, new_y)

        self.agent_pos[:, 0] = np.where(p_both, new_x, np.where(p_x_only, new_x, cur_x))
        self.agent_pos[:, 1] = np.where(p_both, new_y,
                               np.where(p_x_only, cur_y,
                               np.where(p_y_only, new_y, cur_y)))

        # ── Carried objects follow agent ─────────────────────────────────
        for i in range(self.num_objects):
            carrying = (self.object_held == i)
            if np.any(carrying):
                self.object_pos[carrying, i] = self.agent_pos[carrying]

        # ── Pickup ───────────────────────────────────────────────────────
        prev_delivered = np.sum(self.delivered, axis=1)   # (N,)
        prev_held      = self.object_held.copy()           # (N,) saved for dense reward
        can_pickup = (hold > 0.0) & (self.object_held == -1)
        if np.any(can_pickup):
            for i in range(self.num_objects):
                dist = np.linalg.norm(self.agent_pos - self.object_pos[:, i], axis=1)
                do_pickup = (
                    can_pickup
                    & (self.object_held == -1)
                    & ~self.delivered[:, i]
                    & (dist < self.pickup_radius)
                )
                self.object_held[do_pickup] = i

        # ── Drop ─────────────────────────────────────────────────────────
        do_drop = (hold < 0.0) & (self.object_held >= 0)
        if np.any(do_drop):
            for i in range(self.num_objects):
                dropping_i = do_drop & (self.object_held == i)
                if not np.any(dropping_i):
                    continue
                idx_i = np.where(dropping_i)[0]
                self.object_pos[dropping_i, i] = self.agent_pos[dropping_i]
                self.object_held[dropping_i]   = -1
                dist_goal = np.linalg.norm(
                    self.agent_pos[dropping_i] - self.goal_pos[dropping_i, i], axis=1
                )
                at_goal = idx_i[dist_goal < self.drop_goal_radius]
                self.delivered[at_goal, i]   = True
                self.object_pos[at_goal, i]  = self.goal_pos[at_goal, i]

        # ── Delivery-order enforcement ────────────────────────────────────
        # Detect envs that just picked up an object.
        just_picked = (prev_held == -1) & (self.object_held >= 0)
        if np.any(just_picked):
            for n_idx in np.where(just_picked)[0]:
                pidx = int(self.next_pickup_idx[n_idx])
                required = DELIVERY_ORDER[pidx] if pidx < len(DELIVERY_ORDER) else -1
                if int(self.object_held[n_idx]) != required:
                    self.wrong_pickup[n_idx] = True
                else:
                    self.next_pickup_idx[n_idx] += 1

        # ── Rewards ──────────────────────────────────────────────────────
        cur_delivered = np.sum(self.delivered, axis=1)
        if self.dense_reward:
            rewards = self._compute_dense_reward_batch(prev_held)
        elif self.count_reward:
            rewards = (cur_delivered - self.num_objects).astype(np.float32)
        else:
            rewards = (cur_delivered - prev_delivered).astype(np.float32)
        rewards[self.wrong_pickup] = -5.0
        self._ep_return += rewards

        # ── Termination ──────────────────────────────────────────────────
        self.step_count += 1
        terminated = np.all(self.delivered, axis=1)
        truncated  = (self.step_count >= self.max_steps)
        dones      = terminated | truncated

        # ── Observations ─────────────────────────────────────────────────
        all_idx    = np.arange(N)
        base_obs   = self._compute_base_obs(all_idx)   # (N, base_obs_dim)

        if self.hist_length > 0:
            # Save the pre-step base obs into history, then advance pointer
            prev_base = self._prev_base_obs              # (N, base_obs_dim)
            self.obs_history[:, :-1] = self.obs_history[:, 1:]
            self.obs_history[:, -1]  = prev_base
            self.act_history[:, :-1] = self.act_history[:, 1:]
            self.act_history[:, -1]  = actions
            self._prev_base_obs      = base_obs
            obs = self._make_hist_obs(base_obs, all_idx)
        else:
            obs = base_obs

        # ── Infos ────────────────────────────────────────────────────────
        infos: List[Dict[str, Any]] = []
        for n in range(N):
            info: Dict[str, Any] = {
                'objects_delivered': int(np.sum(self.delivered[n])),
                'total_objects':     self.num_objects,
                'success':           bool(terminated[n]),
                'wrong_pickup':      bool(self.wrong_pickup[n]),
            }
            if dones[n]:
                info['episode'] = {
                    'return':       float(self._ep_return[n]),
                    'length':       int(self.step_count[n]),
                    'final_reward': float(rewards[n]),
                    # SB3 aliases
                    'r': float(self._ep_return[n]),
                    'l': int(self.step_count[n]),
                }
            infos.append(info)

        # ── Auto-reset done envs (SB3 convention) ────────────────────────
        done_idx = np.where(dones)[0]
        if len(done_idx) > 0:
            if self.image_obs:
                for n in done_idx:
                    infos[n]['terminal_observation'] = {k: v[n].copy() for k, v in obs.items()}
                reset_obs = self._batch_reset(done_idx)
                for k in obs:
                    obs[k][done_idx] = reset_obs[k]
            else:
                for n in done_idx:
                    infos[n]['terminal_observation'] = obs[n].copy()
                reset_obs = self._batch_reset(done_idx)
                obs[done_idx] = reset_obs

        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    # ── SB3 VecEnv ABC requirements ──────────────────────────────────────

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
        fn = getattr(self, method_name)
        return [fn(*args, **kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: Type, indices=None) -> List[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> List[np.ndarray]:
        from envs.house_env import HouseEnv
        imgs = []
        for n in range(self.num_envs):
            env = HouseEnv(num_objects=self.num_objects)
            env.agent_pos   = self.agent_pos[n].copy()
            env.agent_theta = float(self.agent_theta[n])
            env.object_pos  = self.object_pos[n].copy()
            env.object_held = int(self.object_held[n])
            env.goal_pos    = self.goal_pos[n].copy()
            env.delivered   = self.delivered[n].copy()
            env.step_count  = int(self.step_count[n])
            imgs.append(env.render())
        return imgs

    def _get_indices(self, indices) -> List[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_batched_house_env(
    env_name:          str,
    num_envs:          int  = 4,
    count_reward:      bool = False,
    dense_reward:      bool = False,
    fully_observable:  bool = False,
    small_obs:         bool = False,
    hist_length:       int  = 0,
    hist_stride:       int  = 1,
    seed:              int  = 0,
    max_steps:         int  = 100,
    online_reset_seed: int | None = None,
    randomization:     str  = 'large',
    image_obs:         bool = False,
) -> BatchedHouseEnv:
    """Create a BatchedHouseEnv from an env_name like 'house-n3-v0'."""
    num_objects = _parse_num_objects(env_name)
    return BatchedHouseEnv(
        num_envs=num_envs,
        num_objects=num_objects,
        max_steps=max_steps,
        hist_length=hist_length,
        hist_stride=hist_stride,
        count_reward=count_reward,
        dense_reward=dense_reward,
        fully_observable=fully_observable,
        small_obs=small_obs,
        seed=seed,
        online_reset_seed=online_reset_seed,
        randomization=randomization,
        image_obs=image_obs,
    )
