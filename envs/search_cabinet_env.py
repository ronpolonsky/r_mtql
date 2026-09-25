"""
ManiSkill cabinet-search environment wrapper + offline-RL dataset loader.

The dataset was collected by collect_cabinet_dataset.py and stored as NPZ files
in cabinet-memory-sim/ManiSkill/cabinet_dataset/.  This module:

  1. Defines SearchCabinetManiSkillEnv — a gymnasium wrapper around the
     ManiSkill OpenCabinetPanda-v0 environment.  Actions are 8-dim normalised
     joint positions (same format as the offline dataset); observations are
     (H, W, 3) uint8 RGB images from render_rgb_array().

     ManiSkill already vectorises internally (batch dimension = 1 for now).
     make_vec_eval_env raises ValueError if num_envs > 1.

  2. Provides load_search_cabinet_datasets() and
     make_search_cabinet_env_and_datasets() for use inside env_utils.py.

Dataset NPZ keys (train_cabinet_dataset.npz / val_cabinet_dataset.npz):
    obs_image        (N, H, W, 3)  uint8   — RGB observation
    obs_proprio      (N, 9)        float32 — 7 arm joints + 2 gripper fingers
    next_obs_image   (N, H, W, 3)  uint8
    next_obs_proprio (N, 9)        float32
    actions          (N, 8)        float32 — normalised to [-1, 1]
    rewards          (N,)          float32
    terminals        (N,)          float32
    masks            (N,)          float32
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Dict as DictSpace

# ---------------------------------------------------------------------------
# Path to dataset (relative to this file → project root → data dir)
# ---------------------------------------------------------------------------
_DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cabinet-memory-sim', 'ManiSkill', 'cabinet_dataset_noisy',
)
_TWO_CAMS_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cabinet-memory-sim', 'ManiSkill', 'cabinet_dataset_two_cams',
)
_TWO_CAMS_MORE_RAND_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cabinet-memory-sim', 'ManiSkill', 'cabinet_dataset_two_cams_more_rand',
)
_TWO_CAMS_MORE_RAND_HARD_BC_S25_F15_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cabinet-memory-sim', 'ManiSkill',
    'cabinet_dataset_two_cams_more_rand_hard_bc_s25_f15',
)

# ---------------------------------------------------------------------------
# Action normalisation constants (must match cabinet_dataset_collector.py)
# ---------------------------------------------------------------------------
_ARM_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                      dtype=np.float32)
_ARM_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973],
                      dtype=np.float32)
_GRIPPER_LOWER = np.float32(-1.0)
_GRIPPER_UPPER = np.float32(1.0)

_ACTION_LOWER = np.concatenate([_ARM_LOWER, [_GRIPPER_LOWER]])
_ACTION_UPPER = np.concatenate([_ARM_UPPER, [_GRIPPER_UPPER]])


def _denorm_action(action: np.ndarray) -> np.ndarray:
    """Map 8-dim normalised action [-1, 1] → batched (1, 8) raw pd_joint_pos action.

    The normalisation in the collector is:
        normed = 2 * (raw - lower) / (upper - lower + 1e-8) - 1
    so the inverse is:
        raw = (normed + 1) / 2 * (upper - lower + 1e-8) + lower

    ManiSkill batches actions over num_envs, so we add the leading 1 dimension.
    """
    a = np.clip(np.asarray(action, dtype=np.float32).ravel(), -1.0, 1.0)
    raw = (a + 1.0) / 2.0 * (_ACTION_UPPER - _ACTION_LOWER + 1e-8) + _ACTION_LOWER
    return raw[np.newaxis, :]  # (1, 8)


def _scalar(x):
    """Extract a Python scalar from a numpy array, tensor, or plain number."""
    if hasattr(x, 'item'):
        return x.item()
    if hasattr(x, '__len__'):
        return type(x.flat[0])(x.flat[0])
    return x


# ---------------------------------------------------------------------------
# Real ManiSkill gymnasium wrapper
# ---------------------------------------------------------------------------

class SearchCabinetManiSkillEnv(gymnasium.Env):
    """Gymnasium wrapper around ManiSkill OpenCabinetPanda-v0.

    Accepts 8-dim normalised actions (matching the offline dataset) and returns
    (H, W, 3) uint8 image observations.  The underlying ManiSkill env is
    batched (num_envs=1); batch dimensions are squeezed away here so the
    interface matches the standard gymnasium single-env API.

    Only a single parallel environment is supported.  To create multiple
    copies for vectorised evaluation, use make_vec_eval_env — which will
    raise a ValueError if num_envs > 1 for this environment.
    """
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, seed: int = 0, color_shuffle_seed: Optional[int] = None,
                 image_size: int = 128, randomize_cabinet_pose: bool = False):
        super().__init__()
        self._seed = seed
        self._color_shuffle_seed = color_shuffle_seed
        self._image_size = image_size
        self._randomize_cabinet_pose = randomize_cabinet_pose
        self._ms_env = None
        self._build_ms_env()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _build_ms_env(self):
        import sys as _sys
        _ms_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'cabinet-memory-sim', 'ManiSkill',
        )
        if _ms_path not in _sys.path:
            _sys.path.insert(0, _ms_path)
        import mani_skill.envs  # noqa: F401 — registers ManiSkill envs with gymnasium
        import gymnasium as _gym
        self._ms_env = _gym.make(
            'OpenCabinetPanda-v0',
            render_mode='rgb_array',
            control_mode='pd_joint_pos',
            robot_uids='panda',
            shuffle_colors=True,
            color_shuffle_seed=self._color_shuffle_seed,
            num_cubes=1,
            max_episode_steps=1000,
            render_backend='gpu',
            width=self._image_size,
            height=self._image_size,
            randomize_cabinet_pose=self._randomize_cabinet_pose,
        )
        # Reset once to determine observation shape.
        self._ms_env.reset(seed=self._seed)
        img   = self._ms_env.unwrapped.render_rgb_array().cpu().numpy()[0]
        qpos  = self._ms_env.unwrapped.agent.robot.get_qpos().cpu().numpy().ravel()
        obs_shape    = img.shape    # (H, W, 3)
        proprio_dim  = qpos.shape[0]  # 9

        self.observation_space = DictSpace({
            'agent_view':  Box(low=0,    high=255, shape=obs_shape,   dtype=np.uint8),
            'wrist_view':  Box(low=0,    high=255, shape=obs_shape,   dtype=np.uint8),
            'proprio': Box(low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32),
        })
        self.action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)


    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _get_obs(self):
        env = self._ms_env.unwrapped
        img       = env.render_rgb_array('render_camera').cpu().numpy()[0].astype(np.uint8)
        wrist_img = env.render_rgb_array('wrist_camera').cpu().numpy()[0].astype(np.uint8)
        qpos = env.agent.robot.get_qpos().cpu().numpy().ravel().astype(np.float32)
        return {'agent_view': img, 'wrist_view': wrist_img, 'proprio': qpos}

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self._seed = seed
        self._ms_env.reset(seed=self._seed)
        return self._get_obs(), {}

    def step(self, action):
        raw = _denorm_action(action)
        _, reward, terminated, truncated, info = self._ms_env.step(raw)

        obs = self._get_obs()

        reward     = float(_scalar(reward))
        terminated = bool(_scalar(terminated))
        truncated  = bool(_scalar(truncated))

        # ManiSkill stores success as a (1,) tensor in info.
        success = bool(_scalar(info.get('success', False)))

        # Convert any remaining tensor values in info to Python scalars so
        # EpisodeMonitor / logging code doesn't receive raw tensors.
        clean_info = {}
        for k, v in info.items():
            try:
                clean_info[k] = _scalar(v)
            except Exception:
                clean_info[k] = v

        # Always ensure 'success' is present so eval stats never get NaN from
        # an empty list (ManiSkill may not always include this key).
        clean_info['success'] = success

        return obs, reward, terminated, truncated, clean_info

    def render(self):
        return self._ms_env.unwrapped.render_rgb_array().cpu().numpy()[0].astype(np.uint8)

    def close(self):
        if self._ms_env is not None:
            self._ms_env.close()
            self._ms_env = None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _keep_successful_episodes(data: dict, split_name: str) -> dict:
    """Return complete episodes containing a positive reward.

    This only indexes the arrays already loaded from the merged NPZ; it never
    modifies the source dataset on disk.  Cabinet reconstruction makes the
    first positive-reward transition the final transition of a successful
    episode, but checking the whole episode keeps the filter explicit.
    """
    rewards = np.asarray(data['rewards'])
    terminals = np.asarray(data['terminals'])
    terminal_idxs = np.flatnonzero(terminals > 0)
    if not len(terminal_idxs) or terminal_idxs[-1] != len(rewards) - 1:
        raise ValueError(
            f'{split_name} cabinet dataset does not contain complete terminal-delimited episodes'
        )

    episode_starts = np.concatenate(([0], terminal_idxs[:-1] + 1))
    kept_ranges = [
        np.arange(start, end + 1, dtype=np.int64)
        for start, end in zip(episode_starts, terminal_idxs)
        if np.any(rewards[start:end + 1] > 0)
    ]
    if not kept_ranges:
        raise ValueError(f'{split_name} cabinet dataset contains no successful episodes')

    keep_idxs = np.concatenate(kept_ranges)

    def take(value):
        if isinstance(value, dict):
            return {key: take(child) for key, child in value.items()}
        return value[keep_idxs]

    filtered = {key: take(value) for key, value in data.items()}
    print(
        f'[load_search_cabinet_datasets] {split_name}: retained '
        f'{len(kept_ranges)}/{len(terminal_idxs)} successful episodes '
        f'({len(keep_idxs)}/{len(rewards)} transitions); source files unchanged'
    )
    return filtered


def _select_episode_mix(
    data: dict,
    num_success_demos: int,
    num_failure_demos: int,
    split_name: str,
) -> dict:
    """Keep requested counts of complete success and failure episodes.

    Episodes are selected from their existing deterministic merged order, and
    the selected episodes retain that order. A value of -1 keeps every episode
    in that outcome class; zero excludes that class.
    """
    if num_success_demos < -1 or num_failure_demos < -1:
        raise ValueError('num_success_demos and num_failure_demos must be >= -1')

    rewards = np.asarray(data['rewards'])
    terminals = np.asarray(data['terminals'])
    terminal_idxs = np.flatnonzero(terminals > 0)
    if not len(terminal_idxs) or terminal_idxs[-1] != len(rewards) - 1:
        raise ValueError(
            f'{split_name} cabinet dataset does not contain complete terminal-delimited episodes'
        )

    episode_starts = np.concatenate(([0], terminal_idxs[:-1] + 1))
    episode_ranges = [
        np.arange(start, end + 1, dtype=np.int64)
        for start, end in zip(episode_starts, terminal_idxs)
    ]
    episode_success = np.asarray([
        np.any(rewards[start:end + 1] > 0)
        for start, end in zip(episode_starts, terminal_idxs)
    ])
    success_episode_idxs = np.flatnonzero(episode_success)
    failure_episode_idxs = np.flatnonzero(~episode_success)

    requested_successes = (
        len(success_episode_idxs) if num_success_demos == -1 else num_success_demos
    )
    requested_failures = (
        len(failure_episode_idxs) if num_failure_demos == -1 else num_failure_demos
    )
    if requested_successes > len(success_episode_idxs):
        raise ValueError(
            f'{split_name} requested {requested_successes} successful demos, but only '
            f'{len(success_episode_idxs)} are available'
        )
    if requested_failures > len(failure_episode_idxs):
        raise ValueError(
            f'{split_name} requested {requested_failures} failed demos, but only '
            f'{len(failure_episode_idxs)} are available'
        )

    selected_episode_idxs = set(success_episode_idxs[:requested_successes])
    selected_episode_idxs.update(failure_episode_idxs[:requested_failures])
    if not selected_episode_idxs:
        raise ValueError(f'{split_name} episode mix selected zero demonstrations')

    keep_idxs = np.concatenate([
        episode_ranges[index]
        for index in range(len(episode_ranges))
        if index in selected_episode_idxs
    ])

    def take(value):
        if isinstance(value, dict):
            return {key: take(child) for key, child in value.items()}
        return value[keep_idxs]

    filtered = {key: take(value) for key, value in data.items()}
    print(
        f'[load_search_cabinet_datasets] {split_name}: selected '
        f'{requested_successes} successful + {requested_failures} failed episodes '
        f'({len(keep_idxs)}/{len(rewards)} transitions); source files unchanged'
    )
    return filtered


def _relabel_terminal_sparse_rewards(data: dict, split_name: str) -> dict:
    """Replace the per-step time penalty with terminal outcome rewards.

    The cabinet simulator emits -1 at every unsuccessful step and +1 at the
    successful terminal.  For long search episodes that can make a short
    failure have a larger return than a later success.  This optional offline
    relabeling preserves every transition and boundary while using 0 on
    nonterminal steps, +1 on successful terminals, and -1 on failed terminals.
    """
    rewards = np.asarray(data['rewards'])
    terminals = np.asarray(data['terminals'])
    terminal_idxs = np.flatnonzero(terminals > 0)
    if not len(terminal_idxs) or terminal_idxs[-1] != len(rewards) - 1:
        raise ValueError(
            f'{split_name} cabinet dataset does not contain complete terminal-delimited episodes'
        )

    terminal_outcomes = np.where(rewards[terminal_idxs] > 0, 1.0, -1.0)
    relabeled = np.zeros_like(rewards, dtype=np.float32)
    relabeled[terminal_idxs] = terminal_outcomes.astype(np.float32)
    output = dict(data)
    output['rewards'] = relabeled
    print(
        f'[load_search_cabinet_datasets] {split_name}: terminal-sparse reward '
        f'relabeling ({np.count_nonzero(terminal_outcomes > 0)} successful, '
        f'{np.count_nonzero(terminal_outcomes < 0)} failed terminals); source files unchanged'
    )
    return output


def _relabel_outcome_time_rewards(
    data: dict, split_name: str, step_cost: float = 0.001
) -> dict:
    """Reward successful outcomes, penalize failures, and rank efficiency.

    Every nonterminal transition receives a small time cost. Terminal rewards
    are +1 for success and -1 for failure. This preserves a clear outcome gap
    for all selected episode lengths while allowing the critic to prefer less
    repetitive successful behavior that behavior cloning must imitate.
    """
    if step_cost < 0:
        raise ValueError(f'step_cost must be nonnegative, got {step_cost}')
    rewards = np.asarray(data['rewards'])
    terminals = np.asarray(data['terminals'])
    terminal_idxs = np.flatnonzero(terminals > 0)
    if not len(terminal_idxs) or terminal_idxs[-1] != len(rewards) - 1:
        raise ValueError(
            f'{split_name} cabinet dataset does not contain complete terminal-delimited episodes'
        )

    terminal_outcomes = np.where(rewards[terminal_idxs] > 0, 1.0, -1.0)
    relabeled = np.full_like(rewards, -step_cost, dtype=np.float32)
    relabeled[terminal_idxs] = terminal_outcomes.astype(np.float32)
    output = dict(data)
    output['rewards'] = relabeled
    print(
        f'[load_search_cabinet_datasets] {split_name}: outcome+time reward '
        f'relabeling (step_cost={step_cost}, '
        f'{np.count_nonzero(terminal_outcomes > 0)} successful, '
        f'{np.count_nonzero(terminal_outcomes < 0)} failed terminals); '
        f'source files unchanged'
    )
    return output

def load_search_cabinet_datasets(
    dataset_dir: str = _DEFAULT_DATASET_DIR,
    hist_length: int = 4,
    hist_stride: int = 1,
    online_buf_size: int = 20_000,
    ep_cache_size: int = 40,
    chunk_reload_interval: int = 1000,
    action_chunk_size: int = 1,
    discount: float = 1.0,
    lazy: bool = True,
    max_demos: int = 0,
    successful_demos_only: bool = False,
    num_success_demos: int = -1,
    num_failure_demos: int = -1,
    terminal_sparse_rewards: bool = False,
    outcome_time_rewards: bool = False,
):
    """Load train/val datasets for the cabinet-search task.

    Train dataset: LazyEpisodeReplayBuffer (lazy=True, default) or
        HistoryReplayBuffer (lazy=False) that loads everything into RAM.
        Both support add_transition() for single-step-online RL.

    Val dataset: HistoryDataset loaded from the merged val NPZ (small enough
        to fit in RAM and not used for online collection).

    Args:
        dataset_dir:     Directory containing train_cabinet_dataset.npz,
                         val_cabinet_dataset.npz, and episodes/.
        hist_length:     History window length.
        hist_stride:     Stride between history steps.
        online_buf_size: Max online transitions kept in the train buffer RAM.
        lazy:            If True use LazyEpisodeReplayBuffer; if False load
                         the merged train NPZ fully into a HistoryReplayBuffer.

    Returns:
        (train_dataset, val_dataset)
    """
    import glob as _glob
    from utils.datasets import HistoryDataset, HistoryReplayBuffer, LazyEpisodeReplayBuffer

    mix_requested = num_success_demos >= 0 or num_failure_demos >= 0
    if successful_demos_only and mix_requested:
        raise ValueError(
            'successful_demos_only cannot be combined with num_success_demos '
            'or num_failure_demos'
        )
    if max_demos > 0 and mix_requested:
        raise ValueError(
            'max_demos cannot be combined with num_success_demos or num_failure_demos'
        )
    if terminal_sparse_rewards and outcome_time_rewards:
        raise ValueError(
            'terminal_sparse_rewards and outcome_time_rewards are mutually exclusive'
        )

    if lazy:
        if (successful_demos_only or mix_requested or terminal_sparse_rewards
                or outcome_time_rewards):
            raise ValueError(
                'cabinet outcome filtering/relabeling requires the merged train split; '
                'use --nolazy_dataset'
            )
        # Train: lazy episode-based buffer
        ep_dir = os.path.join(dataset_dir, 'episodes')
        episode_paths = sorted(_glob.glob(os.path.join(ep_dir, 'ep_*.npz')))
        if not episode_paths:
            raise FileNotFoundError(f'No episode NPZ files found in {ep_dir}')

        train_dataset = LazyEpisodeReplayBuffer(
            episode_paths=episode_paths,
            hist_length=hist_length,
            hist_stride=hist_stride,
            online_buf_size=online_buf_size,
            ep_cache_size=ep_cache_size,
            chunk_reload_interval=chunk_reload_interval,
            action_chunk_size=action_chunk_size,
            discount=discount,
        )
    else:
        # Train: load merged NPZ fully into RAM as a HistoryReplayBuffer
        train_path = os.path.join(dataset_dir, 'train_cabinet_dataset.npz')
        with np.load(train_path) as d:
            train_data = dict(
                observations      = {'agent_view': d['obs_image'].copy(),
                                     'wrist_view': d['obs_wrist_image'].copy(),
                                     'proprio':    d['obs_proprio'].astype(np.float32)},
                actions           = d['actions'].astype(np.float32),
                rewards           = d['rewards'].astype(np.float32),
                terminals         = d['terminals'].astype(np.float32),
                masks             = d['masks'].astype(np.float32),
                # Optional metadata used by mtql_transformer_v2 to exclude
                # failed rollouts from its behavior losses.  The committed
                # base mtql_transformer intentionally ignores this field.
                actor_mask        = (
                    d['actor_mask'].astype(np.float32)
                    if 'actor_mask' in d.files
                    else np.ones_like(d['rewards'], dtype=np.float32)
                ),
            )
        if successful_demos_only:
            train_data = _keep_successful_episodes(train_data, 'train')
        elif mix_requested:
            train_data = _select_episode_mix(
                train_data,
                num_success_demos=num_success_demos,
                num_failure_demos=num_failure_demos,
                split_name='train',
            )
        if terminal_sparse_rewards:
            train_data = _relabel_terminal_sparse_rewards(train_data, 'train')
        if outcome_time_rewards:
            train_data = _relabel_outcome_time_rewards(train_data, 'train')
        # Truncate to max_demos complete episodes if requested.
        if max_demos > 0:
            terminal_idxs = np.nonzero(train_data['terminals'] > 0)[0]
            if len(terminal_idxs) > max_demos:
                cutoff = int(terminal_idxs[max_demos - 1]) + 1
                train_data = {
                    k: (v[:cutoff] if not isinstance(v, dict)
                         else {kk: vv[:cutoff] for kk, vv in v.items()})
                    for k, v in train_data.items()
                }
                print(f'[load_search_cabinet_datasets] Truncated to {max_demos} demos '
                      f'({cutoff} transitions)')
        init_dataset = HistoryDataset.create(
            hist_length=hist_length, hist_stride=hist_stride, **train_data,
        )
        train_dataset = HistoryReplayBuffer.create_from_initial_dataset(
            init_dataset._dict,
            size=len(train_data['rewards']) + online_buf_size,
            hist_length=hist_length,
            hist_stride=hist_stride,
        )

    # Val: small enough to load fully (used only for loss logging, not online RL)
    val_path = os.path.join(dataset_dir, 'val_cabinet_dataset.npz')
    with np.load(val_path) as d:
        val_data = dict(
            observations      = {'agent_view': d['obs_image'].copy(),
                                 'wrist_view': d['obs_wrist_image'].copy(),
                                 'proprio':    d['obs_proprio'].astype(np.float32)},
            actions           = d['actions'].astype(np.float32),
            rewards           = d['rewards'].astype(np.float32),
            terminals         = d['terminals'].astype(np.float32),
            masks             = d['masks'].astype(np.float32),
            actor_mask        = (
                d['actor_mask'].astype(np.float32)
                if 'actor_mask' in d.files
                else np.ones_like(d['rewards'], dtype=np.float32)
            ),
        )
    if successful_demos_only:
        val_data = _keep_successful_episodes(val_data, 'validation')
    if terminal_sparse_rewards:
        val_data = _relabel_terminal_sparse_rewards(val_data, 'validation')
    if outcome_time_rewards:
        val_data = _relabel_outcome_time_rewards(val_data, 'validation')
    val_dataset = HistoryDataset.create(
        hist_length=hist_length, hist_stride=hist_stride, **val_data,
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Factories (called from env_utils)
# ---------------------------------------------------------------------------

def make_search_cabinet_eval_env(seed: int = 0, image_size: int = 128,
                                  randomize_cabinet_pose: bool = False) -> SearchCabinetManiSkillEnv:
    """Return a single SearchCabinetManiSkillEnv.

    Used by _make_single_eval_env.  Raises ValueError if called with
    num_envs > 1 (via make_vec_eval_env) — use the single-env path instead.
    """
    return SearchCabinetManiSkillEnv(seed=seed, image_size=image_size,
                                     randomize_cabinet_pose=randomize_cabinet_pose)


def make_search_cabinet_env_and_datasets(
    hist_length: int = 4,
    hist_stride: int = 1,
    dataset_dir: str = _DEFAULT_DATASET_DIR,
    seed: int = 0,
    online_buf_size: int = 20_000,
    ep_cache_size: int = 40,
    chunk_reload_interval: int = 1000,
    action_chunk_size: int = 1,
    discount: float = 1.0,
    image_size: int = 128,
    randomize_cabinet_pose: bool = False,
    lazy: bool = True,
    max_demos: int = 0,
    successful_demos_only: bool = False,
    num_success_demos: int = -1,
    num_failure_demos: int = -1,
    terminal_sparse_rewards: bool = False,
    outcome_time_rewards: bool = False,
):
    """Return (env, eval_env, train_dataset, val_dataset) for the pipeline.

    The returned envs are SearchCabinetManiSkillEnv instances backed by the
    full ManiSkill simulator.  Datasets are loaded from pre-collected NPZ
    files.
    """
    train_dataset, val_dataset = load_search_cabinet_datasets(
        dataset_dir=dataset_dir,
        hist_length=hist_length,
        hist_stride=hist_stride,
        online_buf_size=online_buf_size,
        ep_cache_size=ep_cache_size,
        chunk_reload_interval=chunk_reload_interval,
        action_chunk_size=action_chunk_size,
        discount=discount,
        lazy=lazy,
        max_demos=max_demos,
        successful_demos_only=successful_demos_only,
        num_success_demos=num_success_demos,
        num_failure_demos=num_failure_demos,
        terminal_sparse_rewards=terminal_sparse_rewards,
        outcome_time_rewards=outcome_time_rewards,
    )

    env      = SearchCabinetManiSkillEnv(seed=seed, image_size=image_size,
                                         randomize_cabinet_pose=randomize_cabinet_pose)
    eval_env = SearchCabinetManiSkillEnv(seed=seed, image_size=image_size,
                                         randomize_cabinet_pose=randomize_cabinet_pose)
    return env, eval_env, train_dataset, val_dataset
