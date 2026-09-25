import collections
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box

from utils.datasets import Dataset, HistoryDataset


class OGBenchVisualDictObsWrapper(gymnasium.Wrapper):
    """Wraps a visual OGBench env to return {'image': uint8, 'proprio': float32} dicts.

    The image is the rendered pixel observation; proprio is the concatenation of
    qpos and qvel obtained via env.unwrapped.get_ob('states').
    """

    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space  # Box(0,255,(H,W,C),uint8)
        try:
            proprio_dim = env.unwrapped.model.nq + env.unwrapped.model.nv
            self._has_proprio = True
        except AttributeError:
            print(
                '[OGBenchVisualDictObsWrapper] WARNING: env.unwrapped.model not found — '
                'cannot determine proprio dimension. Proprio will NOT be included in env observations.'
            )
            self._has_proprio = False
            proprio_dim = 0
        if self._has_proprio:
            proprio_space = Box(-np.inf, np.inf, shape=(proprio_dim,), dtype=np.float32)
            self.observation_space = gymnasium.spaces.Dict({
                'image': img_space,
                'proprio': proprio_space,
            })
        else:
            self.observation_space = gymnasium.spaces.Dict({'image': img_space})

    def _make_obs(self, img):
        if not self._has_proprio:
            return {'image': img}
        proprio = self.env.unwrapped.get_ob('states').astype(np.float32)
        return {'image': img, 'proprio': proprio}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._make_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._make_obs(obs), reward, terminated, truncated, info


class OGBenchImageOnlyWrapper(gymnasium.Wrapper):
    """Wraps a visual OGBench env to return {'image': uint8} dicts (no proprio)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Dict({'image': env.observation_space})

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {'image': obs}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {'image': obs}, reward, terminated, truncated, info


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info



def make_mem_env_and_datasets(env_name, hist_length=1, hist_stride=1, frame_stack=None, action_clip_eps=1e-5,
                              num_expert_episodes=500, num_suboptimal_episodes=0, count_reward=False,
                              fully_observable=False, small_obs=False, max_episode_steps=70,
                              randomization='large', image_obs=False, num_cached_episodes=40,
                              chunk_reload_interval=1000, action_chunk_size=1, discount=1.0,
                              online_buf_size=2_000_000,
                              lazy_dataset=True,
                              max_demos=0,
                              successful_demos_only=False,
                              num_success_demos=-1,
                              num_failure_demos=-1):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        num_expert_episodes: Number of expert episodes (house env only).
        num_suboptimal_episodes: Number of suboptimal episodes (house env only).

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if env_name.startswith('visual-') and 'singletask' not in env_name:
        import os
        from ogbench.utils import download_datasets, DEFAULT_DATASET_DIR

        env = OGBenchVisualDictObsWrapper(ogbench.make_env_and_datasets(env_name, env_only=True))
        eval_env = OGBenchVisualDictObsWrapper(ogbench.make_env_and_datasets(env_name, env_only=True))
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])

        dataset_dir = os.path.expanduser(DEFAULT_DATASET_DIR)
        download_datasets([env_name], dataset_dir)  # no-op if already downloaded

        if lazy_dataset:
            from utils.datasets import (
                LazyOGBenchEpisodeReplayBuffer,
                split_ogbench_npz_into_episodes,
            )
            train_npz = os.path.join(dataset_dir, f'{env_name}.npz')
            val_npz = os.path.join(dataset_dir, f'{env_name}-val.npz')

            cache_base = os.path.expanduser(f'~/.ogbench/episodes/{env_name}')
            train_ep_paths = split_ogbench_npz_into_episodes(train_npz, os.path.join(cache_base, 'train'))
            val_ep_paths = split_ogbench_npz_into_episodes(val_npz, os.path.join(cache_base, 'val'))

            _lazy_kwargs = dict(
                hist_length=hist_length,
                hist_stride=hist_stride,
                ep_cache_size=num_cached_episodes,
                chunk_reload_interval=chunk_reload_interval,
                action_chunk_size=action_chunk_size,
                discount=discount,
            )
            train_dataset = LazyOGBenchEpisodeReplayBuffer(
                train_ep_paths, online_buf_size=online_buf_size, **_lazy_kwargs
            )
            val_dataset = LazyOGBenchEpisodeReplayBuffer(val_ep_paths, **_lazy_kwargs)
        else:
            from utils.datasets import load_visual_ogbench_npz
            train_npz = os.path.join(dataset_dir, f'{env_name}.npz')
            val_npz = os.path.join(dataset_dir, f'{env_name}-val.npz')
            train_data = load_visual_ogbench_npz(train_npz, max_demos=max_demos)
            train_dataset = HistoryDataset.create(
                hist_length=hist_length, hist_stride=hist_stride, **train_data,
            )
            val_data = load_visual_ogbench_npz(val_npz)
            val_dataset = HistoryDataset.create(
                hist_length=hist_length, hist_stride=hist_stride, **val_data,
            )
    elif 'singletask' in env_name:
        # OGBench singletask (non-visual or visual singletask): load fully.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        if env_name.startswith('visual-'):
            env = OGBenchImageOnlyWrapper(env)
            eval_env = OGBenchImageOnlyWrapper(eval_env)
            # Wrap dataset observations in {'image': ...} dict to match env output.
            for ds in [train_dataset, val_dataset]:
                ds['observations'] = {'image': ds['observations']}
                if 'next_observations' in ds:
                    del ds['next_observations']
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = HistoryDataset.create(
            hist_length=hist_length,
            hist_stride=hist_stride,
            **train_dataset,
        )
        val_dataset = HistoryDataset.create(
            hist_length=hist_length,
            hist_stride=hist_stride,
            **val_dataset,
        )
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'kitchen' in env_name:
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_kitchen_env(env_name)
        eval_env = d4rl_utils.make_kitchen_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'house' in env_name:
        from envs.house_env import make_house_env, generate_house_dataset

        env      = EpisodeMonitor(make_house_env(env_name, count_reward=count_reward, fully_observable=fully_observable, small_obs=small_obs, max_steps=max_episode_steps, randomization=randomization, image_obs=image_obs))
        eval_env = EpisodeMonitor(make_house_env(env_name, count_reward=count_reward, fully_observable=fully_observable, small_obs=small_obs, max_steps=max_episode_steps, randomization=randomization, image_obs=image_obs))
        train_dataset, val_dataset = generate_house_dataset(
            env_name,
            num_expert_episodes=num_expert_episodes,
            num_suboptimal_episodes=num_suboptimal_episodes,
            count_reward=count_reward,
            fully_observable=fully_observable,
            small_obs=small_obs,
            max_steps=max_episode_steps,
            randomization=randomization,
            image_obs=image_obs,
        )
    elif env_name == 'drawer_task':
        from drawer_task.env import make_drawer_env_and_datasets

        env, eval_env, train_dataset, val_dataset = make_drawer_env_and_datasets(
            hist_length=hist_length,
            hist_stride=hist_stride,
            online_buf_size=online_buf_size,
            ep_cache_size=num_cached_episodes,
            chunk_reload_interval=chunk_reload_interval,
            action_chunk_size=action_chunk_size,
            discount=discount,
            successful_demos_only=successful_demos_only,
            num_success_demos=num_success_demos,
            num_failure_demos=num_failure_demos,
        )
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
    elif env_name == 'counting':
        from envs.counting_env import make_counting_env_and_datasets

        env, eval_env, train_dataset, val_dataset = make_counting_env_and_datasets(
            hist_length=hist_length,
            hist_stride=hist_stride,
            online_buf_size=online_buf_size,
            ep_cache_size=num_cached_episodes,
            chunk_reload_interval=chunk_reload_interval,
            action_chunk_size=action_chunk_size,
            discount=discount,
            image_size=128,
            lazy=lazy_dataset,
            max_demos=max_demos,
        )
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
    elif 'search_cabinet' in env_name:
        from envs.search_cabinet_env import (
            make_search_cabinet_env_and_datasets,
            _TWO_CAMS_DATASET_DIR,
            _TWO_CAMS_MORE_RAND_DATASET_DIR,
            _TWO_CAMS_MORE_RAND_HARD_BC_S25_F15_DATASET_DIR,
        )
        _image_size = 64 if 'low_res' in env_name else 128
        _dataset_dir_kwargs = {}
        if 'two_cams_more_rand_hard_bc_s25_f15' in env_name:
            _dataset_dir_kwargs['dataset_dir'] = (
                _TWO_CAMS_MORE_RAND_HARD_BC_S25_F15_DATASET_DIR
            )
        elif 'two_cams_more_rand' in env_name:
            _dataset_dir_kwargs['dataset_dir'] = _TWO_CAMS_MORE_RAND_DATASET_DIR
        elif 'two_cams' in env_name:
            _dataset_dir_kwargs['dataset_dir'] = _TWO_CAMS_DATASET_DIR
        env, eval_env, train_dataset, val_dataset = make_search_cabinet_env_and_datasets(
            hist_length=hist_length,
            hist_stride=hist_stride,
            online_buf_size=online_buf_size,
            ep_cache_size=num_cached_episodes,
            chunk_reload_interval=chunk_reload_interval,
            action_chunk_size=action_chunk_size,
            discount=discount,
            image_size=_image_size,
            randomize_cabinet_pose='more_rand' in env_name,
            lazy=lazy_dataset,
            max_demos=max_demos,
            successful_demos_only=successful_demos_only,
            num_success_demos=num_success_demos,
            num_failure_demos=num_failure_demos,
            terminal_sparse_rewards='terminal_sparse' in env_name,
            outcome_time_rewards='outcome_time' in env_name,
            **_dataset_dir_kwargs,
        )
        env      = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset




def _make_single_eval_env(env_name, frame_stack=None, count_reward=False, fully_observable=False, small_obs=False, max_episode_steps=70, randomization='large', image_obs=False):
    """Create a single eval env instance (no datasets). Used to build vectorized envs."""
    if 'singletask' in env_name or env_name.startswith('visual-'):
        env = ogbench.make_env_and_datasets(env_name, env_only=True)
        if env_name.startswith('visual-') and 'singletask' in env_name:
            env = OGBenchImageOnlyWrapper(env)
        elif env_name.startswith('visual-'):
            env = OGBenchVisualDictObsWrapper(env)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        from envs import d4rl_utils
        env = d4rl_utils.make_env(env_name)
    elif 'kitchen' in env_name:
        from envs import d4rl_utils
        env = d4rl_utils.make_kitchen_env(env_name)
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils
        env = d4rl_utils.make_env(env_name)
    elif 'house' in env_name:
        from envs.house_env import make_house_env
        env = EpisodeMonitor(make_house_env(env_name, count_reward=count_reward, fully_observable=fully_observable, small_obs=small_obs, max_steps=max_episode_steps, randomization=randomization, image_obs=image_obs))
    elif env_name == 'drawer_task':
        from drawer_task.env import make_drawer_env

        env = EpisodeMonitor(make_drawer_env(image_size=128))
    elif env_name == 'counting':
        from envs.counting_env import make_counting_eval_env

        env = EpisodeMonitor(make_counting_eval_env(image_size=128))
    elif 'search_cabinet' in env_name:
        from envs.search_cabinet_env import make_search_cabinet_eval_env
        _image_size = 64 if 'low_res' in env_name else 128
        env = EpisodeMonitor(make_search_cabinet_eval_env(image_size=_image_size,
                                                          randomize_cabinet_pose='more_rand' in env_name))
    else:
        raise ValueError(f'Unsupported environment: {env_name}')
    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
    return env


def make_vec_eval_env(env_name, num_envs, frame_stack=None, count_reward=False, fully_observable=False, small_obs=False, max_episode_steps=70, randomization='large', image_obs=False):
    """Create a SyncVectorEnv with num_envs parallel eval copies."""
    if (env_name in ('counting', 'drawer_task') or 'search_cabinet' in env_name) and num_envs > 1:
        raise ValueError(
            f'{env_name} uses ManiSkill vectorisation internally and only '
            f'supports num_envs=1 (got {num_envs}). '
            f'Set --num_eval_envs=1 to use the single-env evaluation path.'
        )
    def _env_fn(name, fs, cr, fo, so, mes, rand, io):
        def fn():
            return _make_single_eval_env(name, frame_stack=fs, count_reward=cr, fully_observable=fo, small_obs=so, max_episode_steps=mes, randomization=rand, image_obs=io)
        return fn

    env_fns = [_env_fn(env_name, frame_stack, count_reward, fully_observable, small_obs, max_episode_steps, randomization, image_obs) for _ in range(num_envs)]
    return gymnasium.vector.SyncVectorEnv(env_fns)


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'kitchen' in env_name:
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_kitchen_env(env_name)
        eval_env = d4rl_utils.make_kitchen_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'house' in env_name:
        from envs.house_env import make_house_env, generate_house_dataset

        env      = EpisodeMonitor(make_house_env(env_name))
        eval_env = EpisodeMonitor(make_house_env(env_name))
        train_dataset, val_dataset = generate_house_dataset(env_name)
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
