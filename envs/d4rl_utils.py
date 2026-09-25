import gym
import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import gymnasium
import numpy as np

from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset


def make_env(env_name):
    """Make D4RL environment."""
    env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env = EpisodeMonitor(env)
    return env

def make_kitchen_env(env_name):
    """Make D4RL environment."""
    env = gym.make(env_name)
    env = KitchenWrapper(env)
    env = EpisodeMonitor(env)
    return env


class KitchenWrapper():
    def __init__(self, env):
        self.env = env
        self.unwrapped = self.env.unwrapped
    
    def reset(self):
        return self.env.reset(), 0
    
    def get_normalized_score(self):
        return self.env.get_normalized_score()
    
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, done, info
    
    def get_dataset(self):
        return self.env.get_dataset()


def get_dataset(
    env,
    env_name,
):
    """Make D4RL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
    """
    dataset = d4rl.qlearning_dataset(env)

    terminals = np.zeros_like(dataset['rewards'])  # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            )
            masks[i] = 1 - dataset['terminals'][i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if (
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
                or dataset['terminals'][i] == 1.0
            ):
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - dataset['terminals'][i]
    masks[-1] = 1 - dataset['terminals'][-1]
    terminals[-1] = 1

    return Dataset.create(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32),
        terminals=terminals.astype(np.float32),
        rewards=rewards,
        masks=masks,
    )
