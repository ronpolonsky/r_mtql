from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def _obs_index(obs, i):
    """Index env i from a batched vectorized observation (dict or array)."""
    if isinstance(obs, dict):
        return {k: v[i] for k, v in obs.items()}
    return obs[i]


def _obs_copy(obs):
    """Copy numpy arrays in obs (dict or array)."""
    if isinstance(obs, dict):
        return {k: v.copy() for k, v in obs.items()}
    return obs.copy()


def _obs_stack(obs_list, axis=0):
    """Stack a list of obs (dict or array) into a single array/dict along axis."""
    if isinstance(obs_list[0], dict):
        return {k: np.stack([o[k] for o in obs_list], axis=axis) for k in obs_list[0]}
    return np.stack(obs_list, axis=axis)


def _obs_expand(obs):
    """Add a leading batch dimension to obs (dict or array)."""
    if isinstance(obs, dict):
        return {k: v[None] for k, v in obs.items()}
    return obs[None]


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)



def mem_evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    hist_length=None,  # Set to None or 0 to disable history, or pass a positive int.
    hist_stride=1,
    fixed_env_seed=None,
    eval_seed=None,
    target_counts=None,
    action_clip_eps = 1e-5,
    action_exec_horizon=0,  # How many sub-actions to execute before re-querying. 0 = full chunk.
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate.
        num_video_episodes: Number of episodes to render (not included in stats).
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        hist_length: Number of past history steps. Must match the hist_length
            used during training. None or 0 disables history; the current
            observation is always supplied separately.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    legacy_actor_fn = supply_rng(
        agent.sample_actions,
        rng=jax.random.PRNGKey(np.random.randint(0, 2**32)),
    )
    use_history = hist_length is not None and hist_length > 0
    trajs = []
    stats = defaultdict(list)
    renders = []

    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        # A fixed per-slot seed makes checkpoint comparisons use common random
        # numbers. Recreate the policy RNG per episode so a long prior episode
        # cannot change the noise stream of later episode slots.
        if eval_seed is not None:
            episode_seed = int(eval_seed) + i
            actor_fn = supply_rng(
                agent.sample_actions,
                rng=jax.random.PRNGKey(int(eval_seed) + 1_000_003 + i),
            )
        else:
            episode_seed = fixed_env_seed
            actor_fn = legacy_actor_fn
        reset_options = None
        if target_counts:
            reset_options = {"target_count": int(target_counts[i % len(target_counts)])}
        if reset_options is None:
            observation, info = env.reset(seed=episode_seed)
        else:
            observation, info = env.reset(seed=episode_seed, options=reset_options)
        done = False
        step = 0
        render = []

        # Rolling history buffers — always length hist_length.
        # obs_history[t] = obs before action at step t.
        # act_history[t] = action taken at step t.
        # At episode start both are pre-filled to match training padding:
        #   observations -> repeat initial obs (clamp to ep start)
        #   actions      -> -1 (pad value used in HistoryDataset)
        if use_history:
            history_window = hist_length * hist_stride
            obs_history = [observation] * history_window

        while not done:
            if use_history:
                # Sample sparse history with stride, matching HistoryDataset indexing.
                # Example hist_length=4, hist_stride=2 -> keep indices [0,2,4,6]
                history_observations = _obs_expand(_obs_stack(obs_history[::hist_stride]))
                if step == 0:
                    traj['first_obs']      = observation
                    traj['first_hist_obs'] = _obs_stack(obs_history[::hist_stride])
                action = actor_fn(
                    observations=_obs_expand(observation),
                    temperature=eval_temperature,
                    history_observations=history_observations,
                )
            else:
                if step == 0:
                    traj['first_obs']      = observation
                    traj['first_hist_obs'] = None
                action = actor_fn(observations=_obs_expand(observation), temperature=eval_temperature)

            # action: (1, action_chunk, action_dim) -> squeeze -> (action_chunk, action_dim)
            action = np.squeeze(np.array(action))

            # Execute action chunk sub-actions one by one.
            # History advances and transitions are recorded at every physical step,
            # including the terminal step.
            full_chunk_size = action.shape[0]  # action is (action_chunk, action_dim)
            exec_horizon = action_exec_horizon if action_exec_horizon > 0 else full_chunk_size
            for k in range(exec_horizon):
                if use_history:
                    # Push current observation into history before stepping.
                    obs_history = obs_history[1:] + [observation]
                sub_action = action[k]
                next_observation, r, terminated, truncated, info = env.step(sub_action)
                done = terminated or truncated

                if should_render and (step % video_frame_skip == 0 or done):
                    render.append(env.render().copy())

                transition = dict(
                    observation=observation,
                    next_observation=next_observation,
                    action=sub_action,
                    reward=r,
                    done=done,
                    info=info,
                )
                add_to(traj, transition)
                observation = next_observation
                step += 1

                if done:
                    break

        if i < num_eval_episodes:
            flat_info = flatten(info)
            numeric_info = {
                key: value
                for key, value in flat_info.items()
                if np.asarray(value).ndim == 0
                and (
                    np.issubdtype(np.asarray(value).dtype, np.number)
                    or np.issubdtype(np.asarray(value).dtype, np.bool_)
                )
            }
            add_to(stats, numeric_info)
            target = flat_info.get("target_count")
            if target is not None:
                add_to(
                    stats,
                    {f"target_{int(target)}/{key}": value for key, value in numeric_info.items()},
                )
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    success_values = np.asarray(stats.get("success", []), dtype=np.float32)
    for k, v in stats.items():
        stats[k] = np.mean(v)
    if len(success_values) > 1:
        stats["success_se"] = float(
            success_values.std(ddof=1) / np.sqrt(len(success_values))
        )

    return stats, trajs, renders



def vec_mem_evaluate(
    agent,
    vec_env,
    num_eval_episodes=50,
    eval_temperature=0,
    hist_length=None,
    hist_stride=1,
    fixed_env_seed=None,
    eval_seed=None,
    target_counts=None,
    action_exec_horizon=0,
):
    """Vectorized eval: runs num_envs episodes in parallel with batched policy calls.

    Args:
        agent: Agent with sample_actions supporting batched observations.
        vec_env: A gymnasium SyncVectorEnv (or compatible) with num_envs sub-envs.
        num_eval_episodes: Total completed episodes to collect.
        hist_length: History length (must match training). None or 0 disables history.
        hist_stride: Stride for history sampling.

    Returns:
        (stats dict, [], [])  — same signature as mem_evaluate (no video support).
    """
    num_envs = vec_env.num_envs
    use_history = hist_length is not None and hist_length > 0
    if target_counts:
        raise ValueError('Target-balanced evaluation is unsupported for vector environments.')
    actor_rng_seed = (
        int(eval_seed) + 1_000_003
        if eval_seed is not None
        else np.random.randint(0, 2**32)
    )
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(actor_rng_seed))
    action_shape = vec_env.single_action_space.shape

    stats = defaultdict(list)
    completed = 0

    if eval_seed is not None:
        reset_seed = [int(eval_seed) + i for i in range(num_envs)]
    else:
        reset_seed = [fixed_env_seed] * num_envs if fixed_env_seed is not None else None
    obs, _ = vec_env.reset(seed=reset_seed)  # (num_envs, obs_dim)
    import math
    n_batches = math.ceil(num_eval_episodes / num_envs)
    pbar = trange(n_batches, desc=f'vec_eval ({num_envs} envs)')
    batches_reported = 0

    if use_history:
        history_window = hist_length * hist_stride
        obs_histories = [[_obs_copy(_obs_index(obs, i))] * history_window for i in range(num_envs)]

    while completed < num_eval_episodes:
        if use_history:
            history_observations = _obs_stack(
                [_obs_stack(obs_histories[i][::hist_stride]) for i in range(num_envs)]
            )  # (num_envs, hist_length, ...)
            actions = actor_fn(
                observations=obs,
                temperature=eval_temperature,
                history_observations=history_observations,
            )
        else:
            actions = actor_fn(observations=obs, temperature=eval_temperature)

        # Execute action chunk sub-actions one by one.
        full_chunk_size = actions.shape[1] if actions.ndim == 3 else 1
        exec_horizon = action_exec_horizon if action_exec_horizon > 0 else full_chunk_size

        chunk_dones = np.zeros(num_envs, dtype=bool)
        for k in range(exec_horizon):
            sub_actions = actions[:, k, :] if actions.ndim == 3 else actions

            # Slide history for envs still running in this chunk.
            if use_history:
                for i in range(num_envs):
                    if not chunk_dones[i]:
                        obs_histories[i] = obs_histories[i][1:] + [_obs_copy(_obs_index(obs, i))]

            obs, _rewards, terminateds, truncateds, infos = vec_env.step(sub_actions)
            dones = (terminateds | truncateds) & ~chunk_dones
            chunk_dones |= dones

            for i in range(num_envs):
                if not dones[i] or completed >= num_eval_episodes:
                    continue
                final_infos = infos.get('final_info', None)
                if final_infos is not None and final_infos[i] is not None:
                    add_to(stats, flatten(final_infos[i]))
                completed += 1
                if use_history:
                    obs_histories[i] = [_obs_copy(_obs_index(obs, i))] * history_window

            if chunk_dones.all():
                break

        new_batches = completed // num_envs - batches_reported
        if new_batches > 0:
            pbar.update(new_batches)
            batches_reported += new_batches

    pbar.close()
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, [], []


def collect_online_episodes(
    agent,
    env,
    n_episodes,
    hist_length=None,
    hist_stride=1,
    collect_temperature=0,
    fixed_env_seed=None,
):
    """Roll out the current agent and return collected transitions as numpy arrays.

    Returns a dict with keys: observations, actions, rewards, next_observations,
    terminals, masks — ready to be appended to the offline training dataset.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    use_history = hist_length is not None and hist_length > 0

    obs_list, act_list, rew_list, next_obs_list, terminal_list, mask_list = [], [], [], [], [], []
    episode_stats = defaultdict(list)

    for _ in trange(n_episodes, desc='Collecting online episodes'):
        observation, _ = env.reset(seed=fixed_env_seed)
        done = False

        if use_history:
            history_window = hist_length * hist_stride
            obs_history = [observation] * history_window

        while not done:
            if use_history:
                history_observations = _obs_expand(_obs_stack(obs_history[::hist_stride]))
                action = actor_fn(
                    observations=_obs_expand(observation),
                    temperature=collect_temperature,
                    history_observations=history_observations,
                )
            else:
                action = actor_fn(observations=_obs_expand(observation), temperature=collect_temperature)

            action = np.squeeze(np.array(action))
            action = np.clip(action, -1, 1)

            chunk_size = action.shape[0] // env.action_space.shape[0] if action.ndim > 0 else 1
            sub_dim = env.action_space.shape[0]
            total_reward = 0.0
            next_observation = observation
            terminated = truncated = False
            info = {}
            for k in range(chunk_size):
                if use_history:
                    obs_history = obs_history[1:] + [next_observation]
                sub_action = action[k * sub_dim:(k + 1) * sub_dim]
                next_observation, r, terminated, truncated, info = env.step(sub_action)
                total_reward += r
                if terminated or truncated:
                    break
            reward = total_reward
            done = terminated or truncated

            obs_list.append(observation)
            act_list.append(action)
            rew_list.append(reward)
            next_obs_list.append(next_observation)
            terminal_list.append(1.0 if terminated else 0.0)
            mask_list.append(0.0 if terminated else 1.0)

            observation = next_observation

        add_to(episode_stats, flatten(info))

    for k, v in episode_stats.items():
        episode_stats[k] = np.mean(v)

    return dict(
        observations=np.array(obs_list, dtype=np.float32),
        actions=np.array(act_list, dtype=np.float32),
        rewards=np.array(rew_list, dtype=np.float32),
        next_observations=np.array(next_obs_list, dtype=np.float32),
        terminals=np.array(terminal_list, dtype=np.float32),
        masks=np.array(mask_list, dtype=np.float32),
    ), dict(episode_stats)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            chunk_size = action.shape[0] // env.action_space.shape[0] if action.ndim > 0 else 1
            sub_dim = env.action_space.shape[0]
            total_reward = 0.0
            next_observation = observation
            terminated = truncated = False
            info = {}
            for k in range(chunk_size):
                sub_action = action[k * sub_dim:(k + 1) * sub_dim]
                next_observation, r, terminated, truncated, info = env.step(sub_action)
                total_reward += r
                if should_render and (step % video_frame_skip == 0 or terminated or truncated):
                    render.append(env.render().copy())
                if terminated or truncated:
                    break
            reward = total_reward
            done = terminated or truncated
            step += 1

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
