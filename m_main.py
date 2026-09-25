import os
import multiprocessing as mp

import json
import random
import time
import glob
import jax

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_mem_env_and_datasets, make_env_and_datasets, make_vec_eval_env
from utils.datasets import Dataset, HistoryReplayBuffer, add_mc_returns
from utils.evaluation import (evaluate, flatten, mem_evaluate, vec_mem_evaluate,
                               collect_online_episodes, _obs_stack, _obs_expand)
from utils.flax_utils import restore_agent, save_agent, print_param_stats
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'debug', 'Run group.')
flags.DEFINE_string('project', 'dfrl', 'Run group.')
flags.DEFINE_string('wandb_mode', 'online', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('hist_length', 4, 'Evaluation interval.')
flags.DEFINE_integer('hist_stride', 1, 'Stride for history sampling.')
flags.DEFINE_integer('save_interval', 200000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

flags.DEFINE_integer('num_expert_episodes', 100, 'Number of expert episodes for house dataset.')
flags.DEFINE_integer('num_suboptimal_episodes', 50, 'Number of suboptimal episodes for house dataset.')

flags.DEFINE_integer('online_collect_interval', 0, 'Collect online episodes every N steps after warmup. 0 disables online collection.')
flags.DEFINE_integer('online_warmup_steps', 1000000, 'Steps before online collection begins.')
flags.DEFINE_integer('online_episodes_per_collection', 500, 'Episodes to collect at each online collection step.')
flags.DEFINE_bool('single_step_online', False, 'Collect 1 transition per training step after warmup, append to dataset, then do the normal gradient update.')
flags.DEFINE_integer('online_buf_size', 2000000, 'Max size of the HistoryReplayBuffer used by single_step_online.')
flags.DEFINE_bool('lazy_dataset', True, 'Use LazyEpisodeReplayBuffer (True) or load full dataset into RAM (False).')
flags.DEFINE_integer('max_demos', 0, 'Max number of demo episodes to load (0 = all). Only applies to non-lazy dataset.')
flags.DEFINE_bool(
    'successful_demos_only',
    False,
    'Keep only complete successful demonstrations from supported offline datasets. '
    'Currently supported for non-lazy cabinet datasets.',
)
flags.DEFINE_integer(
    'num_success_demos',
    -1,
    'For non-lazy cabinet data, keep this many complete successful episodes '
    '(-1 = all).',
)
flags.DEFINE_integer(
    'num_failure_demos',
    -1,
    'For non-lazy cabinet data, keep this many complete failed episodes '
    '(-1 = all).',
)
flags.DEFINE_integer('num_cached_episodes', 40, 'Number of offline episodes per chunk (LazyEpisodeReplayBuffer).')
flags.DEFINE_integer('chunk_reload_interval', 1000, 'Reload a new random episode chunk every N sample() calls.')
flags.DEFINE_integer('action_chunk_size', 1, 'Number of consecutive actions to predict and execute per step (action chunking).')
flags.DEFINE_integer('action_exec_horizon', 0, 'Number of sub-actions to execute per chunk before re-querying the policy. 0 = execute the full chunk (default).')
flags.DEFINE_integer('num_online_envs', 1, 'Number of parallel envs for single_step_online collection. >1 uses a SyncVectorEnv and collects N transitions per step.')

flags.DEFINE_integer('debug_train', 0, 'Skip evaluations and saving for faster iteration.')
flags.DEFINE_integer('num_eval_envs', 1, 'Number of parallel envs for vectorized evaluation.')
flags.DEFINE_bool('count_reward', False, 'Use count-based reward (reward = num objects delivered) instead of sparse +1 per delivery.')
flags.DEFINE_bool('env_fully_observable', False, 'Make env fully observable (no FOV cone).')
flags.DEFINE_bool('small_obs', False, 'Use compact vis observation (2*num_objects floats: normalized XY or -1,-1 per object) instead of the full G×G FOV grid.')
flags.DEFINE_integer('fixed_env_seed', -1, 'Fixed seed for eval and online env resets (-1 = random each episode). Dataset generation always uses many seeds.')
flags.DEFINE_integer(
    'eval_seed', -1,
    'Base seed for a reproducible per-episode evaluation suite (-1 = legacy random evaluation).',
)
flags.DEFINE_bool(
    'balanced_counting_eval', False,
    'For counting, cycle explicitly through targets 1, 2, and 3 during evaluation.',
)
flags.DEFINE_bool('visualize_expert', False, 'Save rendered expert demo episodes to save_dir/expert_demos/ as GIF files.')
flags.DEFINE_integer('visualize_expert_episodes', 5, 'Number of expert episodes to render when --visualize_expert is set.')
flags.DEFINE_bool('reset_opt_state', False, 'Reset optimizer state on restore (needed when optimizer chain changes, e.g. adding grad clipping).')
flags.DEFINE_integer('online_reset_seed', -1, 'If >= 0, re-seeds the online env RNG before every episode reset so the agent always sees the same starting layout.')
flags.DEFINE_integer('max_episode_steps', 100, 'Maximum steps per episode (house env only).')
flags.DEFINE_string('env_randomization', 'large', 'Position randomization level for house env: low, medium, or large (default).')
flags.DEFINE_bool('image_obs', False, 'Use 80×80 RGB image observations instead of flat vectors (house env only).')
flags.DEFINE_bool('visualize_online', False, 'Render and log a wandb video for each completed online episode (single-env only).')
flags.DEFINE_bool(
    'use_env_reward',
    False,
    'Store the environment reward during online collection instead of the legacy success/non-success shaping.',
)

config_flags.DEFINE_config_file('agent', 'agents/tql.py', lock_config=False)

def main(argv):
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    if len(argv) != 1:
        extra = argv[1:]
        raise app.UsageError(
            f"Unexpected positional arguments: {extra}. "
            "If you meant to set a boolean flag, use --flag=false (or --noflag), not '--flag False'."
        )

    config = FLAGS.agent
    exp_name = get_exp_name(FLAGS.seed, algo=config.get('agent_name'), hist_length=FLAGS.hist_length, image_obs=FLAGS.image_obs)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    dim = config.get('hidden_dim')
    if dim and 'num_heads' in config:
        assert dim % 32 == 0, f"Hidden dim ({dim}) must be divisible by 32"
        config['num_heads'] = dim // 32
    
    wandb_run_id = None
    if FLAGS.enable_wandb:
        wandb_run, trigger_sync = setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project=FLAGS.project, group=FLAGS.wandb_run_group, name=exp_name,
            mode=FLAGS.wandb_mode
        )
        wandb_run_id = wandb_run.id
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config['train_steps'] = FLAGS.train_steps
    config['action_chunk_size'] = FLAGS.action_chunk_size
    env, eval_env, train_dataset, val_dataset = make_mem_env_and_datasets(
        FLAGS.env_name,
        hist_length=FLAGS.hist_length,
        hist_stride=FLAGS.hist_stride,
        frame_stack=FLAGS.frame_stack,
        num_expert_episodes=FLAGS.num_expert_episodes,
        num_suboptimal_episodes=FLAGS.num_suboptimal_episodes,
        count_reward=FLAGS.count_reward,
        fully_observable=FLAGS.env_fully_observable,
        small_obs=FLAGS.small_obs,
        max_episode_steps=FLAGS.max_episode_steps,
        randomization=FLAGS.env_randomization,
        image_obs=FLAGS.image_obs,
        action_clip_eps=None,
        num_cached_episodes=FLAGS.num_cached_episodes,
        chunk_reload_interval=FLAGS.chunk_reload_interval,
        action_chunk_size=FLAGS.action_chunk_size,
        discount=config.get('discount', 0.99),
        online_buf_size=FLAGS.online_buf_size,
        lazy_dataset=FLAGS.lazy_dataset,
        max_demos=FLAGS.max_demos,
        successful_demos_only=FLAGS.successful_demos_only,
        num_success_demos=FLAGS.num_success_demos,
        num_failure_demos=FLAGS.num_failure_demos,
    )
    
    if FLAGS.num_eval_envs > 1:
        vec_eval_env = make_vec_eval_env(FLAGS.env_name, FLAGS.num_eval_envs, frame_stack=FLAGS.frame_stack, count_reward=FLAGS.count_reward, fully_observable=FLAGS.env_fully_observable, small_obs=FLAGS.small_obs, max_episode_steps=FLAGS.max_episode_steps, randomization=FLAGS.env_randomization, image_obs=FLAGS.image_obs)
    else:
        vec_eval_env = None
    
    if FLAGS.video_episodes > 0:
        render_modes = getattr(eval_env, 'metadata', {}).get('render_modes', [])
        if 'rgb_array' not in render_modes:
            raise ValueError(
                f'Video rendering requested (video_episodes={FLAGS.video_episodes}) '
                f'but env "{FLAGS.env_name}" does not advertise rgb_array rendering.'
            )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    if config['agent_name'] == 'q_transformer':
        train_dataset = add_mc_returns(dict(train_dataset), config["discount"])
    if hasattr(train_dataset, 'add_transition'):
        # Dataset is already a replay buffer (e.g. LazyEpisodeReplayBuffer for
        # image-heavy envs where pre-allocating online_buf_size frames would
        # exhaust RAM).  Use it directly.
        replay_buffer = train_dataset
    else:
        raw_train = dict(train_dataset)
        dataset_size = len(raw_train['actions'])
        buf_size = (FLAGS.online_buf_size + dataset_size) if FLAGS.single_step_online else dataset_size
        train_dataset = HistoryReplayBuffer.create_from_initial_dataset(
            raw_train,
            size=buf_size,
            hist_length=FLAGS.hist_length,
            hist_stride=FLAGS.hist_stride,
        )
        replay_buffer = train_dataset

    _fixed_seed = FLAGS.fixed_env_seed if FLAGS.fixed_env_seed >= 0 else None
    _eval_seed = FLAGS.eval_seed if FLAGS.eval_seed >= 0 else None
    _eval_targets = (1, 2, 3) if (
        FLAGS.env_name == 'counting' and FLAGS.balanced_counting_eval
    ) else None
    if _eval_targets and FLAGS.eval_episodes % len(_eval_targets) != 0:
        raise ValueError(
            '--eval_episodes must be divisible by 3 when '
            '--balanced_counting_eval is enabled.'
        )

    if FLAGS.single_step_online:
        # Keep online collection isolated from evaluation. Evaluation resets and
        # steps eval_env, which would corrupt an in-progress online trajectory if
        # both names referred to the same environment object.
        online_env = env
        assert online_env is not eval_env
        history_window = FLAGS.hist_length * FLAGS.hist_stride if FLAGS.hist_length else 0
        observation, _ = online_env.reset()
        if history_window > 0:
            obs_history = [observation] * history_window
        _ep_buf = []
        _ss_ep_step = 0
        _ss_render_frames = []

    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if hasattr(dataset, 'action_chunk_size') and FLAGS.action_chunk_size > 1:
                dataset.action_chunk_size = FLAGS.action_chunk_size
                dataset.discount = config.get('discount', 0.99)
            if config['agent_name'] in ['rebrac', 'fdrl', 'q_transformer']:
                dataset.return_next_actions = True

    collecting_online = FLAGS.online_collect_interval > 0 and FLAGS.online_episodes_per_collection > 0

    # Create agent.
    example_batch = train_dataset.sample(1)

    assert 'rewards' in train_dataset
    example_batch['min_reward'] = float(train_dataset['rewards'].min())
    example_batch['max_reward'] = float(train_dataset['rewards'].max())
    assert example_batch['min_reward'] <= example_batch['max_reward']
    example_batch['dataset_action_min'] = train_dataset['actions'].min(axis=0).astype(np.float32)
    example_batch['dataset_action_max'] = train_dataset['actions'].max(axis=0).astype(np.float32)

    # Compute min/max normalization stats incrementally (one batch at a time to avoid OOM).
    # Actions are stored flat (B, acs*act_dim); reshape to (B*acs, act_dim) so stats
    # are per-step-action-dim and shared across chunk positions.
    _acs = FLAGS.action_chunk_size
    _act_min = _act_max = None
    _prop_min = _prop_max = None
    for _ in range(20):
        _b = train_dataset.sample(500)
        _a = _b['actions'].astype(np.float32)

        _act_min = _a.min(axis=0).min(axis=0) if _act_min is None else np.minimum(_act_min, _a.min(axis=0).min(axis=0))
        _act_max = _a.max(axis=0).max(axis=0) if _act_max is None else np.maximum(_act_max, _a.max(axis=0).max(axis=0))

        if isinstance(_b.get('observations'), dict) and 'proprio' in _b['observations']:
            _p = _b['observations']['proprio'].astype(np.float32)
            _prop_min = _p.min(axis=0) if _prop_min is None else np.minimum(_prop_min, _p.min(axis=0))
            _prop_max = _p.max(axis=0) if _prop_max is None else np.maximum(_prop_max, _p.max(axis=0))
    
    example_batch['action_min'] = _act_min
    example_batch['action_max'] = _act_max
    if _prop_min is not None:
        example_batch['proprio_min'] = _prop_min
        example_batch['proprio_max'] = _prop_max

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )
    print_param_stats(agent)

    # Restore agent.
    start_step = 1
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch, reset_opt_state=FLAGS.reset_opt_state)
        start_step = int(FLAGS.restore_epoch) + 1

    if start_step > FLAGS.train_steps:
        print(
            f"[main] start_step ({start_step}) > train_steps ({FLAGS.train_steps}); "
            "nothing to do. (Did you mean to increase --train_steps?)"
        )
        return

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    train_batch_size = config['batch_size']
    # Override save cadence for very large models to reduce checkpoint overhead.
    save_interval = FLAGS.save_interval

    _t_sample = _t_update = _t_inference = _t_env_step = _t_render = 0.0

    for i in tqdm.tqdm(range(start_step, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        _t0 = time.time()
        batch = train_dataset.sample(train_batch_size)
        _t_sample += time.time() - _t0

        _t0 = time.time()
        if config['agent_name'] == 'bc_flow_transformer':
            batch = dict(batch)
            batch['x_0'] = np.random.normal(size=batch['actions'].shape).astype(np.float32)
            batch['t']   = np.random.uniform(0, 1, size=(len(batch['actions']), 1)).astype(np.float32)
        if config['agent_name'] == 'rebrac':
            agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
        elif 'tql' in config['agent_name']:
            agent, update_info = agent.update(batch, step=i)
        else:
            agent, update_info = agent.update(batch)
        _t_update += time.time() - _t0

        # Single-step online RL: collect one transition, add to replay buffer.
        if FLAGS.single_step_online and i >= FLAGS.online_warmup_steps:
            _t0 = time.time()
            if history_window > 0:
                history_observations = _obs_expand(_obs_stack(obs_history[::FLAGS.hist_stride]))
                action = np.array(agent.sample_actions(
                    observations=_obs_expand(observation),
                    history_observations=history_observations,
                    seed=jax.random.PRNGKey(i),
                    temperature=0,
                ))[0]  # (1, action_chunk, action_dim) -> (action_chunk, action_dim)
            else:
                action = np.array(agent.sample_actions(
                    observations=_obs_expand(observation),
                    seed=jax.random.PRNGKey(i),
                    temperature=0,
                ))[0]  # (1, action_chunk, action_dim) -> (action_chunk, action_dim)
            _t_inference += time.time() - _t0

            # Execute sub-actions one by one, advancing history at each physical step.
            _t0 = time.time()
            # action is (action_chunk, action_dim)
            _exec_horizon = FLAGS.action_exec_horizon if FLAGS.action_exec_horizon > 0 else action.shape[0]
            next_observation = observation
            terminated = truncated = False
            info = {}
            success = False
            for _k in range(_exec_horizon):
                _step_obs = next_observation
                if history_window > 0:
                    obs_history = obs_history[1:] + [_step_obs]
                sub_action = action[_k]
                next_observation, env_reward, terminated, truncated, info = online_env.step(sub_action)
                if info.get('success', False):
                    success = True
                _step_reward = (
                    float(env_reward)
                    if FLAGS.use_env_reward
                    else (1.0 if success else -1.0)
                )
                _step_done = terminated or truncated
                _ep_buf.append(dict(
                    observations=_step_obs,
                    actions=sub_action,
                    rewards=np.float32(_step_reward),
                    next_observations=next_observation,
                    terminals=np.float32(1.0 if _step_done else 0.0),
                    masks=np.float32(0.0 if _step_done else 1.0),
                ))
                # Capture frames inside the sub-action loop so terminal episodes
                # (including successes) are also recorded.
                if FLAGS.visualize_online and _ss_ep_step % FLAGS.video_frame_skip == 0:
                    _t0_r = time.time()
                    if FLAGS.image_obs and isinstance(next_observation, dict) and 'image' in next_observation:
                        _ss_render_frames.append(next_observation['image'])
                    else:
                        _ss_render_frames.append(online_env.render())
                    _t_render += time.time() - _t0_r
                _ss_ep_step += 1
                if terminated or truncated:
                    break
            _t_env_step += time.time() - _t0
            done = terminated or truncated

            if done:
                # Store outcome metadata for agent variants that optionally
                # mask behavior losses. The base MTQL agent ignores it.
                actor_weight = np.float32(1.0 if success else 0.0)
                for transition in _ep_buf:
                    transition['actor_mask'] = actor_weight
                    replay_buffer.add_transition(transition)
                _ep_buf = []
                if FLAGS.enable_wandb:
                    ep_metrics = {f'online_rollout/{k}': v
                                  for k, v in flatten(info).items()
                                  if not isinstance(v, np.ndarray)}
                    ep_metrics['online_rollout/buffer_size'] = replay_buffer.size
                    if FLAGS.visualize_online and _ss_render_frames:
                        ep_metrics['online_rollout/video'] = get_wandb_video(renders=[np.array(_ss_render_frames)])
                    wandb.log(ep_metrics, step=i)
                observation, _ = online_env.reset()
                if history_window > 0:
                    obs_history = [observation] * history_window
                _ss_ep_step = 0
                _ss_render_frames = []
            else:
                observation = next_observation

        # Compute flow matching reconstruction MSE every 1k steps.
        # if i % 1000 == 0 and hasattr(agent, 'compute_recon_mse'):
        #     recon_metrics = agent.compute_recon_mse(batch)
        #     if not isinstance(recon_metrics, dict):
        #         recon_metrics = {'recon_mse': float(recon_metrics)}
        #     recon_metrics = {f'training/{k}': v for k, v in recon_metrics.items()}
        #     if FLAGS.enable_wandb:
        #         wandb.log(recon_metrics, step=i)
        #     train_logger.log(recon_metrics, step=i)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            # if val_dataset is not None:
            #     val_batch = val_dataset.sample(config['batch_size'])
            #     _, val_info = agent.total_loss(val_batch, grad_params=None)
            #     train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            # TODO: val_dataset hist_legnth
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics['time/sample_ms']    = _t_sample    / FLAGS.log_interval * 1000
            train_metrics['time/update_ms']    = _t_update    / FLAGS.log_interval * 1000
            train_metrics['time/inference_ms'] = _t_inference / FLAGS.log_interval * 1000
            train_metrics['time/env_step_ms']  = _t_env_step  / FLAGS.log_interval * 1000
            train_metrics['time/render_ms']    = _t_render    / FLAGS.log_interval * 1000
            _t_sample = _t_update = _t_inference = _t_env_step = _t_render = 0.0
            # Dataloader breakdown (only for LazyEpisodeReplayBuffer)
            if hasattr(train_dataset, 'log_timing'):
                train_dataset.log_timing(prefix=f'[dataloader step={i}]')
                n = max(train_dataset._t_calls, 1)
                train_metrics['time/dl_index_ms']   = train_dataset._t_index   / n * 1000
                train_metrics['time/dl_load_ms']    = train_dataset._t_load    / n * 1000
                train_metrics['time/dl_history_ms'] = train_dataset._t_history / n * 1000
                train_dataset.reset_timing()
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()

            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if not FLAGS.debug_train and FLAGS.eval_interval != 0 and (i == start_step or i % FLAGS.eval_interval == 0):
            eval_metrics = {}
            if vec_eval_env is not None:
                eval_info, trajs, _ = vec_mem_evaluate(
                    agent=agent,
                    vec_env=vec_eval_env,
                    num_eval_episodes=FLAGS.eval_episodes,
                    hist_length=FLAGS.hist_length,
                    hist_stride=FLAGS.hist_stride,
                    fixed_env_seed=_fixed_seed,
                    eval_seed=_eval_seed,
                    target_counts=_eval_targets,
                    action_exec_horizon=FLAGS.action_exec_horizon,
                ) 
            else:
                eval_info, trajs, _ = mem_evaluate(
                    agent=agent,
                    env=eval_env,
                    hist_length=FLAGS.hist_length,
                    hist_stride=FLAGS.hist_stride,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=0,
                    video_frame_skip=FLAGS.video_frame_skip,
                    fixed_env_seed=_fixed_seed,
                    eval_seed=_eval_seed,
                    target_counts=_eval_targets,
                    action_exec_horizon=FLAGS.action_exec_horizon,
                )
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            # First-chunk MSE: compare model predictions from demo obs vs eval obs,
            # both measured against demo ground-truth actions.
            if hasattr(agent, '_sample_actions_batch') and trajs and trajs[0].get('first_obs') is not None:
                # Stack eval first-step inputs into a batch
                eval_first_obs = _obs_stack([t['first_obs'] for t in trajs])
                first_hist = trajs[0]['first_hist_obs']
                eval_first_hist_obs = _obs_stack([t['first_hist_obs'] for t in trajs]) if first_hist is not None else None

                # Sample demo batch for ground-truth actions and demo obs
                demo_batch = train_dataset.sample(len(trajs))

                # Predict from eval obs, compare to demo GT actions
                eval_pred = agent._sample_actions_batch(eval_first_obs, jax.random.PRNGKey(0), eval_first_hist_obs)
                eval_first_chunk_mse = float(np.mean((np.array(eval_pred) - demo_batch['actions']) ** 2))

                # Predict from demo obs, compare to demo GT actions
                # if hasattr(agent, 'compute_recon_mse'):
                #     for k, v in agent.compute_recon_mse(demo_batch).items():
                #         eval_metrics[f'evaluation/demo_{k}'] = v

                eval_metrics['evaluation/eval_first_chunk_mse'] = eval_first_chunk_mse

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)

                if FLAGS.wandb_mode == 'offline':
                    trigger_sync()

            eval_logger.log(eval_metrics, step=i)

            if FLAGS.video_episodes > 0 and FLAGS.enable_wandb:
                _, _, renders = mem_evaluate(
                    agent=agent,
                    env=eval_env,
                    hist_length=FLAGS.hist_length,
                    hist_stride=FLAGS.hist_stride,
                    num_eval_episodes=0,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    fixed_env_seed=_fixed_seed,
                    eval_seed=_eval_seed,
                    target_counts=_eval_targets,
                    action_exec_horizon=FLAGS.action_exec_horizon,
                )
                if renders:
                    video = get_wandb_video(renders=renders)
                    wandb.log({'video': video}, step=i)
                    if FLAGS.wandb_mode == 'offline':
                        trigger_sync()

        # Save agent.
        if not FLAGS.debug_train and i % save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
