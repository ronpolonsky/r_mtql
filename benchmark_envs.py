"""Benchmark BatchedHouseEnv vs SubprocVecEnv vs DummyVecEnv.

Usage:
    python benchmark_envs.py
    python benchmark_envs.py --num_envs=4,16,64 --steps=2000
"""
import time
import argparse
import numpy as np

WARMUP_STEPS = 200
ENV_NAME     = 'house-n3-v0'


def make_subproc(num_envs, count_reward=False, hist_length=0, hist_stride=1):
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from envs.house_env import make_house_env
    from ppo_main import HistoryWrapper

    def _fn():
        env = make_house_env(ENV_NAME, count_reward=count_reward)
        if hist_length > 0:
            env = HistoryWrapper(env, hist_length, hist_stride)
        return env

    fns = [_fn] * num_envs
    return SubprocVecEnv(fns) if num_envs > 1 else DummyVecEnv(fns)


def make_dummy(num_envs, count_reward=False, hist_length=0, hist_stride=1):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from envs.house_env import make_house_env
    from ppo_main import HistoryWrapper

    def _fn():
        env = make_house_env(ENV_NAME, count_reward=count_reward)
        if hist_length > 0:
            env = HistoryWrapper(env, hist_length, hist_stride)
        return env

    return DummyVecEnv([_fn] * num_envs)


def make_batched(num_envs, count_reward=False, hist_length=0, hist_stride=1):
    from envs.batched_house_env import make_batched_house_env
    return make_batched_house_env(
        ENV_NAME,
        num_envs=num_envs,
        count_reward=count_reward,
        hist_length=hist_length,
        hist_stride=hist_stride,
    )


def run_bench(env, steps: int) -> float:
    """Return steps-per-second (env-steps, not wall-clock calls)."""
    N = env.num_envs
    env.reset()
    actions = np.stack([env.action_space.sample() for _ in range(N)])

    # Warmup
    for _ in range(WARMUP_STEPS):
        env.step(actions)

    env.reset()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    elapsed = time.perf_counter() - t0

    env.close()
    return (steps * N) / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', default='1,4,16,64',
                        help='Comma-separated list of num_envs to benchmark')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Number of step() calls per benchmark run')
    parser.add_argument('--hist_length', type=int, default=0,
                        help='History length (0 = no history)')
    parser.add_argument('--hist_stride', type=int, default=1)
    parser.add_argument('--skip_subproc', action='store_true',
                        help='Skip SubprocVecEnv (faster to run, no fork overhead)')
    args = parser.parse_args()

    env_counts = [int(x) for x in args.num_envs.split(',')]
    hl, hs     = args.hist_length, args.hist_stride

    header = f"{'num_envs':>10}  {'DummyVecEnv':>14}  {'BatchedHouseEnv':>16}"
    if not args.skip_subproc:
        header += f"  {'SubprocVecEnv':>14}"
    header += f"  {'speedup_vs_dummy':>17}  {'speedup_vs_subproc':>18}"
    print(header)
    print('-' * len(header))

    for n in env_counts:
        dummy_sps   = run_bench(make_dummy(n, hist_length=hl, hist_stride=hs), args.steps)
        batched_sps = run_bench(make_batched(n, hist_length=hl, hist_stride=hs), args.steps)

        row = (f"{n:>10}  {dummy_sps:>14,.0f}  {batched_sps:>16,.0f}")

        if not args.skip_subproc:
            subproc_sps = run_bench(make_subproc(n, hist_length=hl, hist_stride=hs), args.steps)
            row += f"  {subproc_sps:>14,.0f}"
            speedup_sub = batched_sps / subproc_sps
        else:
            subproc_sps = None
            speedup_sub = float('nan')

        speedup_dummy = batched_sps / dummy_sps
        row += f"  {speedup_dummy:>17.2f}x"
        row += f"  {speedup_sub:>18.2f}x" if not args.skip_subproc else ""
        print(row)


if __name__ == '__main__':
    main()
