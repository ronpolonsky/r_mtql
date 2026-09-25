"""Expert-policy demo for HouseEnv.

Runs several episodes of an omniscient BFS-planning expert and saves GIF videos.

Usage:
    python demo_house.py                       # 3 objects, 3 episodes
    python demo_house.py --num_objects 2 --episodes 5 --out_dir demos/
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from envs.house_env import HouseEnv, expert_action


def run_episode(env: HouseEnv, seed: int, frame_skip: int = 2) -> list:
    """Run one expert episode; return list of RGB frames."""
    obs, _ = env.reset(seed=seed)
    frames  = [env.render()]
    done    = False
    step    = 0
    total_r = 0.0

    while not done:
        action = expert_action(env, noise_scale=0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        done     = terminated or truncated
        step    += 1
        if step % frame_skip == 0 or done:
            frames.append(env.render())

    print(
        f"  seed={seed:4d}  steps={step:4d}  "
        f"delivered={info['objects_delivered']}/{info['total_objects']}  "
        f"success={info['success']}  total_reward={total_r:.1f}"
    )
    return frames


def save_gif(frames: list, path: str, fps: int = 15) -> None:
    """Save a list of numpy RGB frames as an animated GIF."""
    images = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    duration_ms = int(1000 / fps)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main():
    parser = argparse.ArgumentParser(description='HouseEnv expert demo')
    parser.add_argument('--num_objects', type=int, default=3,
                        help='Number of objects to deliver')
    parser.add_argument('--episodes',    type=int, default=3,
                        help='Number of demo episodes')
    parser.add_argument('--max_steps',  type=int, default=800,
                        help='Max steps per episode')
    parser.add_argument('--fps',        type=int, default=15,
                        help='GIF frame rate')
    parser.add_argument('--frame_skip', type=int, default=2,
                        help='Render every N steps')
    parser.add_argument('--out_dir',    type=str, default='.',
                        help='Output directory for GIFs')
    parser.add_argument('--seed',       type=int, default=0,
                        help='Base random seed')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = HouseEnv(
        num_objects=args.num_objects,
        max_steps=args.max_steps,
        render_size=500,
    )

    print(f"\nHouseEnv expert demo  |  {args.num_objects} objects  |  {args.episodes} episodes\n")
    print(f"Obs dim:    {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.shape[0]}\n")

    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"Episode {ep + 1}/{args.episodes}  (seed={seed})")
        frames = run_episode(env, seed=seed, frame_skip=args.frame_skip)

        out_path = os.path.join(args.out_dir, f'house_expert_ep{ep+1:02d}_seed{seed}.gif')
        save_gif(frames, out_path, fps=args.fps)
        print(f"  → saved {out_path}  ({len(frames)} frames)\n")

    env.close()
    print("Done.")


if __name__ == '__main__':
    main()
