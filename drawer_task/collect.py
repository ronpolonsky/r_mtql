#!/usr/bin/env python3
"""Collect the isolated four-cube visual-cue drawer retrieval dataset.

Each episode contains four differently colored cubes, one in each drawer.
Cube 0 is the target; its color and drawer are randomized with the episode
seed.  A rendered cube cue fades out over the first ``cue_steps`` control
steps.  Success requires placing the target cube on the table.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANISKILL_ROOT = PROJECT_ROOT / "cabinet-memory-sim" / "ManiSkill"
if str(MANISKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MANISKILL_ROOT))

import mani_skill.envs  # noqa: F401  (register environments)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from solve_cabinet_panda import (
    close_bottom_drawer,
    close_top_drawer,
    open_bottom_drawer,
    open_top_drawer,
    pick_cube,
    place_cube_on_table,
)


DRAWERS = ("top_1", "bottom_1", "top_2", "bottom_2")
ARM_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
ARM_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)
ACTION_LOWER = np.concatenate([ARM_LOWER, np.array([-1.0], np.float32)])
ACTION_UPPER = np.concatenate([ARM_UPPER, np.array([1.0], np.float32)])
COLOR_RGB = {
    "blue": (40, 90, 230),
    "green": (40, 190, 40),
    "lime green": (80, 230, 80),
    "orange": (240, 140, 25),
    "purple": (180, 55, 185),
    "red": (230, 40, 40),
    "yellow": (240, 225, 35),
}


class StepLimitReached(RuntimeError):
    pass


def normalize_action(action) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:8]
    norm = 2.0 * (action - ACTION_LOWER) / (ACTION_UPPER - ACTION_LOWER + 1e-8) - 1.0
    return np.clip(norm, -1.0, 1.0).astype(np.float32)


def drawer_key(assignment) -> str:
    cabinet_index, drawer_type, _ = assignment
    return f"{drawer_type}_{cabinet_index + 1}"


def _shade(color, scale):
    return tuple(int(np.clip(channel * scale, 0, 255)) for channel in color)


def add_cube_cue(image: np.ndarray, color_name: str, alpha: float) -> np.ndarray:
    """Overlay a shaded cube icon, alpha-blended into the camera observation."""
    if alpha <= 0:
        return image
    overlay = image.copy()
    color = COLOR_RGB.get(color_name, (220, 220, 220))
    # A compact isometric cube without language or a flat color-card cue.
    top = np.array([[18, 16], [34, 8], [50, 16], [34, 25]], np.int32)
    left = np.array([[18, 16], [34, 25], [34, 47], [18, 37]], np.int32)
    right = np.array([[34, 25], [50, 16], [50, 37], [34, 47]], np.int32)
    cv2.fillConvexPoly(overlay, top, _shade(color, 1.15))
    cv2.fillConvexPoly(overlay, left, _shade(color, 0.72))
    cv2.fillConvexPoly(overlay, right, color)
    cv2.polylines(overlay, [top, left, right], True, (20, 20, 20), 1)
    return cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0)


class RecordingEnv:
    """Proxy recording one two-camera state per 10 Hz control step."""

    def __init__(self, env, target_color: str, cue_steps: int, max_steps: int):
        self.env = env
        self.target_color = target_color
        self.cue_steps = int(cue_steps)
        self.max_steps = int(max_steps)
        self.actions: list[np.ndarray] = []
        self.states = [self._capture(0)]

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _capture(self, state_index: int) -> dict[str, np.ndarray]:
        base = self.env.unwrapped
        agent = base.render_rgb_array("render_camera").cpu().numpy()[0].astype(np.uint8)
        wrist = base.render_rgb_array("wrist_camera").cpu().numpy()[0].astype(np.uint8)
        if self.cue_steps > 0 and state_index <= self.cue_steps:
            alpha = max(0.0, 1.0 - state_index / self.cue_steps)
            agent = add_cube_cue(agent, self.target_color, alpha)
            wrist = add_cube_cue(wrist, self.target_color, alpha)
        proprio = base.agent.robot.get_qpos().cpu().numpy().reshape(-1).astype(np.float32)
        return {"image": agent, "wrist_image": wrist, "proprio": proprio}

    def step(self, action):
        if len(self.actions) >= self.max_steps:
            raise StepLimitReached(f"maximum {self.max_steps} control steps reached")
        result = self.env.step(action)
        self.actions.append(normalize_action(action))
        self.states.append(self._capture(len(self.actions)))
        return result

    def hold_open_gripper(self, steps: int):
        for _ in range(steps):
            arm = self.unwrapped.agent.robot.get_qpos().cpu().numpy()[0, :7]
            self.step(np.concatenate([arm, np.array([1.0], dtype=np.float32)]))


class EpisodeRunner:
    def __init__(self, seed: int, cue_steps: int, max_steps: int):
        raw = gym.make(
            "OpenCabinetPanda-v0",
            render_mode="rgb_array",
            control_mode="pd_joint_pos",
            shuffle_colors=True,
            color_shuffle_seed=seed,
            num_cubes=4,
            max_episode_steps=max_steps + 1,
            width=128,
            height=128,
            randomize_cabinet_pose=False,
        )
        raw.reset(seed=seed)
        self.raw = raw
        self.seed = int(seed)
        self.target_cube_index = 0
        self.target_color = str(raw.unwrapped.cube_colors[self.target_cube_index])
        self.assignments = list(raw.unwrapped.cube_slot_assignments)
        self.target_drawer = drawer_key(self.assignments[self.target_cube_index])
        self.env = RecordingEnv(raw, self.target_color, cue_steps, max_steps)
        self.planner = PandaArmMotionPlanningSolver(
            self.env,
            debug=False,
            vis=False,
            base_pose=raw.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        self.handles: dict[str, np.ndarray] = {}
        self.high_level_actions: list[str] = []
        self.target_open_step: int | None = None

    def close(self):
        self.raw.close()

    def cube_drawer(self, cube_index: int) -> str:
        return drawer_key(self.assignments[cube_index])

    def open_drawer(self, drawer: str, stay_at_drawer: bool = False):
        drawer_type, cabinet_number = drawer.rsplit("_", 1)
        self.raw.unwrapped.set_active_cabinet(int(cabinet_number) - 1)
        if drawer_type == "top":
            handle = open_top_drawer(
                self.env, self.planner, return_home=not stay_at_drawer
            )
        else:
            handle = open_bottom_drawer(
                self.env, self.planner, return_home=not stay_at_drawer
            )
        self.handles[drawer] = handle
        self.high_level_actions.append(f"open_{drawer}")
        if drawer == self.target_drawer:
            self.target_open_step = len(self.env.actions) - 1

    def close_drawer(self, drawer: str):
        if drawer.startswith("top"):
            close_top_drawer(self.env, self.planner, self.handles[drawer])
        else:
            close_bottom_drawer(self.env, self.planner, self.handles[drawer])
        self.high_level_actions.append(f"close_{drawer}")

    def pick_and_place(self, cube_index: int, drawer: str):
        pick_cube(
            self.env,
            self.planner,
            cube_idx=cube_index,
            is_bottom_drawer=drawer.startswith("bottom"),
        )
        self.high_level_actions.append(f"pick_cube_{cube_index}")
        placed = place_cube_on_table(self.env, self.planner)
        if placed is not True:
            raise RuntimeError("place_cube_on_table did not complete")
        self.high_level_actions.append(f"place_cube_{cube_index}_on_table")

    def run_success(self, global_index: int, rng: random.Random) -> str:
        distractors = [drawer for drawer in DRAWERS if drawer != self.target_drawer]
        rng.shuffle(distractors)
        mode = global_index % 5
        if mode < 4:
            target_rank = mode + 1
            search = distractors[: target_rank - 1] + [self.target_drawer]
            strategy = f"target_rank_{target_rank}"
        else:
            # A successful but inefficient path.  This gives BC undesirable
            # behavior to imitate while reward-aware learning can rank it below
            # the efficient paths.
            search = distractors + [distractors[0], self.target_drawer]
            strategy = "target_last_with_revisit"

        for drawer in search:
            is_target = drawer == self.target_drawer
            self.open_drawer(drawer, stay_at_drawer=is_target)
            if is_target:
                self.pick_and_place(self.target_cube_index, drawer)
                break
            self.close_drawer(drawer)
        return strategy

    def run_failure(self, global_index: int, rng: random.Random) -> str:
        mode = global_index % 2
        if mode == 0:
            wrong_index = rng.choice([1, 2, 3])
            drawer = self.cube_drawer(wrong_index)
            self.open_drawer(drawer, stay_at_drawer=True)
            pick_cube(
                self.env,
                self.planner,
                cube_idx=wrong_index,
                is_bottom_drawer=drawer.startswith("bottom"),
            )
            self.high_level_actions.append(f"pick_wrong_cube_{wrong_index}")
            return "wrong_object_grasp"

        distractors = [drawer for drawer in DRAWERS if drawer != self.target_drawer]
        rng.shuffle(distractors)
        strategy = "revisit_timeout"
        cycle_index = 0
        try:
            while True:
                drawer = distractors[cycle_index % len(distractors)]
                self.open_drawer(drawer)
                self.close_drawer(drawer)
                cycle_index += 1
        except StepLimitReached:
            self.high_level_actions.append("timeout")
        return strategy


def save_episode(
    runner: EpisodeRunner,
    path: Path,
    success: bool,
    strategy: str,
    split: str,
    step_penalty: float,
):
    actions = np.asarray(runner.env.actions, dtype=np.float32)
    states = runner.env.states
    if len(actions) == 0 or len(states) != len(actions) + 1:
        raise RuntimeError("invalid recorded episode lengths")

    rewards = np.full(len(actions), step_penalty, dtype=np.float32)
    if runner.target_open_step is not None:
        rewards[runner.target_open_step] += np.float32(0.25)
    rewards[-1] = np.float32(0.75 if success else -1.0)
    terminals = np.zeros(len(actions), dtype=np.float32)
    terminals[-1] = 1.0
    masks = np.ones(len(actions), dtype=np.float32)
    masks[-1] = 0.0
    actor_mask = np.full(len(actions), 1.0 if success else 0.0, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs_image=np.stack([state["image"] for state in states[:-1]]),
        obs_wrist_image=np.stack([state["wrist_image"] for state in states[:-1]]),
        obs_proprio=np.stack([state["proprio"] for state in states[:-1]]),
        next_obs_image=np.stack([state["image"] for state in states[1:]]),
        next_obs_wrist_image=np.stack([state["wrist_image"] for state in states[1:]]),
        next_obs_proprio=np.stack([state["proprio"] for state in states[1:]]),
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        masks=masks,
        actor_mask=actor_mask,
        meta_success=np.asarray(success, dtype=np.bool_),
        meta_failure_reason=np.asarray("" if success else strategy),
        meta_target_color=np.asarray(runner.target_color),
        meta_target_drawer=np.asarray(runner.target_drawer),
    )
    return {
        "path": str(path),
        "split": split,
        "success": bool(success),
        "failure_reason": "" if success else strategy,
        "seed": runner.seed,
        "target_color": runner.target_color,
        "target_drawer": runner.target_drawer,
        "strategy": strategy,
        "length": len(actions),
        "return": float(rewards.sum()),
        "high_level_actions": runner.high_level_actions,
    }


def collect_one(
    output_dir: Path,
    success: bool,
    global_index: int,
    base_seed: int,
    cue_steps: int,
    max_steps: int,
    step_penalty: float,
    max_attempts: int,
):
    for attempt in range(max_attempts):
        seed = base_seed + global_index * 100 + attempt
        runner = None
        try:
            runner = EpisodeRunner(seed, cue_steps, max_steps)
            runner.env.hold_open_gripper(cue_steps)
            rng = random.Random(seed + 10_000)
            strategy = (
                runner.run_success(global_index, rng)
                if success
                else runner.run_failure(global_index, rng)
            )
            if success and len(runner.env.actions) >= max_steps:
                raise RuntimeError("successful trajectory reached the step limit")
            outcome = "success" if success else "failure"
            path = output_dir / "episodes" / f"{outcome}_{global_index:05d}.npz"
            return save_episode(
                runner, path, success, strategy, "train", step_penalty
            )
        except Exception as exc:
            print(
                f"retry success={success} index={global_index} seed={seed}: {exc}",
                flush=True,
            )
            traceback.print_exc()
        finally:
            if runner is not None:
                runner.close()
    raise RuntimeError(f"could not collect success={success} index={global_index}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--successes", type=int, default=300)
    parser.add_argument("--failures", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=840_000)
    parser.add_argument("--cue-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--step-penalty", type=float, default=-0.0001)
    parser.add_argument("--max-attempts", type=int, default=12)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    manifest = []
    for success, count, seed_offset in (
        (True, args.successes, 0),
        (False, args.failures, 100_000_000),
    ):
        for index in range(args.shard_index, count, args.num_shards):
            manifest.append(
                collect_one(
                    args.output_dir,
                    success,
                    index,
                    args.base_seed + seed_offset,
                    args.cue_steps,
                    args.max_steps,
                    args.step_penalty,
                    args.max_attempts,
                )
            )

    manifest_path = args.output_dir / "manifests" / f"manifest_{args.shard_index:02d}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path} with {len(manifest)} episodes", flush=True)


if __name__ == "__main__":
    main()
