#!/usr/bin/env python3
"""Replay one recorded Shuffle trajectory through the DROID rollout server.

The default mode only inspects the trajectory.  Pass ``--execute`` to reset
the Shuffle environment and send the recorded seven-dimensional
``action/executed_action`` commands at the collection control rate.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import termios
import time
import tty
from pathlib import Path

import h5py
import numpy as np


DEFAULT_TRAJECTORY = (
    "/iris/u/ronpo/expo-ft-data/shuffle/orange_edited5/"
    "success/0/traj.hdf5"
)


def load_actions(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "action/executed_action" not in handle:
            raise KeyError(f"Missing action/executed_action in {path}")
        actions = np.asarray(handle["action/executed_action"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                f"Expected action/executed_action with shape (T, 7), got "
                f"{actions.shape}."
            )
    if not np.isfinite(actions).all():
        raise ValueError(f"Non-finite action found in {path}")
    return actions


def describe(path: Path, actions: np.ndarray, control_hz: float) -> None:
    gripper = actions[:, 6]
    nonzero = np.flatnonzero(np.abs(gripper) > 1e-5)
    first_nonzero = int(nonzero[0]) if len(nonzero) else None
    print(f"trajectory: {path}")
    print(f"steps: {len(actions)}")
    print(f"duration_s: {len(actions) / control_hz:.2f}")
    print(f"first_nonzero_gripper_step: {first_nonzero}")
    print(f"gripper_values: {np.unique(gripper).tolist()}")
    print(
        "arm_max_abs: "
        + np.array2string(np.max(np.abs(actions[:, :6]), axis=0), precision=6)
    )


def wait_for_space(prompt: str) -> None:
    """Wait for a single SPACE keypress without requiring ENTER."""
    print(prompt, flush=True)
    if not sys.stdin.isatty():
        input()
        return

    file_descriptor = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        while True:
            key = sys.stdin.read(1)
            if key == "\x03":
                raise KeyboardInterrupt
            if key == " ":
                print(flush=True)
                return
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)


def wait_for_label(prompt: str) -> tuple[str, int]:
    """Wait for 1 (correct) or 2 (incorrect) and return label plus code."""
    print(prompt, flush=True)
    if not sys.stdin.isatty():
        while True:
            key = input().strip()
            if key in ("1", "2"):
                break
            print("Invalid input. Type 1 for correct or 2 for incorrect.", flush=True)
    else:
        file_descriptor = sys.stdin.fileno()
        previous_settings = termios.tcgetattr(file_descriptor)
        try:
            tty.setcbreak(file_descriptor)
            while True:
                key = sys.stdin.read(1)
                if key == "\x03":
                    raise KeyboardInterrupt
                if key in ("1", "2"):
                    print(key, flush=True)
                    break
        finally:
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)
    return ("correct", 1) if key == "1" else ("incorrect", 2)


def poll_review_key() -> str | None:
    """Return one pending review key without blocking during replay."""
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    key = sys.stdin.read(1)
    if key == "\x03":
        raise KeyboardInterrupt
    return key if key in ("1", "2") else None


def save_review_result(
    results_path: Path,
    trajectory: Path,
    actions: np.ndarray,
    label: str,
    label_code: int,
    video_dir: str,
    control_hz: float,
    executed_steps: int,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "trajectory": str(trajectory),
        "label": label,
        "label_code": label_code,
        "steps": int(executed_steps),
        "planned_steps": int(len(actions)),
        "duration_s": executed_steps / control_hz,
        "planned_duration_s": len(actions) / control_hz,
        "control_hz": control_hz,
        "video_dir": video_dir,
        "timestamp_unix": time.time(),
    }
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
        handle.flush()


def discover_trajectories(dataset_root: Path) -> list[Path]:
    trajectories = list(dataset_root.glob("success/*/traj.hdf5"))
    trajectories.sort(key=lambda path: int(path.parent.name))
    if not trajectories:
        raise FileNotFoundError(
            f"No success/*/traj.hdf5 trajectories found under {dataset_root}"
        )
    return trajectories


def create_replay_env(host: str, port: int, video_dir: str):
    # Import lazily so dry-run inspection does not need the rollout client
    # dependencies or create a server environment.
    from expo_ft.env.env_client import EnvClientWrapper

    print(f"Connecting to Shuffle rollout server at {host}:{port}...", flush=True)
    try:
        with socket.create_connection((host, port), timeout=3.0):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Could not connect to rollout server at {host}:{port}. Start "
            "run_client.py with configs/task/shuffle.py first."
        ) from exc

    env = EnvClientWrapper(
        env_creation_request={
            "example_action": np.zeros((1, 7), dtype=np.float32),
            "env_usage": "replay",
            "video_dir": video_dir,
        },
        host=host,
        port=port,
    )
    task_description = str(env.task_description).strip().lower()
    if task_description != "shuffle":
        raise RuntimeError(
            "Refusing replay: rollout server reported task description "
            f"{env.task_description!r}, expected 'shuffle'. Restart the "
            "server with configs/task/shuffle.py."
        )
    return env


def reset_replay_env(env, attempts: int = 3):
    """Reset the existing server environment without generic env recreation."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            observation, _ = env.client.reset(env.env_id)
            required_keys = ("exterior_image_1_left", "wrist_image_left")
            missing = [key for key in required_keys if key not in observation]
            if missing:
                raise RuntimeError(
                    f"reset returned an incomplete observation; missing {missing}"
                )
            return observation
        except (RuntimeError, KeyError) as exc:
            last_error = exc
            print(
                f"Shuffle reset attempt {attempt}/{attempts} failed: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(2.0)
    raise RuntimeError(
        "Shuffle reset failed after retries. Restart the rollout server and "
        "verify both ZED cameras before replaying again."
    ) from last_error


def replay(
    env,
    actions: np.ndarray,
    *,
    host: str,
    port: int,
    control_hz: float,
    video_dir: str,
    print_actions: bool,
    wait_to_start: bool,
) -> tuple[tuple[str, int] | None, int]:
    try:
        # ShuffleEnv.reset() returns the robot to the configured Shuffle base
        # joints and opens the gripper before any replay action is sent.
        observation = reset_replay_env(env)
        print("Shuffle base reset complete; no replay action sent yet.", flush=True)

        video_frames = {
            "exterior_image_1_left": [],
            "wrist_image_left": [],
        }

        def record_video_observation(obs) -> None:
            if not video_dir:
                return
            for key in video_frames:
                if key in obs:
                    video_frames[key].append(np.asarray(obs[key], dtype=np.uint8))

        record_video_observation(observation)
        if wait_to_start:
            wait_for_space(
                "Reset complete. Press SPACE to begin this recorded trajectory, "
                "or Ctrl-C to abort."
            )
        else:
            print("Reset complete. Starting the next trajectory automatically.", flush=True)

        review = None
        executed_steps = 0
        terminal_settings = None
        if sys.stdin.isatty():
            file_descriptor = sys.stdin.fileno()
            terminal_settings = termios.tcgetattr(file_descriptor)
            tty.setcbreak(file_descriptor)

        try:
            period = 1.0 / control_hz
            next_tick = time.monotonic()
            for index, action in enumerate(actions):
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)

                env.step(
                    action.tolist(),
                    allow_human_override=False,
                    control_gripper=True,
                )
                observation = env.get_observation()
                record_video_observation(observation)
                executed_steps = index + 1
                if review is None:
                    key = poll_review_key()
                    if key is not None:
                        review = ("correct", 1) if key == "1" else ("incorrect", 2)
                        print(
                            f"Early review captured: {review[0]}. "
                            "Stopping this trajectory and advancing after saving.",
                            flush=True,
                        )
                if print_actions or index < 3 or index % 10 == 0 or index == len(actions) - 1:
                    gripper_position = float(
                        np.asarray(observation["gripper_position"]).reshape(-1)[0]
                    )
                    cartesian_position = np.asarray(
                        observation["cartesian_position"], dtype=np.float64
                    ).reshape(-1)
                    print(
                        f"step {index + 1}/{len(actions)}: "
                        f"action={np.array2string(action, precision=6, suppress_small=False)} "
                        f"physical_cartesian_position="
                        f"{np.array2string(cartesian_position, precision=6, suppress_small=False)} "
                        f"gripper_action={action[6]:.6f} "
                        f"physical_gripper_position={gripper_position:.6f}",
                        flush=True,
                    )
                next_tick += period
                if review is not None:
                    break
        finally:
            if terminal_settings is not None:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, terminal_settings
                )

        if video_dir:
            import imageio.v2 as imageio

            output_dir = Path(video_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for key, frames in video_frames.items():
                if not frames:
                    continue
                output_path = output_dir / f"replay_{key}.mp4"
                imageio.mimsave(
                    output_path,
                    frames,
                    fps=control_hz,
                    macro_block_size=1,
                )
                print(
                    f"Saved replay video: {output_path} "
                    f"({len(frames)} frames)",
                    flush=True,
                )
        return review, executed_steps
    finally:
        # The server owns the robot connection; there is no local robot handle
        # to close here.  The environment remains available for the operator's
        # normal manual reset/cleanup procedure.
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default=DEFAULT_TRAJECTORY)
    parser.add_argument(
        "--dataset_root",
        default="",
        help=(
            "Replay success/*/traj.hdf5 sequentially from this processed "
            "training-data directory."
        ),
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=None,
        help="For --dataset_root, start at this numeric success episode.",
    )
    parser.add_argument("--client_host", default="localhost")
    parser.add_argument("--client_port", type=int, default=8102)
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument(
        "--video_dir",
        default="",
        help="Optional server-side directory for the replay video.",
    )
    parser.add_argument(
        "--results_path",
        default="",
        help="Append one JSON review record per completed trajectory to this file.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reset the server environment and replay actions physically.",
    )
    parser.add_argument(
        "--print_actions",
        action="store_true",
        help="Print every replayed 7D action and observed gripper position.",
    )
    args = parser.parse_args()
    if args.control_hz <= 0:
        parser.error("--control_hz must be positive")

    if args.dataset_root and args.trajectory != DEFAULT_TRAJECTORY:
        parser.error("Use either --trajectory or --dataset_root, not both.")

    if args.dataset_root:
        trajectories = discover_trajectories(Path(args.dataset_root))
        if args.start_episode is not None:
            trajectories = [
                path
                for path in trajectories
                if int(path.parent.name) >= args.start_episode
            ]
            if not trajectories:
                parser.error(
                    f"No trajectories at or after success/{args.start_episode} "
                    f"under {args.dataset_root}"
                )
    else:
        if args.start_episode is not None:
            parser.error("--start_episode requires --dataset_root")
        trajectories = [Path(args.trajectory)]

    loaded_trajectories = [(path, load_actions(path)) for path in trajectories]
    for trajectory, actions in loaded_trajectories:
        describe(trajectory, actions, args.control_hz)
    if not args.execute:
        print("dry-run only; pass --execute to perform the physical replay")
        return

    if len(loaded_trajectories) > 1 and not args.results_path:
        parser.error("--results_path is required when replaying a dataset")

    results_path = Path(args.results_path) if args.results_path else None
    replay_env = create_replay_env(
        args.client_host,
        args.client_port,
        args.video_dir,
    )

    for index, (trajectory, actions) in enumerate(loaded_trajectories):
        if args.video_dir and len(loaded_trajectories) > 1:
            video_dir = str(
                Path(args.video_dir)
                / f"{trajectory.parent.parent.parent.name}_success_{trajectory.parent.name}"
            )
        else:
            video_dir = args.video_dir

        print(
            f"\n=== Replaying training trajectory {index + 1}/"
            f"{len(loaded_trajectories)}: {trajectory} ===",
            flush=True,
        )
        review, executed_steps = replay(
            replay_env,
            actions,
            host=args.client_host,
            port=args.client_port,
            control_hz=args.control_hz,
            video_dir=video_dir,
            print_actions=args.print_actions,
            wait_to_start=index == 0,
        )
        if results_path is not None and review is None:
            label, label_code = wait_for_label(
                "Episode finished. Press 1 for CORRECT or 2 for INCORRECT."
            )
        elif results_path is not None:
            label, label_code = review
        if results_path is not None:
            save_review_result(
                results_path,
                trajectory,
                actions,
                label,
                label_code,
                video_dir,
                args.control_hz,
                executed_steps,
            )
            print(
                f"Saved review: {label} -> {results_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
