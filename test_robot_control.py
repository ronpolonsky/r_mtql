#!/usr/bin/env python3
"""Record and replay small real-robot control trajectories.

The rollout service must already be running.  In record mode, a zero policy
action lets the service's SpaceMouse human override provide the actual action.
The script stores the executed action and measured before/after robot states.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expo_ft.env.env_client import EnvClientWrapper


def _state(observation: dict) -> np.ndarray:
    """Return [x, y, z, rx, ry, rz, gripper] from either observation schema."""
    if "robot_state" in observation:
        observation = observation["robot_state"]
    return np.concatenate(
        [
            np.asarray(observation["cartesian_position"], dtype=np.float64).reshape(-1),
            np.asarray(observation["gripper_position"], dtype=np.float64).reshape(-1),
        ]
    )


def _joint_positions(observation: dict) -> np.ndarray:
    if "robot_state" in observation:
        observation = observation["robot_state"]
    return np.asarray(observation["joint_positions"], dtype=np.float64).reshape(-1)


def _make_env(args: argparse.Namespace) -> EnvClientWrapper:
    return EnvClientWrapper(
        env_creation_request={
            "example_action": np.zeros(7, dtype=np.float32),
            "env_usage": "eval",
            "video_dir": "",
        },
        host=args.host,
        port=args.port,
    )


def _reset_if_requested(env: EnvClientWrapper, args: argparse.Namespace) -> None:
    if not args.reset:
        return
    print("Reset requested: returning robot to the configured reset pose...", flush=True)
    observation = env.reset()
    print(
        "Reset complete. Cartesian state:",
        np.round(_state(observation), 6),
        flush=True,
    )


def _record(args: argparse.Namespace) -> None:
    env = _make_env(args)
    _reset_if_requested(env, args)

    period = 1.0 / args.hz
    zero_action = np.zeros(7, dtype=np.float64)
    records = []
    print("Recording SpaceMouse actions. Move the robot, then press Ctrl-C to save.")
    print("Keep your hand near the physical emergency stop.")

    try:
        while args.seconds <= 0 or len(records) < int(args.seconds * args.hz):
            started = time.monotonic()
            before_observation = env.get_observation()
            before = _state(before_observation)
            before_joints = _joint_positions(before_observation)
            executed, action_type = env.step(
                zero_action,
                allow_human_override=True,
                control_gripper=args.control_gripper,
            )
            executed = np.asarray(executed, dtype=np.float64).reshape(-1)
            sent_minus_executed = zero_action - executed
            time.sleep(max(0.0, period - (time.monotonic() - started)))
            after_observation = env.get_observation()
            after = _state(after_observation)
            after_joints = _joint_positions(after_observation)
            measured_delta = after - before
            joint_delta = after_joints - before_joints
            records.append(
                {
                    "index": len(records),
                    "action_type": action_type,
                    "sent_action": zero_action.tolist(),
                    "executed_action": executed.tolist(),
                    "executed_minus_sent": (-sent_minus_executed).tolist(),
                    "before_state": before.tolist(),
                    "after_state": after.tolist(),
                    "measured_delta": measured_delta.tolist(),
                    "before_joint_positions": before_joints.tolist(),
                    "after_joint_positions": after_joints.tolist(),
                    "joint_delta": joint_delta.tolist(),
                    "expected_delta_from_velocity": (
                        np.concatenate([executed[:6] / args.hz, [executed[6] / args.hz]])
                    ).tolist(),
                }
            )
            if len(records) % max(1, int(args.hz)) == 0:
                print(f"Recorded {len(records)} steps; last action_type={action_type}")
    except KeyboardInterrupt:
        print("\nStopping recording.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "control_hz": args.hz,
        "host": args.host,
        "port": args.port,
        "control_gripper": args.control_gripper,
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {len(records)} steps to {output}")


def _replay(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.input).read_text())
    records = payload["records"]
    if args.action_index is not None:
        records = [records[args.action_index]] * args.repeats

    env = _make_env(args)
    _reset_if_requested(env, args)

    period = 1.0 / args.hz
    print(f"Replaying {len(records)} action(s). Press Ctrl-C to stop.")
    for replay_index, record in enumerate(records):
        started = time.monotonic()
        action = np.asarray(record["executed_action"], dtype=np.float64)
        before_observation = env.get_observation()
        before = _state(before_observation)
        before_joints = _joint_positions(before_observation)
        executed, action_type = env.step(
            action,
            allow_human_override=False,
            control_gripper=args.control_gripper,
        )
        executed = np.asarray(executed, dtype=np.float64)
        executed_minus_sent = executed - action
        time.sleep(max(0.0, period - (time.monotonic() - started)))
        after_observation = env.get_observation()
        after = _state(after_observation)
        after_joints = _joint_positions(after_observation)
        measured_delta = after - before
        joint_delta = after_joints - before_joints
        expected_delta = np.concatenate([executed[:6] / args.hz, [executed[6] / args.hz]])
        print(
            f"{replay_index:04d} type={action_type} "
            f"sent={np.round(action, 5)} "
            f"executed={np.round(executed, 5)} "
            f"executed_minus_sent={np.round(executed_minus_sent, 5)} "
            f"joint_delta={np.round(joint_delta, 5)} "
            f"measured_delta={np.round(measured_delta, 5)} "
            f"error={np.round(measured_delta - expected_delta, 5)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("record", "replay"))
    parser.add_argument("--output", default="control_trajectory.json")
    parser.add_argument("--input", default="control_trajectory.json")
    parser.add_argument("--action-index", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--control-gripper", action="store_true")
    args = parser.parse_args()
    if args.hz <= 0 or args.repeats <= 0:
        parser.error("--hz must be positive and --repeats must be positive")
    if args.mode == "record":
        _record(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
