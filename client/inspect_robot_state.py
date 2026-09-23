"""Print live DROID robot state while optionally moving with a SpaceMouse.

This is intentionally separate from data collection: it does not write HDF5,
invoke task success detectors, or call ``env.reset()``.  The arm remains under
manual SpaceMouse control and the gripper is enabled by default.

Run from the EXPO-FT root, for example:

    client/.venv/bin/python -m client.inspect_robot_state \
        --task_config=configs/task/pick.py

Stop with Ctrl-C.  Use the printed ``joint_positions`` for ``reset_joints`` and
the printed Cartesian XYZ values to choose conservative task bounds.
"""

from __future__ import annotations

import json
import os
import select
import termios
import time
import tty

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from client.real_utils.spacemouse import SpaceMousePolicy


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "task_config",
    "configs/task/pick.py",
    "Task configuration used to construct the DROID environment.",
    lock_config=False,
)
flags.DEFINE_float(
    "print_interval",
    0.5,
    "Seconds between printed robot-state samples.",
)
flags.DEFINE_bool(
    "move",
    True,
    "Enable SpaceMouse Cartesian motion. Use --move=false to monitor only.",
)
flags.DEFINE_bool(
    "allow_gripper",
    True,
    "Pass SpaceMouse gripper commands through; use --allow_gripper=false to hold it still.",
)
flags.DEFINE_bool(
    "launch_controller",
    False,
    "Launch/restart the NUC robot and gripper controllers (disabled by default).",
)
flags.DEFINE_bool(
    "print_actions",
    True,
    "Print nonzero Cartesian SpaceMouse commands while moving.",
)
flags.DEFINE_string(
    "capture_file",
    "/dev/shm/expo-ft-robot-state-captures.jsonl",
    "JSONL file receiving states captured by pressing SPACE.",
)
flags.DEFINE_bool(
    "space_resets_to_base",
    False,
    "When true, SPACE returns the robot to the task config reset_joints instead of saving a snapshot.",
)



_SKIP = object()


def _is_image_key(key) -> bool:
    """Return whether a nested observation key contains image-like data."""
    if key is None:
        return False
    name = str(key).lower()
    return (
        name in {"image", "images", "depth", "pointcloud", "point_cloud"}
        or "image" in name
        or "pointcloud" in name
        or "point_cloud" in name
    )


def _jsonable(value, key=None):
    """Convert DROID observations/actions to JSON while dropping image arrays."""
    if _is_image_key(key):
        return _SKIP
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            converted = _jsonable(child_value, key=child_key)
            if converted is not _SKIP:
                result[str(child_key)] = converted
        return result
    if isinstance(value, (list, tuple)):
        return [
            converted
            for child_value in value
            for converted in [_jsonable(child_value)]
            if converted is not _SKIP
        ]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), key=key)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), key=key)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return _jsonable(value.item(), key=key)
    except (AttributeError, ValueError, TypeError):
        return str(value)

def _print_state(raw_observation: dict) -> None:
    robot_state = raw_observation["robot_state"]
    joints = np.asarray(robot_state["joint_positions"], dtype=np.float64)
    cartesian = np.asarray(robot_state["cartesian_position"], dtype=np.float64)
    gripper = np.asarray(robot_state["gripper_position"], dtype=np.float64)
    print(
        "joint_positions="
        + np.array2string(joints, precision=6, separator=", ")
        + "  xyz_m=(x={:.6f}, y={:.6f}, z={:.6f})".format(
            cartesian[0], cartesian[1], cartesian[2]
        )
        + "  rotation="
        + np.array2string(cartesian[3:], precision=6, separator=", ")
        + "  gripper_position="
        + np.array2string(gripper, precision=6, separator=", "),
        flush=True,
    )


def _state_record(
    raw_observation: dict,
    transformed_observation: dict | None = None,
    action_record: dict | None = None,
    inspector_config: dict | None = None,
) -> dict:
    """Build a complete non-image observation/action snapshot."""
    raw = _jsonable(raw_observation)
    record = {
        "timestamp": time.time(),
        "observation": raw,
    }
    if transformed_observation is not None:
        record["policy_observation"] = _jsonable(transformed_observation)
    if action_record is not None:
        record["action"] = _jsonable(action_record)
    if inspector_config is not None:
        record["inspector_config"] = _jsonable(inspector_config)

    # Keep the most-used fields at the top level for quick inspection and
    # compatibility with the older capture-file format.
    robot_state = raw_observation.get("robot_state", {})
    for field in ("joint_positions", "joint_velocities", "cartesian_position", "gripper_position"):
        if field in robot_state:
            record[field] = _jsonable(robot_state[field])
    return record


def _open_capture_terminal():
    """Put the controlling terminal in single-key mode, if available."""
    try:
        terminal = open("/dev/tty", "r")
        original = termios.tcgetattr(terminal.fileno())
        tty.setcbreak(terminal.fileno())
        return terminal, original
    except (OSError, termios.error):
        return None, None


def _restore_capture_terminal(terminal, original) -> None:
    if terminal is None or original is None:
        return
    try:
        termios.tcsetattr(terminal.fileno(), termios.TCSADRAIN, original)
    finally:
        terminal.close()


def _capture_if_requested(
    terminal,
    raw_observation,
    capture_handle,
    transformed_observation=None,
    action_record=None,
    inspector_config=None,
    reset_to_base=None,
) -> None:
    if terminal is None:
        return
    ready, _, _ = select.select([terminal], [], [], 0.0)
    if not ready:
        return
    key = terminal.read(1)
    if key != " ":
        return

    if reset_to_base is not None:
        print("[RESET] Returning robot to task base joints...", flush=True)
        reset_to_base()
        print("[RESET COMPLETE] Robot is at task base joints.", flush=True)
        return

    record = _state_record(
        raw_observation,
        transformed_observation=transformed_observation,
        action_record=action_record,
        inspector_config=inspector_config,
    )
    capture_handle.write(json.dumps(record) + "\n")
    capture_handle.flush()
    print(f"[SNAPSHOT SAVED] {capture_handle.name}", flush=True)
    print("[SNAPSHOT DETAILS - images omitted]")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
    if FLAGS.print_interval <= 0:
        raise ValueError("--print_interval must be positive.")

    task_config = FLAGS.task_config
    env_kwargs = dict(task_config)
    env_kwargs["launch_controller"] = FLAGS.launch_controller
    env = task_config.env(**env_kwargs)
    controller = SpaceMousePolicy(
        max_lin_vel=float(task_config.get("collect_max_lin_vel", 0.2)),
        max_rot_vel=float(task_config.get("collect_max_rot_vel", 0.1)),
    )

    period = 1.0 / float(task_config.get("control_hz", 10.0))
    next_print = 0.0
    next_action_print = 0.0
    inspector_config = {
        "action_space": task_config.get("action_space", "cartesian_velocity"),
        "gripper_action_space": task_config.get("gripper_action_space", "velocity"),
        "control_hz": task_config.get("control_hz", 10.0),
        "image_size": task_config.get("image_size", None),
        "side_camera_id": task_config.get("side_camera_id", None),
        "wrist_camera_id": task_config.get("wrist_camera_id", None),
        "reset_joints": getattr(env, "reset_joints", None),
        "bounds": getattr(env, "bounds", None),
        "move": FLAGS.move,
        "allow_gripper": FLAGS.allow_gripper,
        "launch_controller": FLAGS.launch_controller,
        "space_resets_to_base": FLAGS.space_resets_to_base,
    }
    os.makedirs(os.path.dirname(FLAGS.capture_file) or ".", exist_ok=True)
    capture_terminal, original_terminal = _open_capture_terminal()
    capture_handle = open(FLAGS.capture_file, "a", buffering=1)
    print("State inspector started. The arm will not reset automatically.")
    if FLAGS.space_resets_to_base:
        print("Move with the SpaceMouse; press SPACE to return to task base joints; Ctrl-C to stop.")
    else:
        print("Move with the SpaceMouse; press SPACE to capture a full state/action snapshot; Ctrl-C to stop.")
    print(f"Captures are written to {FLAGS.capture_file}")
    if capture_terminal is None:
        print("Warning: no controlling TTY; SPACE capture is unavailable.")

    try:
        while True:
            loop_start = time.monotonic()
            raw_observation = env.get_raw_observation()
            # This is the non-image observation structure presented to the
            # policy/data writer, including prompt and target_count.
            policy_observation = env.transform_observation(raw_observation)

            now = time.monotonic()
            if now >= next_print:
                _print_state(raw_observation)
                next_print = now + FLAGS.print_interval

            action_record = None
            if FLAGS.move:
                action = np.asarray(
                    controller.forward(raw_observation), dtype=np.float64
                )
                requested_action = action.copy()
                if not FLAGS.allow_gripper:
                    action[-1] = 0.0
                if (
                    FLAGS.print_actions
                    and np.linalg.norm(action[:6]) > 1e-4
                    and now >= next_action_print
                ):
                    print(
                        "SPACEMOUSE COMMAND "
                        "[vx, vy, vz, wx, wy, wz, gripper]="
                        + np.array2string(
                            action, precision=5, separator=", "
                        ),
                        flush=True,
                    )
                    next_action_print = now + FLAGS.print_interval
                if FLAGS.allow_gripper:
                    action_info = env.step(action)
                else:
                    # Build the same action dictionary as the dataset while
                    # sending only the 6D arm command to the robot.
                    action_info = env._robot.create_action_dict(
                        action,
                        action_space=task_config.get("action_space", "cartesian_velocity"),
                    )
                    action_info["executed_action"] = action.copy()
                    env._robot.update_pose(action[:6], velocity=True, blocking=False)
                action_record = {
                    "requested_action": requested_action,
                    "executed_action": action.copy(),
                    "action_info": action_info,
                    "space_mouse_movement_enabled": controller.movement_enabled,
                }

            # Pair the current observation with the command just constructed.
            _capture_if_requested(
                capture_terminal,
                raw_observation,
                capture_handle,
                transformed_observation=policy_observation,
                action_record=action_record,
                inspector_config=inspector_config,
                reset_to_base=env.reset if FLAGS.space_resets_to_base else None,
            )

            remaining = period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopping state inspector.")
    finally:
        # Send one zero Cartesian command so the last SpaceMouse command is not
        # left active while the process exits. Keep the gripper unchanged.
        try:
            if FLAGS.allow_gripper:
                env.step(np.zeros(7, dtype=np.float64))
            else:
                env._robot.update_pose(
                    np.zeros(6, dtype=np.float64),
                    velocity=True,
                    blocking=False,
                )
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        capture_handle.close()
        _restore_capture_terminal(capture_terminal, original_terminal)


if __name__ == "__main__":
    app.run(main)
