"""Candy Scoop task configuration for DROID teleoperation and collection.

Episodes use terminal-based manual success and reset labels.
"""

import ml_collections
import numpy as np

from configs.task import real_base

try:
    from client.envs.droid_env import CandyScoopEnv
except Exception:
    print("Not importing droid env [module]")


def get_config():
    config = real_base.get_config()

    try:
        config.env = CandyScoopEnv
    except Exception:
        print("Not importing droid env [env]")

    config.env_type = "droid"
    config.env_name = "candy_scoop"
    config.language_instruction = "scoop candy"

    # Controllers are started separately on the NUC; do not relaunch them per run.
    config.launch_controller = False
    config.reset_gripper = False

    # Retract vertically before the joint-space return; no gripper command is sent.
    config.reset_clearance_z = 0.30
    config.reset_clearance_velocity = 0.5
    config.reset_clearance_timeout = 8.0

    # Keep collection images raw; target_count remains available for post hoc cues.
    config.include_target_cue = False

    # Random visual goal: each shuffled block contains targets 1, 2, and 3.
    config.min_target_count = 1
    config.max_target_count = 3
    config.target_seed = 0

    # Verified ZED camera IDs for this robot.
    config.side_camera_id = "38651013_left"
    config.wrist_camera_id = "15577469_left"

    # Averaged from two consecutive stable inspection snapshots on 2026-09-11.
    config.reset_joints = np.array([
        -0.096075,
        -0.211313,
        -0.083119,
        -2.774708,
        -0.044159,
        2.598322,
        -0.174025,
    ], dtype=np.float64)

    # Reuse the validated Pick-task workspace bounds for this robot setup.
    config.bounds = np.array([
        [0.20, 0.68],
        [-0.25, 0.25],
        [0.10, 0.45],
    ], dtype=np.float64)

    # Full Cartesian translation + rotation + gripper action is required.
    config.residual_action_xyzg = False

    # Temporary evaluation timeout; collection disables auto-reset.
    config.auto_reset_steps = 1000

    # SpaceMouse scaling for translation and scoop rotation.
    config.collect_max_lin_vel = 0.5
    config.collect_max_rot_vel = 0.3

    return config
