"""Egg task configuration for fast manual DROID inspection.

This is a separate manual arm-control profile.  It reuses the Candy Scoop
robot workspace and camera layout but does not change the Candy Scoop task.
Use with ``client.inspect_robot_state`` and ``--allow_gripper=false``.
"""

import ml_collections
import numpy as np

from configs.task import real_base

try:
    from client.envs.droid_env import EggEnv
except Exception:
    print("Not importing droid env [module]")


def get_config():
    config = real_base.get_config()

    try:
        config.env = EggEnv
    except Exception:
        print("Not importing droid env [env]")

    config.env_type = "droid"
    config.env_name = "egg"
    config.language_instruction = "make an egg"
    # Override per collection session with --task_config.card_color=white/black.
    config.card_color = "unset"
    config.control_gripper = True

    # Controllers are started separately on the NUC.
    config.launch_controller = False
    config.reset_gripper = True

    # Same physical robot workspace and camera IDs as Candy Scoop.
    config.side_camera_id = "38651013_left"
    config.wrist_camera_id = "15577469_left"
    config.reset_joints = np.array([
        0.042359,
        -0.310721,
        0.066479,
        -2.774950,
        0.026206,
        2.326404,
        0.095418,
    ], dtype=np.float64)
    config.bounds = np.array([
        [0.10, 0.70],
        [-0.30, 0.40],
        [0.01, 0.40],
    ], dtype=np.float64)

    # Faster manual SpaceMouse scaling than Candy Scoop (0.5 / 0.3).
    config.collect_max_lin_vel = 0.75
    config.collect_max_rot_vel = 0.75

    config.residual_action_xyzg = False
    config.auto_reset_steps = 1000

    return config
