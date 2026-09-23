"""Egg v2 task configuration with a new base pose."""

import numpy as np

from configs.task import egg


def get_config():
    config = egg.get_config()
    config.env_name = "egg_v2"
    config.reset_joints = np.array([
        0.705474,
        1.000687,
        0.266987,
        -1.537979,
        -0.838564,
        0.991580,
        -0.285612,
    ], dtype=np.float64)
    config.bounds = np.array([
        [0.10, 0.70],
        [-0.30, 0.75],
        [0.01, 0.40],
    ], dtype=np.float64)
    return config
