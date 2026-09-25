#!/usr/bin/env python3
"""Egg success-actor trainer with an opt-in device-resident frame cache.

This entry point patches only its own process.  The standard Egg trainers,
dataset classes, and launchers remain unchanged.
"""

from __future__ import annotations

import sys

from absl import app, flags

import m_real_success_actor
from utils.mtql_device_cache import (
    DeviceCachedDroidHistoryDataset,
    augment_droid_batch_device,
)


FLAGS = flags.FLAGS

EXPO_ROOT = "/iris/u/ronpo/projects/expo-ft"
EGG_TASK_CONFIG = "/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py"

if EXPO_ROOT not in sys.path:
    sys.path.insert(0, EXPO_ROOT)

flags.DEFINE_bool(
    "device_frame_cache",
    True,
    "Keep compact selected observations and fixed history indices on device.",
)

FLAGS.set_default("dataset_kind", "egg")
FLAGS.set_default("cue_mode", "none")
FLAGS.set_default("config_task", EGG_TASK_CONFIG)
FLAGS.set_default("hist_length", 19)
FLAGS.set_default("hist_stride", 7)
FLAGS.set_default("p_aug", 1.0)


_load_egg_history_dataset = m_real_success_actor.load_egg_history_dataset


def _load_device_cached_egg_dataset(*args, **kwargs):
    host_dataset, cache_info = _load_egg_history_dataset(*args, **kwargs)
    return DeviceCachedDroidHistoryDataset(host_dataset), cache_info


def main(argv):
    if not FLAGS.device_frame_cache:
        raise ValueError(
            "This isolated entry point requires --device_frame_cache=true."
        )
    if FLAGS.dataset_kind != "egg":
        raise ValueError(
            "m_real_egg_success_actor_device_cache.py requires dataset_kind=egg."
        )
    if FLAGS.cue_mode != "none":
        raise ValueError(
            "The device-cache Egg trainer is cue-free and requires cue_mode=none."
        )
    if not str(getattr(FLAGS.config_task, "env_name", "")).startswith("egg"):
        raise ValueError(
            "The device-cache trainer requires an Egg task configuration; "
            f"received env_name={getattr(FLAGS.config_task, 'env_name', None)!r}."
        )

    # Patch names imported directly by the isolated base trainer.  This affects
    # only this Python process and leaves every existing source path unchanged.
    m_real_success_actor.load_egg_history_dataset = (
        _load_device_cached_egg_dataset
    )
    m_real_success_actor.augment_droid_batch = augment_droid_batch_device
    return m_real_success_actor.main(argv)


if __name__ == "__main__":
    app.run(main)
