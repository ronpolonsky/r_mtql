#!/usr/bin/env python3
"""Shuffle pure-BC trainer with the same virtual-history device cache.

This is an isolated entry point. It leaves ``m_real.py`` unchanged, patches
only its loader/augmentation hooks at runtime, and retains its established
contract that the Egg-shaped BC baseline samples successful transitions only.
"""

from __future__ import annotations

import sys

from absl import app, flags


EXPO_ROOT = "/iris/u/ronpo/projects/expo-ft"
SHUFFLE_DATASET = "/iris/u/ronpo/mtql-runs/datasets/shuffle_edited5_view"
SHUFFLE_CACHE = "/iris/u/ronpo/mtql-runs/caches/shuffle_edited5_224"
SHUFFLE_NORM_STATS = "/iris/u/ronpo/expo-ft-data/shuffle_edited5_norm_stats"
SHUFFLE_TASK_CONFIG = "/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py"
SHUFFLE_RUN_ROOT = "/iris/u/ronpo/mtql-runs/shuffle_edited5"

if EXPO_ROOT not in sys.path:
    sys.path.insert(0, EXPO_ROOT)

import m_real  # noqa: E402
from utils.mtql_device_cache import augment_droid_batch_device  # noqa: E402
from utils.mtql_shuffle_device_cache import (  # noqa: E402
    DeviceCachedShuffleHistoryDataset,
)


FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "device_frame_cache",
    True,
    "Keep compact observations and fixed virtual-history indices on the accelerator.",
)
flags.DEFINE_integer(
    "cue_frames",
    5,
    "Number of initial shuffle cue frames expanded in virtual index space.",
)

FLAGS.set_default("agent", "agents/new_bc_flow_transformer_real.py")
FLAGS.set_default("dataset_kind", "egg")
FLAGS.set_default("cue_mode", "none")
FLAGS.set_default("dataset_path", SHUFFLE_DATASET)
FLAGS.set_default("dataset_cache", SHUFFLE_CACHE)
FLAGS.set_default("norm_stats_path", SHUFFLE_NORM_STATS)
FLAGS.set_default("config_task", SHUFFLE_TASK_CONFIG)
FLAGS.set_default("save_dir", SHUFFLE_RUN_ROOT)
FLAGS.set_default("project", "real_shuffle_edited5_bc")
FLAGS.set_default("wandb_run_group", "shuffle_edited5_bc")
FLAGS.set_default("hist_length", 14)
FLAGS.set_default("hist_stride", 6)
FLAGS.set_default("train_steps", 1000000)
FLAGS.set_default("log_interval", 10000)
FLAGS.set_default("save_interval", 30000)
FLAGS.set_default("p_aug", 1.0)
FLAGS.set_default("n_succ", -1)
FLAGS.set_default("n_fails", -1)

_load_egg_history_dataset = m_real.load_egg_history_dataset
_checkpoint_metadata = m_real._checkpoint_metadata


def _load_device_cached_shuffle_dataset(*args, **kwargs):
    host_dataset, cache_info = _load_egg_history_dataset(*args, **kwargs)
    return DeviceCachedShuffleHistoryDataset(
        host_dataset,
        cue_frames=FLAGS.cue_frames,
    ), cache_info


def _shuffle_checkpoint_metadata(**training_contract):
    training_contract.update(
        {
            "task_name": "shuffle",
            "dataset_view": "shuffle_edited5_view",
            "history_layout": "virtual_initial_cue_expansion_then_standard_stride",
            "history_cue_frames": int(FLAGS.cue_frames),
            "device_frame_cache": True,
        }
    )
    return _checkpoint_metadata(**training_contract)


def main(argv):
    if not FLAGS.device_frame_cache:
        raise ValueError("This shuffle BC entry point requires device_frame_cache=true.")
    if FLAGS.dataset_kind != "egg" or FLAGS.cue_mode != "none":
        raise ValueError("Shuffle BC requires --dataset_kind=egg and --cue_mode=none.")
    if FLAGS.cue_frames != 5:
        raise ValueError("This shuffle setup uses exactly five initial cue frames.")
    if FLAGS.hist_length < FLAGS.cue_frames:
        raise ValueError("hist_length must be at least cue_frames.")

    m_real.load_egg_history_dataset = _load_device_cached_shuffle_dataset
    m_real.augment_droid_batch = augment_droid_batch_device
    m_real._checkpoint_metadata = _shuffle_checkpoint_metadata
    return m_real.main(argv)


if __name__ == "__main__":
    app.run(main)
