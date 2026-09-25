#!/usr/bin/env python3
"""Cue-free offline training on an Egg dataset.

This is a deliberately thin entry point around :mod:`m_real`.  Keeping the
training loop shared ensures Egg uses the same agents, augmentation,
checkpointing, RNG continuation, and logging implementation as the other
real-robot experiments. MTQL agents sample all selected outcomes; pure BC
samples only selected successful episodes.
"""

from __future__ import annotations

import os
import sys

from absl import app, flags

import m_real


FLAGS = flags.FLAGS

EGG_V2_DATASET = "/iris/u/ronpo/expo-ft-data/egg_v2"
EGG_V2_CACHE = "/iris/u/ronpo/mtql-runs/caches/egg_v2_224"
EGG_V2_NORM_STATS = "/iris/u/ronpo/expo-ft-data/egg_v2_norm_stats"
EXPO_ROOT = "/iris/u/ronpo/projects/expo-ft"
EGG_TASK_CONFIG = "/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py"
EGG_V2_RUN_ROOT = "/iris/u/ronpo/mtql-runs/egg_v2"

# The task config imports ``configs.task.real_base`` from EXPO-FT. This
# project also contains a partial ``configs`` namespace, so make the intended
# package resolution explicit before ml_collections loads the config file.
if EXPO_ROOT not in sys.path:
    sys.path.insert(0, EXPO_ROOT)

# Safe egg_v2 defaults for backwards compatibility. Dataset paths remain
# overrideable so this entry point also supports egg_v4 and later datasets.
FLAGS.set_default("dataset_kind", "egg")
FLAGS.set_default("cue_mode", "none")
FLAGS.set_default("dataset_path", EGG_V2_DATASET)
FLAGS.set_default("dataset_cache", EGG_V2_CACHE)
FLAGS.set_default("norm_stats_path", EGG_V2_NORM_STATS)
FLAGS.set_default("config_task", EGG_TASK_CONFIG)
FLAGS.set_default("save_dir", EGG_V2_RUN_ROOT)
FLAGS.set_default("project", "real_egg_v2")
FLAGS.set_default("wandb_run_group", "egg_v2")
FLAGS.set_default("hist_length", 12)
FLAGS.set_default("hist_stride", 7)
FLAGS.set_default("train_steps", 300000)
FLAGS.set_default("log_interval", 10000)
FLAGS.set_default("save_interval", 10000)
FLAGS.set_default("p_aug", 1.0)
FLAGS.set_default("n_succ", -1)
FLAGS.set_default("n_fails", -1)


def _same_path(actual: str | None, expected: str) -> bool:
    return (
        actual is not None
        and os.path.realpath(actual) == os.path.realpath(expected)
    )


def main(argv):
    if FLAGS.dataset_kind != "egg":
        raise ValueError("m_real_egg.py is fixed to --dataset_kind=egg.")
    if FLAGS.cue_mode != "none":
        raise ValueError("m_real_egg.py is cue-free and requires --cue_mode=none.")
    if not str(getattr(FLAGS.config_task, "env_name", "")).startswith("egg"):
        raise ValueError(
            "m_real_egg.py requires an Egg task configuration; received "
            f"env_name={getattr(FLAGS.config_task, 'env_name', None)!r}."
        )
    return m_real.main(argv)


if __name__ == "__main__":
    app.run(main)
