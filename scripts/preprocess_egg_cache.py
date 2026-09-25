#!/usr/bin/env python3
"""Build the full-data, cue-free egg cache used by m_real_egg.py."""

from __future__ import annotations

import os

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from utils.mtql_droid import (
    OpenPINormalizer,
    create_droid_history_dataset,
    process_egg_dataset,
    save_egg_cache,
)


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "dataset_path",
    "/iris/u/ronpo/expo-ft-data/egg_v1",
    "Root containing finalized egg episodes.",
)
flags.DEFINE_string("norm_stats_path", None, "Egg norm-stats directory.")
flags.DEFINE_string("cache_dir", None, "Output directory for the egg cache.")
flags.DEFINE_integer("image_size", 224, "Square image size used by the adapter.")
flags.DEFINE_bool("overwrite", False, "Replace an existing cache intentionally.")
config_flags.DEFINE_config_file(
    "config_task",
    None,
    "EXPO-FT egg task config used to select action keys.",
    lock_config=False,
)


def main(argv):
    del argv
    required = {
        "dataset_path": FLAGS.dataset_path,
        "norm_stats_path": FLAGS.norm_stats_path,
        "cache_dir": FLAGS.cache_dir,
        "config_task": FLAGS.config_task,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise app.UsageError(
            "Missing required flags: "
            + ", ".join(f"--{name}" for name in missing)
        )
    if not os.path.isdir(FLAGS.dataset_path):
        raise FileNotFoundError(f"Egg dataset not found: {FLAGS.dataset_path}")
    if FLAGS.image_size < 1:
        raise ValueError("--image_size must be positive.")

    transitions = process_egg_dataset(FLAGS.dataset_path, FLAGS.config_task)
    raw_actions = np.asarray(
        [transition["actions"] for transition in transitions],
        dtype=np.float32,
    )
    dataset = create_droid_history_dataset(
        transitions,
        OpenPINormalizer.from_path(FLAGS.norm_stats_path),
        hist_length=0,
        hist_stride=1,
        action_chunk_size=1,
        image_size=FLAGS.image_size,
        store_next_observations=False,
    )
    save_egg_cache(
        dataset,
        raw_actions,
        cache_dir=FLAGS.cache_dir,
        dataset_path=FLAGS.dataset_path,
        norm_stats_path=FLAGS.norm_stats_path,
        task_config=FLAGS.config_task,
        image_size=FLAGS.image_size,
        overwrite=FLAGS.overwrite,
    )


if __name__ == "__main__":
    app.run(main)
