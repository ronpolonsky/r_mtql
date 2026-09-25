#!/usr/bin/env python3
"""Materialize deterministic DROID preprocessing for offline MTQL runs.

This performs the expensive raw-HDF5 decode, image resize, and OpenPI
normalization once.  Training still chooses history length/stride and applies
augmentation at runtime.
"""

from __future__ import annotations

import os

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from utils.mtql_droid import (
    OpenPINormalizer,
    create_droid_history_dataset,
    process_droid_dataset,
    save_droid_cache,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", None, "Directory containing DROID episodes.")
flags.DEFINE_string(
    "norm_stats_path",
    None,
    "OpenPI norm-stats directory or exact norm_stats.json path.",
)
flags.DEFINE_string("cache_dir", None, "Output directory for the shared cache.")
flags.DEFINE_integer("image_size", 224, "Square image size used by the adapter.")
flags.DEFINE_integer(
    "n_succ",
    -1,
    "Successful episodes to include; -1 includes all successes.",
)
flags.DEFINE_integer(
    "n_fails",
    -1,
    "Failed episodes to include; -1 includes all failures.",
)
flags.DEFINE_bool(
    "overwrite",
    False,
    "Replace an existing cache directory intentionally.",
)
config_flags.DEFINE_config_file(
    "config_task",
    None,
    "EXPO-FT DROID task config used to select action keys.",
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
        raise FileNotFoundError(f"DROID dataset not found: {FLAGS.dataset_path}")
    if FLAGS.n_succ < -1 or FLAGS.n_fails < -1:
        raise ValueError("--n_succ and --n_fails must be -1 or nonnegative.")
    if FLAGS.image_size < 1:
        raise ValueError("--image_size must be positive.")

    normalizer = OpenPINormalizer.from_path(FLAGS.norm_stats_path)
    transitions = process_droid_dataset(
        FLAGS.dataset_path,
        FLAGS.config_task,
        n_success=FLAGS.n_succ,
        n_failure=FLAGS.n_fails,
    )
    raw_actions = np.asarray(
        [transition["actions"] for transition in transitions],
        dtype=np.float32,
    )

    # History and action-chunk construction are intentionally deferred to the
    # trainer. The underlying preprocessed transition arrays are independent
    # of those choices.
    dataset = create_droid_history_dataset(
        transitions,
        normalizer,
        hist_length=0,
        hist_stride=1,
        action_chunk_size=1,
        image_size=FLAGS.image_size,
    )
    save_droid_cache(
        dataset,
        raw_actions,
        cache_dir=FLAGS.cache_dir,
        dataset_path=FLAGS.dataset_path,
        norm_stats_path=FLAGS.norm_stats_path,
        task_config=FLAGS.config_task,
        image_size=FLAGS.image_size,
        n_success=FLAGS.n_succ,
        n_failure=FLAGS.n_fails,
        overwrite=FLAGS.overwrite,
    )


if __name__ == "__main__":
    app.run(main)
