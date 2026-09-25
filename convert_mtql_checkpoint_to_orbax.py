#!/usr/bin/env python
"""Convert a native MTQL V2 checkpoint to EXPO-compatible Orbax."""

from __future__ import annotations

import logging
from pathlib import Path

from absl import app, flags
from ml_collections import config_flags
import numpy as np

from agents import agents
from expo_ft.env.droid_utils import process_droid_dataset
from utils.flax_utils import restore_agent
from utils.mtql_droid import (
    DEFAULT_IMAGE_SIZE,
    OpenPINormalizer,
    create_droid_history_dataset,
)
from utils.mtql_orbax import initialize_checkpoint_dir, save_checkpoint


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "agent",
    "agents/mtql_transformer_v2.py",
    "V2 configuration matching the native checkpoint.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_task",
    None,
    "EXPO DROID task configuration.",
    lock_config=False,
)

flags.DEFINE_string(
    "dataset_path",
    None,
    "DROID dataset used to construct checkpoint-compatible input shapes.",
)
flags.DEFINE_string(
    "norm_stats_path",
    None,
    "OpenPI norm-stats directory or exact norm_stats.json path.",
)
flags.DEFINE_string(
    "native_restore_path",
    None,
    "Directory containing the native params_<epoch>.pkl checkpoint.",
)
flags.DEFINE_integer(
    "native_restore_epoch",
    None,
    "Native MTQL checkpoint epoch.",
)
flags.DEFINE_string(
    "orbax_checkpoint_dir",
    None,
    "Destination EXPO-compatible Orbax checkpoint directory.",
)
flags.DEFINE_integer(
    "orbax_step",
    None,
    "Destination Orbax step; defaults to native_restore_epoch.",
)
flags.DEFINE_integer(
    "num_data",
    1,
    "Number of complete DROID episodes used to establish input shapes.",
)
flags.DEFINE_integer("seed", 0, "Agent initialization seed.")
flags.DEFINE_integer("hist_length", 4, "Checkpoint history length.")
flags.DEFINE_integer("hist_stride", 1, "Checkpoint history stride.")
flags.DEFINE_integer("action_chunk_size", 1, "Checkpoint action chunk size.")
flags.DEFINE_integer(
    "image_size",
    DEFAULT_IMAGE_SIZE,
    "Checkpoint camera resolution.",
)
flags.DEFINE_integer(
    "keep_period",
    None,
    "Keep every Nth Orbax checkpoint permanently.",
)
flags.DEFINE_bool(
    "overwrite",
    False,
    "Replace an existing Orbax checkpoint directory.",
)
flags.DEFINE_bool(
    "resume",
    False,
    "Add the converted step to an existing Orbax checkpoint directory.",
)


def _validate_flags() -> None:
    required = {
        "config_task": FLAGS.config_task,
        "dataset_path": FLAGS.dataset_path,
        "norm_stats_path": FLAGS.norm_stats_path,
        "native_restore_path": FLAGS.native_restore_path,
        "native_restore_epoch": FLAGS.native_restore_epoch,
        "orbax_checkpoint_dir": FLAGS.orbax_checkpoint_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Missing required flags: "
            + ", ".join(f"--{name}" for name in missing)
        )
    if FLAGS.agent.get("agent_name") != "mtql_transformer_v2":
        raise ValueError(
            "Conversion only supports agent_name='mtql_transformer_v2'."
        )
    if FLAGS.num_data < 1:
        raise ValueError("--num_data must be positive.")
    if FLAGS.hist_length < 0:
        raise ValueError("--hist_length must be nonnegative.")
    if FLAGS.hist_stride < 1:
        raise ValueError("--hist_stride must be at least one.")
    if FLAGS.action_chunk_size < 1:
        raise ValueError("--action_chunk_size must be positive.")
    if FLAGS.image_size < 1:
        raise ValueError("--image_size must be positive.")
    if FLAGS.overwrite and FLAGS.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")


def main(argv) -> None:
    if len(argv) != 1:
        raise app.UsageError(
            f"Unexpected positional arguments: {argv[1:]}"
        )
    _validate_flags()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = FLAGS.agent
    hidden_dim = config.get("hidden_dim")
    if hidden_dim and "num_heads" in config:
        if hidden_dim % 32:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by 32."
            )
        config["num_heads"] = hidden_dim // 32
    config["action_chunk_size"] = FLAGS.action_chunk_size

    transitions = process_droid_dataset(
        FLAGS.dataset_path,
        FLAGS.config_task,
        num_data=FLAGS.num_data,
    )
    normalizer = OpenPINormalizer.from_path(FLAGS.norm_stats_path)
    dataset = create_droid_history_dataset(
        transitions,
        normalizer,
        hist_length=FLAGS.hist_length,
        hist_stride=FLAGS.hist_stride,
        action_chunk_size=FLAGS.action_chunk_size,
        discount=config.get("discount", 0.99),
        image_size=FLAGS.image_size,
    )
    example_batch = dataset.sample(
        1,
        idxs=np.array([0], dtype=np.int64),
    )

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )
    agent = restore_agent(
        agent,
        FLAGS.native_restore_path,
        FLAGS.native_restore_epoch,
    )

    checkpoint_dir = Path(FLAGS.orbax_checkpoint_dir)
    checkpoint_manager, _ = initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
    )
    step = (
        FLAGS.native_restore_epoch
        if FLAGS.orbax_step is None
        else FLAGS.orbax_step
    )
    try:
        if step in checkpoint_manager.all_steps():
            raise FileExistsError(
                f"Orbax checkpoint step {step} already exists."
            )
        if not save_checkpoint(checkpoint_manager, agent, step):
            raise RuntimeError(
                f"Orbax declined to save checkpoint step {step}."
            )
        checkpoint_manager.wait_until_finished()
    finally:
        checkpoint_manager.close()

    logging.info(
        "Converted native MTQL epoch %d to Orbax step %d in %s.",
        FLAGS.native_restore_epoch,
        step,
        checkpoint_dir,
    )


if __name__ == "__main__":
    app.run(main)
