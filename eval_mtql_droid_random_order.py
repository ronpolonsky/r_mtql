#!/usr/bin/env python
"""Run the live DROID evaluator with a shuffled balanced target order.

The existing eval_mtql_droid.py and the rollout task config remain unchanged.
The rollout environment still samples targets using its normal task config; this
wrapper consumes resets until each target matches a shuffled evaluation plan.
For 20 episodes the plan contains 7x target 1, 7x target 2, and 6x target 3.
"""

from __future__ import annotations

import logging
import secrets

from absl import app, flags
import numpy as np

import eval_mtql_droid as base_eval
from expo_ft.env.env_client import EnvClientWrapper as _BaseEnvClientWrapper


FLAGS = base_eval.FLAGS
flags.DEFINE_integer(
    "target_order_seed",
    -1,
    "Seed for the shuffled balanced target order; -1 uses fresh randomness.",
)
flags.DEFINE_string(
    "target_order",
    "",
    "Optional comma-separated target order, e.g. 1,1,2,3. "
    "Length must equal --num_episodes.",
)

_LOGGER = logging.getLogger(__name__)


class _RandomOrderEnvClientWrapper(_BaseEnvClientWrapper):
    """Env client that selects a shuffled balanced order from server resets."""

    def __init__(self, *args, **kwargs):
        self._target_order_seed = (
            secrets.randbits(63)
            if FLAGS.target_order_seed < 0
            else int(FLAGS.target_order_seed)
        )
        super().__init__(*args, **kwargs)

        episode_count = int(FLAGS.num_episodes)
        if FLAGS.target_order:
            try:
                explicit_order = [
                    int(item.strip())
                    for item in FLAGS.target_order.split(",")
                    if item.strip()
                ]
            except ValueError as exc:
                raise ValueError(
                    "--target_order must be comma-separated integers"
                ) from exc
            if len(explicit_order) != episode_count:
                raise ValueError(
                    "--target_order length must equal --num_episodes "
                    f"({len(explicit_order)} != {episode_count})"
                )
            if any(target not in (1, 2, 3) for target in explicit_order):
                raise ValueError("--target_order values must be 1, 2, or 3")
            self._target_order = explicit_order
        else:
            base_order = np.resize(
                np.array([1, 2, 3], dtype=np.int32), episode_count
            )
            rng = np.random.default_rng(self._target_order_seed)
            rng.shuffle(base_order)
            self._target_order = base_order.tolist()
        _LOGGER.info(
            "Randomized balanced target order seed=%d order=%s",
            self._target_order_seed,
            self._target_order,
        )

    def reset(self, eval_episode=None, eval_num_episodes=None):
        if eval_episode is None:
            raise RuntimeError(
                "Random target ordering requires eval_episode metadata on reset."
            )
        target_slot = int(eval_episode) - 1
        if target_slot < 0 or target_slot >= len(self._target_order):
            raise RuntimeError("Target-order schedule is exhausted.")

        desired_target = int(self._target_order[target_slot])
        attempts = 0
        while True:
            observation = super().reset(
                eval_episode=eval_episode,
                eval_num_episodes=eval_num_episodes,
            )
            actual_target = int(
                np.asarray(observation["target_count"]).reshape(-1)[0]
            )
            if actual_target == desired_target:
                _LOGGER.info(
                    "Selected target %d for evaluation slot %d after %d reset(s)",
                    actual_target,
                    target_slot + 1,
                    attempts + 1,
                )
                return observation

            attempts += 1
            _LOGGER.info(
                "Discarding server target %d; waiting for scheduled target %d",
                actual_target,
                desired_target,
            )


# main() resolves EnvClientWrapper from the imported evaluator module. Replace
# only that reference for this entry point; the original evaluator is untouched.
base_eval.EnvClientWrapper = _RandomOrderEnvClientWrapper


if __name__ == "__main__":
    app.run(base_eval.main)
