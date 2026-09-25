#!/usr/bin/env python3
"""Evaluate Shuffle edited5 checkpoints with evaluator-owned SPACE gates.

The rollout service is unchanged. This wrapper captures five stationary cue
observations and uses the exact virtual-cue history layout used in training.
"""

from __future__ import annotations

import os
import select
import termios
import time
import tty
from typing import Any, Mapping, Sequence

import numpy as np
from absl import app, flags

import eval_mtql_droid as base_eval
from utils.mtql_droid import _stack_observations, convert_droid_observation
from utils.mtql_shuffle_device_cache import build_shuffle_history_indices


FLAGS = base_eval.FLAGS
FLAGS.set_default("hist_length", 10)
FLAGS.set_default("hist_stride", 4)
FLAGS.set_default("cue_mode", "none")
FLAGS.set_default("gripper_mode", "policy")
FLAGS.set_default("inject_training_gripper_state", False)

flags.DEFINE_integer(
    "shuffle_cue_frames",
    5,
    "Number of stationary cue observations captured before policy control.",
)
flags.DEFINE_float(
    "shuffle_cue_hz",
    0.0,
    "Cue capture rate; zero uses the task control_hz.",
)


def _wait_for_local_space(prompt: str) -> None:
    """Wait for SPACE locally and always restore the evaluator terminal."""
    print(prompt, flush=True)
    file_descriptor = os.open("/dev/tty", os.O_RDONLY)
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        while os.read(file_descriptor, 1) != b" ":
            pass
    finally:
        termios.tcsetattr(
            file_descriptor,
            termios.TCSADRAIN,
            previous_settings,
        )
        os.close(file_descriptor)


def _poll_local_manual_choice() -> str:
    """Return a manual result without blocking normal policy execution."""
    try:
        terminal = open("/dev/tty", "r")
    except OSError:
        return "keep_going"
    with terminal:
        readable, _, _ = select.select([terminal.fileno()], [], [], 0.0)
        if not readable:
            return "keep_going"
        while True:
            choice = terminal.readline().strip()
            if choice == "1":
                print("[manual] Success annotated.", flush=True)
                return "success"
            if choice == "2":
                print("[manual] Doing reset.", flush=True)
                return "reset"
            if choice == "3":
                print("[manual] Keep going.", flush=True)
                return "keep_going"
            if choice == "4":
                print(
                    "[manual] Discarding episode; it will not count.",
                    flush=True,
                )
                return "discard"
            print(
                "[manual] Invalid input. Please type 1 (success), 2 (reset), "
                "3 (keep going), or 4 (discard), then ENTER:",
                flush=True,
            )


class ShuffleRolloutClient(base_eval.EnvClientWrapper):
    """Add local two-SPACE gating without changing the rollout service."""

    _latest_cues: tuple[Mapping[str, Any], ...] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._motion_started = False
        try:
            self.client._call_operation(
                "enable_evaluator_manual_control",
                {"env_id": self.env_id},
            )
        except RuntimeError as error:
            raise RuntimeError(
                "This Shuffle evaluator requires the separate "
                "/tmp/ronpo-shuffle-eval/run_shuffle_service.py launcher. "
                "The stock rollout service intentionally remains unchanged."
            ) from error

    def _call(self, op_name, thunk):
        del op_name
        # Never hide failures behind the generic recreate/reset retry loop.
        return thunk()

    @classmethod
    def latest_cues(cls) -> Sequence[Mapping[str, Any]]:
        return cls._latest_cues

    def reset(self, eval_episode=None, eval_num_episodes=None):
        reset_observation = super().reset(
            eval_episode=eval_episode,
            eval_num_episodes=eval_num_episodes,
        )
        _wait_for_local_space(
            "Reset complete. Press SPACE to capture five stationary Shuffle cues."
        )

        cue_hz = float(FLAGS.shuffle_cue_hz)
        if cue_hz <= 0:
            cue_hz = float(FLAGS.config_task.control_hz)
        cue_period = 1.0 / cue_hz

        cues = []
        for cue_index in range(int(FLAGS.shuffle_cue_frames)):
            if cue_index:
                time.sleep(cue_period)
            cues.append(super().get_observation())
            print(
                f"Captured Shuffle cue {cue_index + 1}/"
                f"{FLAGS.shuffle_cue_frames}",
                flush=True,
            )

        type(self)._latest_cues = tuple(cues)
        self._motion_started = False
        print(
            "Five cues captured. Robot motion remains locked until the second SPACE.",
            flush=True,
        )
        return reset_observation

    def get_observation(self):
        if not self._motion_started:
            _wait_for_local_space("Press SPACE to start Shuffle policy control.")
            self._motion_started = True
            print("Starting Shuffle policy control.", flush=True)
            print(
                "[manual] During policy control, enter 1 for success, 2 for "
                "reset/failure, 3 to keep going, or 4 to discard, then ENTER.",
                flush=True,
            )
        return super().get_observation()

    def get_info_for_step(self):
        manual_choice = _poll_local_manual_choice()

        def request_info():
            response = self.client._call_operation(
                "get_info_for_step",
                {
                    "env_id": self.env_id,
                    "manual_choice": manual_choice,
                },
            )
            return (
                response["done"],
                response["success"],
                response["reward"],
                response["mask"],
            )

        return self._call("get_info_for_step", request_info)


class ShuffleHistoryBuffer(base_eval.DroidHistoryBuffer):
    """Live history using training's exact virtual-cue index builder."""

    def reset(self, initial_observation: Mapping[str, Any]) -> None:
        del initial_observation
        cues = ShuffleRolloutClient.latest_cues()
        expected = int(FLAGS.shuffle_cue_frames)
        if len(cues) != expected:
            raise RuntimeError(
                f"Expected exactly {expected} captured cues; received {len(cues)}."
            )
        self._observations = [
            convert_droid_observation(
                cue,
                self.normalizer,
                image_size=self.image_size,
                target_count=None,
            )
            for cue in cues
        ]
        # The next current observation is real trajectory index five. It is
        # appended only after the action selected from it is executed.
        self._anchor = expected

    def append(self, observation: Mapping[str, Any]) -> None:
        if not self._observations:
            raise RuntimeError("Call reset() before append().")
        self._observations.append(
            convert_droid_observation(
                observation,
                self.normalizer,
                image_size=self.image_size,
                target_count=None,
            )
        )
        self._anchor += 1

    def history_observations(self) -> dict[str, np.ndarray] | None:
        if self.hist_length == 0:
            return None
        if not self._observations:
            raise RuntimeError("Call reset() before requesting history.")

        # Reuse the production training function instead of duplicating its
        # virtual expansion formula in the evaluator.
        table_size = self._anchor + 1
        history_indices, _ = build_shuffle_history_indices(
            np.asarray([0], dtype=np.int64),
            np.asarray([table_size - 1], dtype=np.int64),
            size=table_size,
            hist_length=int(self.hist_length),
            hist_stride=int(self.hist_stride),
            cue_frames=int(FLAGS.shuffle_cue_frames),
        )
        indices = history_indices[self._anchor]
        if np.any(indices >= len(self._observations)):
            raise RuntimeError(
                "Shuffle history selected the current observation before its "
                "action was executed."
            )
        return _stack_observations(
            [self._observations[int(index)] for index in indices]
        )


def _validate_settings() -> None:
    required = {
        "hist_length": (FLAGS.hist_length, 10),
        "hist_stride": (FLAGS.hist_stride, 4),
        "shuffle_cue_frames": (FLAGS.shuffle_cue_frames, 5),
        "cue_mode": (FLAGS.cue_mode, "none"),
        "gripper_mode": (FLAGS.gripper_mode, "policy"),
        "inject_training_gripper_state": (
            FLAGS.inject_training_gripper_state,
            False,
        ),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "Invalid Shuffle edited5 evaluation settings: "
            + ", ".join(mismatches)
        )
    if str(FLAGS.config_task.get("env_name", "")).lower() != "shuffle":
        raise ValueError("Shuffle evaluation requires the shuffle task config.")


def main(argv):
    _validate_settings()
    inference_agents = {
        "mtql_transformer_success_actor_real": "mtql_transformer_real",
        "mtql_mlp_success_actor_real": "mtql_mlp_real",
    }
    training_name = FLAGS.agent.get("agent_name")
    if training_name in inference_agents:
        FLAGS.agent.agent_name = inference_agents[training_name]

    # Process-local substitutions leave normal Egg/Candy evaluation unchanged.
    base_eval.EnvClientWrapper = ShuffleRolloutClient
    base_eval.DroidHistoryBuffer = ShuffleHistoryBuffer
    return base_eval.main(argv)


if __name__ == "__main__":
    app.run(main)
