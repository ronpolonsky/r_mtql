#!/usr/bin/env python3
"""Evaluate Shuffle edited7 checkpoints with evaluator-owned SPACE gates.

The rollout service is unchanged. This wrapper captures seven stationary cue
observations and uses the exact virtual-cue history layout used in training.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from absl import app, flags

# Keep standalone evaluator launches consistent with the service adapter and
# training launchers. ``eval_mtql_droid`` imports ``expo_ft`` at module import
# time, so relying on the caller's PYTHONPATH can fail before Shuffle settings
# are validated.
PROJECT_ROOT = Path(__file__).resolve().parent
EXPO_ROOT = PROJECT_ROOT.parent / "expo-ft"
if str(EXPO_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPO_ROOT))

import eval_mtql_droid as base_eval
from client.real_utils.vis_utils import save_episode_video
from utils.mtql_droid import _stack_observations, convert_droid_observation
from utils.mtql_shuffle_device_cache import build_shuffle_history_indices


FLAGS = base_eval.FLAGS
FLAGS.set_default("hist_length", 14)
FLAGS.set_default("hist_stride", 6)
FLAGS.set_default("cue_mode", "none")
FLAGS.set_default("gripper_mode", "policy")
FLAGS.set_default("inject_training_gripper_state", False)


def _shuffle_cue_frame_count() -> int:
    """Read the cue-frame value safely before or after Abseil parsing."""
    return int(FLAGS["shuffle_cue_frames"].value)

flags.DEFINE_integer(
    "shuffle_cue_frames",
    7,
    "Number of stationary cue observations captured before policy control.",
)
flags.DEFINE_float(
    "shuffle_cue_hz",
    0.0,
    "Cue capture rate; zero uses the task control_hz.",
)


def shuffle_history_indices_for_anchor(
    local_anchor: int,
    size: int,
    *,
    hist_length: int,
    hist_stride: int,
    cue_frames: int = 7,
) -> np.ndarray:
    """Return one live history row using the production training builder."""
    if local_anchor < 0 or size <= local_anchor:
        raise ValueError(
            f"Expected 0 <= local_anchor < size, got {local_anchor}, {size}."
        )
    history_indices, _ = build_shuffle_history_indices(
        np.asarray([0], dtype=np.int64),
        np.asarray([size - 1], dtype=np.int64),
        size=int(size),
        hist_length=int(hist_length),
        hist_stride=int(hist_stride),
        cue_frames=int(cue_frames),
    )
    return history_indices[int(local_anchor)]


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
        self._local_video_enabled = bool(
            FLAGS.video_dir or FLAGS.full_video_dir
        )
        self._local_video_frames = []
        self._full_video_frames = []
        self._local_video_episode = 0

        # When making the optional evaluator-side recordings, do not also ask
        # the service to record. Otherwise the service would record the
        # pre-control polling frames into the normal raw video as well.
        if FLAGS.full_video_dir:
            request = kwargs.get("env_creation_request")
            if request is not None:
                request = dict(request)
                request["video_dir"] = ""
                kwargs["env_creation_request"] = request

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

    @staticmethod
    def _video_frame(observation: Mapping[str, Any]) -> np.ndarray | None:
        """Build the same side-plus-wrist frame used by the service video."""
        try:
            side = np.asarray(observation["exterior_image_1_left"])
            wrist = np.asarray(observation["wrist_image_left"])
        except (KeyError, TypeError):
            return None
        if side.ndim != 3 or wrist.ndim != 3:
            return None
        if side.shape[-1] != 3 or wrist.shape[-1] != 3:
            return None
        if side.dtype != np.uint8:
            side = np.clip(side, 0, 255).astype(np.uint8)
        if wrist.dtype != np.uint8:
            wrist = np.clip(wrist, 0, 255).astype(np.uint8)
        if side.shape[:2] != wrist.shape[:2]:
            return None
        return np.concatenate([side, wrist], axis=1)

    def _record_local_frame(
        self, observation: Mapping[str, Any], *, policy_visible: bool
    ) -> None:
        if not self._local_video_enabled:
            return
        frame = self._video_frame(observation)
        if frame is None:
            return
        self._full_video_frames.append(frame)
        if policy_visible:
            self._local_video_frames.append(frame)

    def _flush_local_videos(self, *, save: bool) -> None:
        if not self._local_video_enabled:
            return
        if save:
            if FLAGS.video_dir and self._local_video_frames:
                save_episode_video(
                    self._local_video_frames,
                    FLAGS.video_dir,
                    self._local_video_episode,
                    prefix="raw",
                )
            if FLAGS.full_video_dir and self._full_video_frames:
                save_episode_video(
                    self._full_video_frames,
                    FLAGS.full_video_dir,
                    self._local_video_episode,
                    prefix="full",
                )
        self._local_video_frames = []
        self._full_video_frames = []

    def _wait_for_start_with_full_video(self) -> None:
        """Poll observations while waiting for the second SPACE press."""
        print(
            "Press SPACE to start Shuffle policy control. "
            "Recording the full pre-control interval...",
            flush=True,
        )
        file_descriptor = os.open("/dev/tty", os.O_RDONLY)
        previous_settings = termios.tcgetattr(file_descriptor)
        try:
            tty.setcbreak(file_descriptor)
            poll_period = 1.0 / max(float(FLAGS.config_task.control_hz), 1.0)
            while True:
                poll_start = time.monotonic()
                observation = super().get_observation()
                self._record_local_frame(observation, policy_visible=False)
                readable, _, _ = select.select(
                    [file_descriptor], [], [], max(
                        0.0, poll_period - (time.monotonic() - poll_start)
                    )
                )
                if readable and b" " in os.read(file_descriptor, 64):
                    return
        finally:
            termios.tcsetattr(
                file_descriptor,
                termios.TCSADRAIN,
                previous_settings,
            )
            os.close(file_descriptor)

    def reset(self, eval_episode=None, eval_num_episodes=None):
        self._flush_local_videos(save=False)
        self._local_video_episode = int(eval_episode or (self._local_video_episode + 1))
        reset_observation = super().reset(
            eval_episode=eval_episode,
            eval_num_episodes=eval_num_episodes,
        )
        self._record_local_frame(reset_observation, policy_visible=True)
        _wait_for_local_space(
            "Reset complete. Press SPACE to capture seven stationary Shuffle cues."
        )

        cue_hz = float(FLAGS.shuffle_cue_hz)
        if cue_hz <= 0:
            cue_hz = float(FLAGS.config_task.control_hz)
        cue_period = 1.0 / cue_hz

        cues = []
        for cue_index in range(int(FLAGS.shuffle_cue_frames)):
            if cue_index:
                time.sleep(cue_period)
            cue = super().get_observation()
            cues.append(cue)
            self._record_local_frame(cue, policy_visible=True)
            print(
                f"Captured Shuffle cue {cue_index + 1}/"
                f"{FLAGS.shuffle_cue_frames}",
                flush=True,
            )

        type(self)._latest_cues = tuple(cues)
        self._motion_started = False
        print(
            "Seven cues captured. Robot motion remains locked until the second SPACE.",
            flush=True,
        )
        return reset_observation

    def get_observation(self):
        if not self._motion_started:
            if FLAGS.full_video_dir:
                self._wait_for_start_with_full_video()
            else:
                _wait_for_local_space("Press SPACE to start Shuffle policy control.")
            self._motion_started = True
            print("Starting Shuffle policy control.", flush=True)
            print(
                "[manual] During policy control, enter 1 for success, 2 for "
                "reset/failure, 3 to keep going, or 4 to discard, then ENTER.",
                flush=True,
            )
        observation = super().get_observation()
        self._record_local_frame(observation, policy_visible=True)
        return observation

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

        result = self._call("get_info_for_step", request_info)
        done, success, _, _ = result
        if done:
            self._flush_local_videos(save=success is not None)
        return result


class ShuffleHistoryBuffer(base_eval.DroidHistoryBuffer):
    """Live history using training's exact virtual-cue index builder."""

    def reset(self, initial_observation: Mapping[str, Any]) -> None:
        del initial_observation
        cues = ShuffleRolloutClient.latest_cues()
        expected = _shuffle_cue_frame_count()
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
        # The next current observation is real trajectory index seven. It is
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
        indices = shuffle_history_indices_for_anchor(
            self._anchor,
            table_size,
            hist_length=int(self.hist_length),
            hist_stride=int(self.hist_stride),
            cue_frames=_shuffle_cue_frame_count(),
        )
        if np.any(indices >= len(self._observations)):
            raise RuntimeError(
                "Shuffle history selected the current observation before its "
                "action was executed."
            )
        return _stack_observations(
            [self._observations[int(index)] for index in indices]
        )

    def history_debug_payload(self) -> dict[str, np.ndarray]:
        """Return selected indices and frames for history debugging."""
        table_size = self._anchor + 1
        indices = shuffle_history_indices_for_anchor(
            self._anchor,
            table_size,
            hist_length=int(self.hist_length),
            hist_stride=int(self.hist_stride),
            cue_frames=_shuffle_cue_frame_count(),
        )
        history = _stack_observations(
            [self._observations[int(index)] for index in indices]
        )
        return {"indices": np.asarray(indices, dtype=np.int64), **history}


def _validate_settings() -> None:
    required = {
        "hist_length": (FLAGS.hist_length, 14),
        "hist_stride": (FLAGS.hist_stride, 6),
        "shuffle_cue_frames": (_shuffle_cue_frame_count(), 7),
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
            "Invalid Shuffle edited7 evaluation settings: "
            + ", ".join(mismatches)
        )
    if str(FLAGS.config_task.get("env_name", "")).lower() != "shuffle":
        raise ValueError("Shuffle evaluation requires the shuffle task config.")


def main(argv):
    _validate_settings()
    inference_agents = {
        "mtql_transformer_success_actor_real": "mtql_transformer_real",
        "mtql_mlp_success_actor_real": "mtql_mlp_real",
        "new_bc_flow_transformer_real": "new_bc_flow_transformer_real",
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
