#!/usr/bin/env python3
"""Opt-in Shuffle service adapter for evaluator-owned manual controls.

This imports the existing rollout service without modifying it. The adapter
accepts a manual choice attached to ``get_info_for_step`` and supplies that
choice to the existing DROID environment's normal result-handling path.
"""

from __future__ import annotations

from pathlib import Path
import sys


EXPO_FT_ROOT = Path("/afs/cs.stanford.edu/u/ronpo/projects/expo-ft")
sys.path.insert(0, str(EXPO_FT_ROOT))

from client import run_client as service  # noqa: E402
from client.envs import droid_env as droid_env_module  # noqa: E402
from configs.task import egg as egg_task_config  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402

# ``configs.task.egg`` deliberately suppresses hardware-import failures so it
# can also be loaded by training/evaluation-only environments. This launcher
# has already imported the real client environment successfully, so bind that
# exact class explicitly before the service asks the config to construct it.
egg_task_config.EggEnv = droid_env_module.EggEnv


_original_handler = service._handle_environment_request
_original_manual_detector = droid_env_module.success_detector_manual
_next_manual_choice: str | None = None


def _evaluator_manual_detector(force_prompt=False):
    """Consume the evaluator's choice through the normal environment path."""
    del force_prompt
    global _next_manual_choice
    choice = _next_manual_choice
    _next_manual_choice = None
    if choice is None:
        # Only used before the evaluator enables this opt-in protocol.
        return _original_manual_detector(force_prompt=False)
    return choice


class _EvaluatorControlConnection:
    """Intercept opt-in control metadata before the stock request handler."""

    def __init__(self, websocket):
        self._websocket = websocket
        self._enabled = False
        self._packer = msgpack_numpy.Packer()

    async def recv(self, *args, **kwargs):
        global _next_manual_choice
        while True:
            packed_request = await self._websocket.recv(*args, **kwargs)
            request = msgpack_numpy.unpackb(packed_request)
            operation = request.get("operation")

            if operation == "enable_evaluator_manual_control":
                self._enabled = True
                await self._websocket.send(
                    self._packer.pack({"status": "success"})
                )
                continue

            if operation == "get_info_for_step" and self._enabled:
                choice = request.get("manual_choice")
                if choice not in {"success", "reset", "keep_going", "discard"}:
                    raise ValueError(f"Invalid evaluator manual choice: {choice!r}")
                _next_manual_choice = choice

            return packed_request

    def __getattr__(self, name):
        return getattr(self._websocket, name)


async def _shuffle_handler(websocket):
    await _original_handler(_EvaluatorControlConnection(websocket))


def main() -> None:
    droid_env_module.success_detector_manual = _evaluator_manual_detector
    service._handle_environment_request = _shuffle_handler
    args = service.tyro.cli(service.Args)
    service.main(args)


if __name__ == "__main__":
    main()
