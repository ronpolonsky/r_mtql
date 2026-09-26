#!/usr/bin/env python3
"""Opt-in Shuffle service adapter for evaluator-owned manual controls.

This imports the existing rollout service without modifying it. The adapter
accepts a manual choice attached to ``get_info_for_step`` and supplies that
choice to the existing DROID environment's normal result-handling path.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


_configured_root = os.environ.get("EXPO_FT_ROOT")
_candidate_roots = [
    Path(_configured_root) if _configured_root else None,
    Path("/iris/u/ronpo/projects/expo-ft"),
    Path("/afs/cs.stanford.edu/u/ronpo/projects/expo-ft"),
]
EXPO_FT_ROOT = next(
    (root for root in _candidate_roots if root is not None and root.exists()),
    Path("/iris/u/ronpo/projects/expo-ft"),
)
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


class _EvaluatorDisabledShuffleGates(dict):
    """Registry that prevents the stock five-cue gate from being installed.

    ``run_client`` creates a ``ShuffleEpisodeGate`` during ``create_env`` by
    assigning into ``_shuffle_episode_gates``.  Replacing the registry with a
    normal empty dict is therefore not enough: the legacy gate is immediately
    recreated.  The evaluator owns cue capture and motion gating, so this
    registry deliberately ignores registrations and always reports no gate.
    """

    def __setitem__(self, key, value):
        del key, value

    def get(self, key, default=None):
        del key
        return default


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
    # The MTQL evaluator owns the edited7 two-SPACE protocol and captures
    # exactly seven cue observations.  The stock service gate is an older
    # five-cue collector gate; leaving it enabled would double-block cue
    # capture and construct the wrong temporal history.  Keep the original
    # handler intact, but isolate it from that gate for this opt-in service.
    # Older EXPO client versions do not define this field.  In that case
    # there is no stock shuffle gate to disable.
    gate_attr = "_shuffle_episode_gates"
    has_stock_gate = hasattr(service, gate_attr)
    original_gates = getattr(service, gate_attr, None)
    if has_stock_gate:
        setattr(service, gate_attr, _EvaluatorDisabledShuffleGates())
    try:
        await _original_handler(_EvaluatorControlConnection(websocket))
    finally:
        if has_stock_gate:
            setattr(service, gate_attr, original_gates)


def main() -> None:
    droid_env_module.success_detector_manual = _evaluator_manual_detector
    service._handle_environment_request = _shuffle_handler
    args = service.tyro.cli(service.Args)
    service.main(args)


if __name__ == "__main__":
    main()
