#!/usr/bin/env bash
set -euo pipefail

MTQL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPO_FT_ROOT="${EXPO_FT_ROOT:-${MTQL_ROOT}/../expo-ft}"
# Keep the two repositories independent of local .venv symlinks. Override this
# on another workstation with CLIENT_PYTHON when the EXPO-FT client env lives
# elsewhere.
CLIENT_PYTHON="${CLIENT_PYTHON:-/dev/shm/expo-ft-client-venv/bin/python}"
TASK_CONFIG="${TASK_CONFIG:-${EXPO_FT_ROOT}/configs/task/pick.py}"
DATA_ROOT="${DATA_ROOT:-/tmp/ronpo-new-mtql-data}"
SAVE_ROOT="${SAVE_ROOT:-${DATA_ROOT}/pick_cube_balance}"
NUM_EPISODES="${NUM_EPISODES:-1}"

if [[ ! -d "${EXPO_FT_ROOT}" ]]; then
  echo "EXPO_FT_ROOT does not exist: ${EXPO_FT_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${CLIENT_PYTHON}" ]]; then
  echo "DROID client Python not found: ${CLIENT_PYTHON}" >&2
  echo "Set CLIENT_PYTHON to the EXPO-FT client environment." >&2
  exit 1
fi
if [[ ! -f "${TASK_CONFIG}" ]]; then
  echo "Task config not found: ${TASK_CONFIG}" >&2
  exit 1
fi

# The DROID package is vendored under EXPO-FT's client tree. Keep the wrapper
# usable after the repositories are moved together without reinstalling an
# editable package whose metadata may point at the old checkout path.
export PYTHONPATH="${EXPO_FT_ROOT}/client/droid:${EXPO_FT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${SAVE_ROOT}"
args=(
  "${CLIENT_PYTHON}" -m client.collect_data
  "--save_root=${SAVE_ROOT}"
  "--num_episodes=${NUM_EPISODES}"
  "--task_config=${TASK_CONFIG}"
)
args+=("$@")

run_command() {
  cd "${EXPO_FT_ROOT}"
  exec "${args[@]}"
}

# ZED access is group-protected on this workstation. If the current shell has
# not picked up the new group membership yet, rerun through sg zed. Compare
# numeric IDs so a stale/unknown group entry in the login session is harmless.
zed_gid="$(getent group zed | cut -d: -f3)"
if [[ -n "${zed_gid}" ]] && id -G | tr ' ' '\n' | grep -qx "${zed_gid}"; then
  run_command
fi

printf -v command_q '%q ' "${args[@]}"
exec sg zed -c "cd $(printf '%q' "${EXPO_FT_ROOT}") && ${command_q}"
