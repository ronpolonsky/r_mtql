#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/dev/shm/expo-ft-client-venv/bin/python}"
NUM_EPISODES="${NUM_EPISODES:-25}"
TARGET_COUNT="${TARGET_COUNT:-3}"
SAVE_ROOT="${SAVE_ROOT:-/iris/u/ronpo/expo-ft-data/candy_scoop_v2}"
NUC_IP="${NUC_IP:-172.16.0.1}"
TARGET_DIR_PREFIX="${TARGET_DIR_PREFIX:-}"

if [[ "$TARGET_COUNT" == "shuffle" || "$TARGET_COUNT" == "random" ]]; then
    TARGET_FLAGS=(--task_config.min_target_count=1 --task_config.max_target_count=3)
    : "${TARGET_DIR_PREFIX:=target_num_}"
else
    TARGET_FLAGS=(--task_config.min_target_count="$TARGET_COUNT" --task_config.max_target_count="$TARGET_COUNT")
    : "${TARGET_DIR_PREFIX:=target_}"
fi

if [[ ! -x "$CLIENT_PYTHON" ]]; then
    echo "Python environment not found: $CLIENT_PYTHON" >&2
    echo "Set CLIENT_PYTHON to the client environment's python executable." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUC_IP
export LD_LIBRARY_PATH=/usr/local/zed/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Attach the DROID server to the already-running robot and gripper controllers.
"$CLIENT_PYTHON" -c 'import os, zerorpc; client=zerorpc.Client(heartbeat=20); client.connect("tcp://" + os.environ["NUC_IP"] + ":4242"); client.launch_robot(); print("DROID robot and gripper handles ready")'

exec "$CLIENT_PYTHON" -m client.collect_data \
    --save_root="$SAVE_ROOT" \
    --num_episodes="$NUM_EPISODES" \
    --task_config=configs/task/candy_scoop.py \
    --task_config.launch_controller=false \
    --task_config.include_target_cue=false \
    --target_dir_prefix="$TARGET_DIR_PREFIX" \
    "${TARGET_FLAGS[@]}"
