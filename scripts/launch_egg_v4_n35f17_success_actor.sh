#!/usr/bin/env bash
# Submit the 35-success / 17-failure Transformer and MLP separate-batch jobs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export N_SUCC="${N_SUCC:-35}"
export N_FAILS="${N_FAILS:-17}"
export SETTING="${SETTING:-n35f17}"
export NODELIST="${NODELIST:-iris[5,7,9]}"

exec "${SCRIPT_DIR}/launch_egg_v4_n50f25_success_actor.sh"
