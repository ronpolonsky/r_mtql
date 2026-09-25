#!/usr/bin/env bash
# Submit the long-span 50-success / 25-failure separate-batch comparison.
# H=19, S=7 spans roughly 13.3 seconds at the Egg task's 10 Hz control rate.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export N_SUCC="${N_SUCC:-50}"
export N_FAILS="${N_FAILS:-25}"
export HIST_LENGTH="${HIST_LENGTH:-19}"
export HIST_STRIDE="${HIST_STRIDE:-7}"
export SETTING="${SETTING:-n50f25_h19s7}"

exec "${SCRIPT_DIR}/launch_egg_v4_n50f25_success_actor.sh"
