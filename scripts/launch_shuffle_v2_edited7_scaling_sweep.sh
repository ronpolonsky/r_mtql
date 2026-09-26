#!/usr/bin/env bash
# Submit the matched Shuffle edited7 data-scaling sweep.
#
# Four settings x three agents = 12 jobs:
#   150S/45F on the H100/H200 pool (iris-hi)
#   120S/36F on iris-hi regular nodes iris7/9
#    90S/27F on iris regular nodes iris7/9
#   105S/32F on iris regular nodes iris7/9
#
# The failure counts preserve the full-data 30% failure-episode ratio, with
# nearest-integer rounding.  The loader distributes selected episodes nearly
# evenly across the three Shuffle colors.

set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
TRAIN_WRAPPER=${ROOT}/scripts/launch_shuffle_v2_edited7_benchmark.sh
RUN_ROOT=${RUN_ROOT:-/iris/u/ronpo/mtql-runs/shuffle_v2_edited7/scaling_h14s6}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-16}
TRAIN_STEPS=${TRAIN_STEPS:-1000000}
LOG_INTERVAL=${LOG_INTERVAL:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-30000}
CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-100000}
TIME_LIMIT=${TIME_LIMIT:-21-00:00:00}
PROJECT=${PROJECT:-shuffle_v2_edited7_scaling_h14s6}
ENABLE_WANDB=${ENABLE_WANDB:-0}
WANDB_MODE=${WANDB_MODE:-offline}
DRY_RUN=${DRY_RUN:-0}

[[ -f "${TRAIN_WRAPPER}" ]] || { echo "Missing ${TRAIN_WRAPPER}" >&2; exit 2; }
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || {
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
}

job_ids=()

submit_setting() {
  local partition="$1"
  local nodelist="$2"
  local setting="$3"
  local n_succ="$4"
  local n_fails="$5"

  local method name run_dir group exports job_id
  for method in transformer mlp bc; do
    name="shuffle-e7-${method}-s${n_succ}f${n_fails}-h14s6"
    run_dir="${RUN_ROOT}/${setting}/${method}"
    group="shuffle_e7_h14s6_${setting}_${method}"
    exports="ALL,METHOD=${method},H=14,S=6,SEED=${SEED}"
    exports+=",BATCH_SIZE=${BATCH_SIZE},TRAIN_STEPS=${TRAIN_STEPS}"
    exports+=",LOG_INTERVAL=${LOG_INTERVAL},SAVE_INTERVAL=${SAVE_INTERVAL}"
    exports+=",CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP}"
    exports+=",N_SUCC=${n_succ},N_FAILS=${n_fails}"
    exports+=",ENABLE_WANDB=${ENABLE_WANDB},WANDB_MODE=${WANDB_MODE}"
    exports+=",PROJECT=${PROJECT},WANDB_RUN_GROUP=${group},RUN=${run_dir}"

    local -a command=(
      sbatch --parsable
      --account=iris
      --partition="${partition}"
      --nodelist="${nodelist}"
      --nodes=1
      --gres=gpu:1
      --cpus-per-task=16
      --mem=128G
      --time="${TIME_LIMIT}"
      --job-name="${name}"
      --output="${ROOT}/slurm/%x-%j.out"
      --export="${exports}"
      "${TRAIN_WRAPPER}"
    )

    if [[ "${DRY_RUN}" == 1 ]]; then
      printf 'DRY RUN %-42s:' "${name}"
      printf ' %q' "${command[@]}"
      printf '\n'
      continue
    fi

    job_id=$("${command[@]}")
    job_id="${job_id%%;*}"
    job_ids+=("${job_id}")
    printf 'Submitted %-42s job=%s partition=%s nodes=%s data=%sS/%sF\n' \
      "${name}" "${job_id}" "${partition}" "${nodelist}" "${n_succ}" "${n_fails}"
  done
}

cd "${ROOT}"
mkdir -p "${ROOT}/slurm"

# Full ceiling on the dedicated H100/H200 pair.
submit_setting iris-hi 'iris-hgx-[1-2]' full 150 45

# The remaining three matched data scales on the requested regular Iris nodes.
submit_setting iris-hi 'iris[7,9]' s120 120 36
submit_setting iris 'iris[7,9]' s90 90 27
submit_setting iris 'iris[7,9]' s105 105 32

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated 12 Shuffle scaling submissions; no jobs submitted."
else
  echo "Submitted ${#job_ids[@]} Shuffle scaling jobs: ${job_ids[*]}"
  squeue -j "$(IFS=,; echo "${job_ids[*]}")" \
    -o '%.18i %.35j %.10P %.10T %.10M %.8l %R' || true
fi
