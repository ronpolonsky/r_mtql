#!/usr/bin/env bash
# Submit the 12 matched egg_v1 runs behind the cache-building job.
set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql_candy_scooping
mkdir -p slurm

CACHE_JOB_ID="${CACHE_JOB_ID:-17498522}"
DEPENDENCY="afterok:${CACHE_JOB_ID}"
DRY_RUN="${DRY_RUN:-0}"
DATASET_PATH=/iris/u/ronpo/expo-ft-data/egg_v1
DATASET_CACHE=/iris/u/ronpo/mtql-runs/caches/egg_v1_224
NORM_STATS_PATH=/iris/u/ronpo/expo-ft-data/egg_v1_norm_stats
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/egg.py
PROJECT=real_egg_v1
RUN_ROOT=/iris/u/ronpo/mtql-runs/egg_v1
TRAIN_SCRIPT=run_egg_real.sh

agents=(
  "transformer:agents/mtql_transformer_real.py"
  "mlp:agents/mtql_mlp_real.py"
  "bc:agents/new_bc_flow_transformer_real.py"
)

submit() {
  local account="$1"
  local partition="$2"
  local nodelist="$3"
  local setting="$4"
  local hist_length="$5"
  local hist_stride="$6"
  local agent_label="$7"
  local agent_config="$8"

  local name="egg-${agent_label}-${setting}"
  local save_dir="${RUN_ROOT}/${agent_label}_${setting}"
  local -a command=(sbatch \
    --account="${account}" \
    --partition="${partition}" \
    --nodelist="${nodelist}" \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=120:00:00 \
    --dependency="${DEPENDENCY}" \
    --job-name="${name}" \
    --export="ALL,DATASET_PATH=${DATASET_PATH},DATASET_CACHE=${DATASET_CACHE},NORM_STATS_PATH=${NORM_STATS_PATH},TASK_CONFIG=${TASK_CONFIG},PROJECT=${PROJECT},WANDB_RUN_GROUP=egg_v1,P_AUG=1.0,SEED=0,HIST_LENGTH=${hist_length},HIST_STRIDE=${hist_stride},BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=10000,SAVE_INTERVAL=30000,AGENT_CONFIG=${agent_config},SAVE_DIR=${save_dir}" \
    "${TRAIN_SCRIPT}")
  local output
  if [[ "${DRY_RUN}" == 1 ]]; then
    output='DRY RUN:'
    for argument in "${command[@]}"; do
      printf -v output '%s %q' "${output}" "${argument}"
    done
  else
    output=$("${command[@]}")
  fi
  printf '%s: %s (dependency=%s, nodes=%s)\n' \
    "${name}" "${output}" "${DEPENDENCY}" "${nodelist}"
}

# Primary recommendation: three jobs on the H100/H200 HGX pool.
for entry in "${agents[@]}"; do
  IFS=: read -r label config <<< "${entry}"
  submit iris iris-hi 'iris-hgx-1,iris-hgx-2' h25s7 25 7 "${label}" "${config}"
done

# Under-20 recommendation: one job on each of iris5, iris7, and iris9.
submit iris iris-hi iris5 h19s10 19 10 transformer agents/mtql_transformer_real.py
submit iris iris-hi iris7 h19s10 19 10 mlp agents/mtql_mlp_real.py
submit iris iris-hi iris9 h19s10 19 10 bc agents/new_bc_flow_transformer_real.py

# Increased-stride variants: six jobs distributed by Slurm over sphinx[4-11].
for setting in h25s25 h19s25; do
  if [[ "${setting}" == h25s25 ]]; then
    hist_length=25
  else
    hist_length=19
  fi
  for entry in "${agents[@]}"; do
    IFS=: read -r label config <<< "${entry}"
    submit nlp sphinx 'sphinx[4-11]' "${setting}" "${hist_length}" 25 "${label}" "${config}"
  done
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated all 12 egg_v1 submission commands; no jobs submitted."
else
  echo "Submitted all 12 egg_v1 jobs behind cache job ${CACHE_JOB_ID}."
  squeue -u ronpo -o "%.18i %.32j %.2t %.10M %R" || true
fi
