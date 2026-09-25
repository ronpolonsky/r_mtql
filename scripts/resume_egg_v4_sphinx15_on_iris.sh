#!/usr/bin/env bash
# Resume the 15 canceled Egg v4 jobs on high-memory Iris GPUs.
#
# The original checkpoint directories remain both the restore source and the
# checkpoint destination. Each job therefore restores model parameters,
# optimizer state, agent/Python/NumPy RNG state, and the exact network step,
# then appends checkpoints to its original 30k-spaced series through 1M.
#
# Validate without submitting:
#   DRY_RUN=1 ./scripts/resume_egg_v4_sphinx15_on_iris.sh
#
# Submit after the storage preflight passes:
#   ./scripts/resume_egg_v4_sphinx15_on_iris.sh
set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
DATASET_PATH=/iris/u/ronpo/expo-ft-data/egg_v4
DATASET_CACHE=/iris/u/ronpo/mtql-runs/caches/egg_v4_224
NORM_STATS_PATH=/iris/u/ronpo/expo-ft-data/egg_v4_norm_stats
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py
TRAIN_SCRIPT="${ROOT}/run_egg_real.sh"
RUN_ROOT=/iris/u/ronpo/mtql-runs/egg_v4

ACCOUNT=iris
PARTITION=iris
# Use only high-memory Iris GPUs: A40, A6000, L40S, H100, or H200.
# iris8 is excluded because its A4500 GPUs have only 20 GB.
NODELIST="${NODELIST:-iris[5-7,9-10],iris-hgx-[1-2]}"
TIME_LIMIT=168:00:00

TRAIN_STEPS=1000000
LOG_INTERVAL=10000
SAVE_INTERVAL=30000
CHECKPOINT_MAX_TO_KEEP=100000
HIST_LENGTH=14
HIST_STRIDE=5
BATCH_SIZE=16

DRY_RUN="${DRY_RUN:-0}"
ALLOW_LOW_SPACE="${ALLOW_LOW_SPACE:-0}"
STORAGE_MARGIN_GIB=30

# setting:n_success:n_failure:label:agent-config:agent-name:original-job-id
matrix=(
  'n10f5:10:5:transformer:agents/mtql_transformer_real.py:mtql_transformer_real:17525935'
  'n10f5:10:5:mlp:agents/mtql_mlp_real.py:mtql_mlp_real:17525936'
  'n10f5:10:5:bc:agents/new_bc_flow_transformer_real.py:new_bc_flow_transformer_real:17525937'
  'n15f8:15:8:transformer:agents/mtql_transformer_real.py:mtql_transformer_real:17525938'
  'n15f8:15:8:mlp:agents/mtql_mlp_real.py:mtql_mlp_real:17525939'
  'n15f8:15:8:bc:agents/new_bc_flow_transformer_real.py:new_bc_flow_transformer_real:17525940'
  'n20f10:20:10:transformer:agents/mtql_transformer_real.py:mtql_transformer_real:17525941'
  'n20f10:20:10:mlp:agents/mtql_mlp_real.py:mtql_mlp_real:17525942'
  'n20f10:20:10:bc:agents/new_bc_flow_transformer_real.py:new_bc_flow_transformer_real:17525943'
  'n35f17:35:17:transformer:agents/mtql_transformer_real.py:mtql_transformer_real:17525944'
  'n35f17:35:17:mlp:agents/mtql_mlp_real.py:mtql_mlp_real:17525945'
  'n35f17:35:17:bc:agents/new_bc_flow_transformer_real.py:new_bc_flow_transformer_real:17525946'
  'n70f35:70:35:transformer:agents/mtql_transformer_real.py:mtql_transformer_real:17525947'
  'n70f35:70:35:mlp:agents/mtql_mlp_real.py:mtql_mlp_real:17525948'
  'n70f35:70:35:bc:agents/new_bc_flow_transformer_real.py:new_bc_flow_transformer_real:17525949'
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_inputs() {
  [[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] \
    || die "DRY_RUN must be 0 or 1"
  [[ "${ALLOW_LOW_SPACE}" == 0 || "${ALLOW_LOW_SPACE}" == 1 ]] \
    || die "ALLOW_LOW_SPACE must be 0 or 1"
  [[ "${#matrix[@]}" -eq 15 ]] || die "Expected exactly 15 resume jobs"
  command -v jq >/dev/null 2>&1 || die "jq is required"
  command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
  command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"
  command -v scontrol >/dev/null 2>&1 || die "scontrol is unavailable"
  [[ -x "${PYTHON}" ]] || die "Training Python is not executable: ${PYTHON}"
  [[ -x "${TRAIN_SCRIPT}" ]] || die "Training wrapper is missing: ${TRAIN_SCRIPT}"
  [[ -d "${DATASET_PATH}" ]] || die "Egg v4 dataset is missing: ${DATASET_PATH}"
  [[ -s "${DATASET_CACHE}/metadata.json" ]] || die "Egg v4 cache is missing"
  [[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] \
    || die "Egg v4 norm stats are missing"
  [[ -f "${TASK_CONFIG}" ]] || die "Egg task configuration is missing"
  scontrol show hostnames "${NODELIST}" >/dev/null \
    || die "Invalid Iris node list: ${NODELIST}"

  "${PYTHON}" - "${DATASET_CACHE}/metadata.json" \
    "${DATASET_PATH}" "${NORM_STATS_PATH}" <<'PY'
import json
import os
import sys

metadata_path, dataset_path, norm_stats_path = sys.argv[1:]
with open(metadata_path) as handle:
    metadata = json.load(handle)

expected = {
    "cache_format_version": 1,
    "dataset_kind": "egg",
    "full_finalized_dataset": True,
    "dataset_path": os.path.realpath(dataset_path),
    "norm_stats_dir": os.path.realpath(norm_stats_path),
    "action_space": "cartesian_velocity",
    "gripper_action_space": "velocity",
    "image_size": 224,
    "num_episodes": 140,
    "num_transitions": 15799,
}
mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
}
if mismatches:
    raise SystemExit(f"Egg v4 cache validation failed: {mismatches}")
print("Validated Egg v4 data/cache/norm contract: 140 episodes, 15,799 transitions")
PY
}

assert_no_duplicate_resumes() {
  local active
  active=$(squeue -h -u ronpo -o '%j' | awk '/^egg-v4-res-/ {print}')
  [[ -z "${active}" ]] \
    || die "Egg v4 Iris resume jobs already exist; refusing duplicates: ${active}"
}

resolve_run() {
  local setting="$1"
  local label="$2"
  local job_id="$3"
  local -a matches

  shopt -s nullglob
  matches=("${RUN_ROOT}/${setting}/${label}"/*/*_s_"${job_id}".0.*)
  shopt -u nullglob
  [[ "${#matches[@]}" -eq 1 ]] \
    || die "Expected one origin for ${setting}/${label} job ${job_id}; found ${#matches[@]}"
  printf '%s\n' "${matches[0]}"
}

validate_origin() {
  local run_dir="$1"
  local setting="$2"
  local n_succ="$3"
  local n_fails="$4"
  local expected_agent="$5"
  local flags_file="${run_dir}/flags.json"

  [[ -s "${flags_file}" ]] || die "Original flags are missing: ${flags_file}"
  jq -e \
    --arg dataset "${DATASET_PATH}" \
    --arg cache "${DATASET_CACHE}" \
    --arg stats "${NORM_STATS_PATH}" \
    --arg project "real_egg_v4" \
    --arg group "egg_v4_${setting}" \
    --arg agent "${expected_agent}" \
    --argjson n_succ "${n_succ}" \
    --argjson n_fails "${n_fails}" \
    '.seed == 0 and .resume == false and
     .restore_path == null and .restore_epoch == null and
     .checkpoint_dir == null and .save_checkpoints == true and
     .dataset_kind == "egg" and .dataset_path == $dataset and
     .dataset_cache == $cache and .norm_stats_path == $stats and
     .config_task.env_name == "egg_v2" and
     .config_task.control_hz == 10 and
     .config_task.action_space == "cartesian_velocity" and
     .config_task.gripper_action_space == "velocity" and
     .n_succ == $n_succ and .n_fails == $n_fails and
     .train_steps == 1000000 and .log_interval == 10000 and
     .save_interval == 30000 and .checkpoint_max_to_keep == 100000 and
     .hist_length == 14 and .hist_stride == 5 and
     .action_chunk_size == 25 and .image_size == 224 and
     .p_aug == 1.0 and .cue_mode == "none" and
     .project == $project and .wandb_run_group == $group and
     .agent.agent_name == $agent and .agent.batch_size == 16 and
     .agent.action_chunk_size == 25 and .agent.hidden_dim == 256 and
     .agent.encoder == "resnet"' \
    "${flags_file}" >/dev/null \
    || die "Origin configuration mismatch: ${run_dir}"

  if [[ "${expected_agent}" == new_bc_flow_transformer_real ]]; then
    jq -e \
      --argjson n_succ "${n_succ}" \
      --argjson n_fails "${n_fails}" \
      '.dataset_requested_n_succ == $n_succ and
       .dataset_requested_n_fails == $n_fails and
       .egg_success_episodes == $n_succ and
       .egg_failure_episodes == $n_fails and
       .training_sampling_policy == "success_only" and
       .training_sampling_transitions == .egg_success_transitions' \
      "${run_dir}/data_contract.json" >/dev/null \
      || die "BC origin lacks its exact success-only contract: ${run_dir}"
  else
    jq -e \
      '.agent.alpha == 300.0 and .agent.critic_grad_clip == 5.0 and
       .agent.normalize_q_loss == false and
       .agent.tokenization_mode == "per_modality" and
       .agent.attention_entropy_target == [[3.5,3.5],[3.0,3.0]]' \
      "${flags_file}" >/dev/null \
      || die "MTQL optimization settings differ from the origin: ${run_dir}"
  fi
}

validate_checkpoint() {
  local run_dir="$1"
  local expected_agent="$2"
  local checkpoint_root="${run_dir}/checkpoints"
  local restore_step checkpoint item data_file

  restore_step=$(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 \
    -type d -printf '%f\n' | awk '/^[0-9]+$/' | sort -n | tail -1)
  [[ "${restore_step}" =~ ^[0-9]+$ ]] \
    || die "No numeric checkpoint found: ${checkpoint_root}"
  (( restore_step < TRAIN_STEPS )) \
    || die "Checkpoint is already at or beyond ${TRAIN_STEPS}: ${checkpoint_root}/${restore_step}"
  checkpoint="${checkpoint_root}/${restore_step}"

  jq -e \
    '.commit_timestamp_nsecs != null and
     .item_handlers.agent != null and .item_handlers.params != null and
     .item_handlers.metadata != null' \
    "${checkpoint}/_CHECKPOINT_METADATA" >/dev/null \
    || die "Checkpoint is not a finalized three-item Orbax commit: ${checkpoint}"

  for item in agent params metadata; do
    [[ -s "${checkpoint}/${item}/_METADATA" ]] \
      || die "Missing ${item} tree metadata: ${checkpoint}"
    [[ -s "${checkpoint}/${item}/manifest.ocdbt" ]] \
      || die "Missing ${item} OCDBT manifest: ${checkpoint}"
    [[ -s "${checkpoint}/${item}/ocdbt.process_0/manifest.ocdbt" ]] \
      || die "Missing ${item} process manifest: ${checkpoint}"
    data_file=$(find "${checkpoint}/${item}/d" -maxdepth 1 -type f \
      -size +0c -print -quit)
    [[ -n "${data_file}" ]] || die "Missing ${item} OCDBT data: ${checkpoint}"
  done

  jq -e \
    '.tree_metadata | keys |
     any(contains("rng")) and any(contains("opt_state")) and
     any(contains("network") and contains("step"))' \
    "${checkpoint}/agent/_METADATA" >/dev/null \
    || die "Agent RNG, optimizer, or network step is missing: ${checkpoint}"
  jq -e '.tree_metadata | keys | any(contains("network_params"))' \
    "${checkpoint}/params/_METADATA" >/dev/null \
    || die "Model parameters are missing: ${checkpoint}"
  jq -e \
    '.tree_metadata | keys |
     any(contains("numpy_random_state")) and
     any(contains("python_random_state"))' \
    "${checkpoint}/metadata/_METADATA" >/dev/null \
    || die "Python or NumPy RNG state is missing: ${checkpoint}"

  if [[ "${expected_agent}" == new_bc_flow_transformer_real ]]; then
    jq -e '.training_sampling_policy == "success_only"' \
      "${checkpoint}/metadata/_strings.json" >/dev/null \
      || die "BC checkpoint lacks the success-only contract: ${checkpoint}"
    jq -e '.tree_metadata | keys | any(contains("modules_actor_bc_flow"))' \
      "${checkpoint}/params/_METADATA" >/dev/null \
      || die "BC actor parameters are missing: ${checkpoint}"
  else
    jq -e \
      '.tree_metadata | keys |
       any(contains("modules_actor_bc_flow")) and
       any(contains("modules_actor_onestep_flow")) and
       any(contains("modules_critic")) and
       any(contains("modules_target_critic"))' \
      "${checkpoint}/params/_METADATA" >/dev/null \
      || die "MTQL actor/critic/target state is incomplete: ${checkpoint}"
    if [[ "${expected_agent}" == mtql_transformer_real ]]; then
      jq -e \
        '.tree_metadata | keys |
         any(contains("modules_attention_entropy_temperature"))' \
        "${checkpoint}/params/_METADATA" >/dev/null \
        || die "Transformer entropy-temperature state is missing: ${checkpoint}"
    fi
  fi

  printf '%s\n' "${restore_step}"
}

# Estimate all future checkpoint writes for both the six jobs still running
# and these 15 resumes, using each series' actual latest checkpoint size.
validate_storage() {
  local root latest size_kib remaining future_kib=0 series=0
  local available_kib margin_kib required_kib available_gib required_gib

  shopt -s nullglob
  for root in "${RUN_ROOT}"/*/*/*/*_s_175259*.*/checkpoints; do
    latest=$(find "${root}" -mindepth 1 -maxdepth 1 -type d \
      -printf '%f\n' | awk '/^[0-9]+$/' | sort -n | tail -1)
    [[ "${latest}" =~ ^[0-9]+$ ]] || continue
    (( latest < TRAIN_STEPS )) || continue
    size_kib=$(du -sk "${root}/${latest}" | awk '{print $1}')
    if (( latest <= 990000 )); then
      remaining=$(((990000 - latest) / SAVE_INTERVAL + 1))
    else
      remaining=1
    fi
    future_kib=$((future_kib + size_kib * remaining))
    series=$((series + 1))
  done
  shopt -u nullglob

  [[ "${series}" -eq 21 ]] \
    || die "Storage audit expected 21 active checkpoint series; found ${series}"
  available_kib=$(df -Pk "${RUN_ROOT}" | awk 'NR == 2 {print $4}')
  margin_kib=$((STORAGE_MARGIN_GIB * 1024 * 1024))
  required_kib=$((future_kib + margin_kib))
  available_gib=$((available_kib / 1024 / 1024))
  required_gib=$(((required_kib + 1024 * 1024 - 1) / 1024 / 1024))
  echo "Storage preflight: ${available_gib} GiB available; approximately ${required_gib} GiB required for all remaining checkpoints plus margin"

  if (( available_kib < required_kib )); then
    if [[ "${DRY_RUN}" == 1 ]]; then
      echo "DRY RUN WARNING: storage is insufficient to finish all 21 keep-all series."
    elif [[ "${ALLOW_LOW_SPACE}" != 1 ]]; then
      die "Insufficient storage; free space or explicitly set ALLOW_LOW_SPACE=1"
    else
      echo "WARNING: resuming despite insufficient projected storage." >&2
    fi
  fi
}

declare -a resolved_rows=()
declare -a submitted_jobs=()

validate_all_origins_and_checkpoints() {
  local entry setting n_succ n_fails label agent_config expected_agent job_id
  local run_dir restore_step

  for entry in "${matrix[@]}"; do
    IFS=: read -r setting n_succ n_fails label agent_config expected_agent job_id \
      <<< "${entry}"
    run_dir=$(resolve_run "${setting}" "${label}" "${job_id}")
    validate_origin "${run_dir}" "${setting}" "${n_succ}" "${n_fails}" \
      "${expected_agent}"
    restore_step=$(validate_checkpoint "${run_dir}" "${expected_agent}")
    resolved_rows+=("${setting}:${n_succ}:${n_fails}:${label}:${agent_config}:${run_dir}:${restore_step}")
    printf 'Validated %-12s %-11s checkpoint=%s\n' \
      "${setting}" "${label}" "${restore_step}"
  done
}

submit_all() {
  local row setting n_succ n_fails label agent_config run_dir restore_step
  local checkpoint_root save_dir exports submission job_id name
  local -a command

  for row in "${resolved_rows[@]}"; do
    IFS=: read -r setting n_succ n_fails label agent_config run_dir restore_step \
      <<< "${row}"
    checkpoint_root="${run_dir}/checkpoints"
    save_dir="${RUN_ROOT}/${setting}/${label}"
    name="egg-v4-res-${label}-${setting}"

    exports="ALL,PYTHON=${PYTHON},DATASET_PATH=${DATASET_PATH}"
    exports+=",DATASET_CACHE=${DATASET_CACHE},NORM_STATS_PATH=${NORM_STATS_PATH}"
    exports+=",TASK_CONFIG=${TASK_CONFIG},PROJECT=real_egg_v4"
    exports+=",WANDB_RUN_GROUP=egg_v4_${setting}_resume_iris,P_AUG=1.0,SEED=0"
    exports+=",HIST_LENGTH=${HIST_LENGTH},HIST_STRIDE=${HIST_STRIDE}"
    exports+=",BATCH_SIZE=${BATCH_SIZE},TRAIN_STEPS=${TRAIN_STEPS}"
    exports+=",LOG_INTERVAL=${LOG_INTERVAL},SAVE_INTERVAL=${SAVE_INTERVAL}"
    exports+=",CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP}"
    exports+=",N_SUCC=${n_succ},N_FAILS=${n_fails},AGENT_CONFIG=${agent_config}"
    exports+=",SAVE_DIR=${save_dir},CHECKPOINT_DIR=${checkpoint_root}"
    exports+=",RESTORE_PATH=${checkpoint_root},RESTORE_EPOCH=${restore_step}"
    exports+=",RESUME=1,OVERWRITE=0"

    command=(
      sbatch --parsable --account="${ACCOUNT}" --partition="${PARTITION}"
      --nodelist="${NODELIST}" --nodes=1 --gres=gpu:1
      --cpus-per-task=8 --mem=128G --time="${TIME_LIMIT}"
      --job-name="${name}" --output="${ROOT}/slurm/%x-%j.out"
      --export="${exports}" "${TRAIN_SCRIPT}"
    )

    if [[ "${DRY_RUN}" == 1 ]]; then
      printf 'DRY RUN %-29s restore=%s target=%s:' \
        "${name}" "${restore_step}" "${TRAIN_STEPS}"
      printf ' %q' "${command[@]}"
      printf '\n'
      continue
    fi

    submission=$("${command[@]}")
    job_id=${submission%%;*}
    submitted_jobs+=("${job_id}")
    printf 'Submitted %-29s job=%s restore=%s target=%s\n' \
      "${name}" "${job_id}" "${restore_step}" "${TRAIN_STEPS}"
  done
}

cd "${ROOT}"
mkdir -p slurm
validate_inputs
assert_no_duplicate_resumes
validate_all_origins_and_checkpoints
validate_storage
submit_all

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated all 15 exact Egg v4 Iris resumes; no jobs submitted."
  exit 0
fi

[[ "${#submitted_jobs[@]}" -eq 15 ]] \
  || die "Expected 15 submitted jobs, got ${#submitted_jobs[@]}"
echo "Submitted all 15 Egg v4 Iris resumes: ${submitted_jobs[*]}"
job_csv=$(IFS=,; echo "${submitted_jobs[*]}")
squeue -j "${job_csv}" -o '%.18i %.34j %.10P %.2t %.10M %R' || true
