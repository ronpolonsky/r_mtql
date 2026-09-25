#!/usr/bin/env bash
# Resume the three Egg v2 H14/S5 agents for exactly 500k additional updates.
#
# The checkpoint source and destination are intentionally identical. Orbax
# therefore restores the complete training state and appends new checkpoints
# to the original checkpoint series. Logs and W&B metadata are written to a
# separate continuation run directory.
set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
DATASET_PATH=/iris/u/ronpo/expo-ft-data/egg_v2
DATASET_CACHE=/iris/u/ronpo/mtql-runs/caches/egg_v2_224
NORM_STATS_PATH=/iris/u/ronpo/expo-ft-data/egg_v2_norm_stats
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py
TRAIN_SCRIPT="${ROOT}/run_egg_real.sh"

PROJECT=real_egg_v2
WANDB_RUN_GROUP=egg_v2_h14s5_resume500k
HIST_LENGTH=14
HIST_STRIDE=5
ADDITIONAL_STEPS=500000
SAVE_INTERVAL=30000
CHECKPOINT_MAX_TO_KEEP=100
TIME_LIMIT=168:00:00
ACCOUNT=nlp
PARTITION=sphinx
NODELIST='sphinx[4-11]'
DRY_RUN="${DRY_RUN:-0}"

TRANSFORMER_RUN=/iris/u/ronpo/mtql-runs/egg_v2/h14s5/transformer/egg_v2_h14s5/mtql_transformer_real_h14_img_sd000_s_17505913.0.20260919_013558
MLP_RUN=/iris/u/ronpo/mtql-runs/egg_v2/h14s5/mlp/egg_v2_h14s5/mtql_mlp_real_h14_img_sd000_s_17505914.0.20260919_014059
BC_RUN=/iris/u/ronpo/mtql-runs/egg_v2/h14s5/bc/egg_v2_h14s5/new_bc_flow_transformer_real_h14_img_sd000_s_17513735.0.20260919_125101

TRANSFORMER_STEP=220000
MLP_STEP=250000
BC_STEP=210000

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_inputs() {
  command -v jq >/dev/null 2>&1 || die "jq is required"
  command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
  [[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] \
    || die "DRY_RUN must be 0 or 1"
  [[ -x "${PYTHON}" ]] || die "Training Python is not executable: ${PYTHON}"
  [[ -x "${TRAIN_SCRIPT}" ]] || die "Training wrapper is not executable: ${TRAIN_SCRIPT}"
  [[ -d "${DATASET_PATH}" ]] || die "Egg v2 dataset is missing: ${DATASET_PATH}"
  [[ -s "${DATASET_CACHE}/metadata.json" ]] \
    || die "Egg v2 cache metadata is missing"
  [[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] \
    || die "Egg v2 normalization statistics are missing"
  [[ -f "${TASK_CONFIG}" ]] || die "Egg v2 task config is missing"

  jq -e \
    --arg dataset "${DATASET_PATH}" \
    --arg stats "${NORM_STATS_PATH}" \
    '.cache_format_version == 1 and
     .dataset_kind == "egg" and .full_finalized_dataset == true and
     .dataset_path == $dataset and .norm_stats_dir == $stats and
     .action_space == "cartesian_velocity" and
     .gripper_action_space == "velocity" and .image_size == 224 and
     .num_episodes == 133 and .num_transitions == 15616 and
     (.episode_manifest_sha256 | length) == 64' \
    "${DATASET_CACHE}/metadata.json" >/dev/null \
    || die "Cache provenance does not match the audited Egg v2 dataset and stats"

  echo "Validated Egg v2 data/cache/norm contract: 133 episodes, 15,616 transitions"
}

validate_origin() {
  local run_dir="$1"
  local expected_agent="$2"
  local flags_file="${run_dir}/flags.json"

  [[ -s "${flags_file}" ]] || die "Original flags are missing: ${flags_file}"
  jq -e \
    --arg dataset "${DATASET_PATH}" \
    --arg cache "${DATASET_CACHE}" \
    --arg stats "${NORM_STATS_PATH}" \
    --arg agent "${expected_agent}" \
    '.seed == 0 and .resume == false and
     .restore_path == null and .restore_epoch == null and
     .checkpoint_dir == null and .save_checkpoints == true and
     .dataset_kind == "egg" and .dataset_path == $dataset and
     .dataset_cache == $cache and .norm_stats_path == $stats and
     .config_task.env_name == "egg_v2" and
     .config_task.control_hz == 10 and
     .config_task.action_space == "cartesian_velocity" and
     .config_task.gripper_action_space == "velocity" and
     .n_succ == -1 and .n_fails == -1 and
     .train_steps == 300000 and .log_interval == 10000 and
     .save_interval == 10000 and .hist_length == 14 and
     .hist_stride == 5 and .action_chunk_size == 25 and
     .image_size == 224 and .p_aug == 1.0 and .cue_mode == "none" and
     .project == "real_egg_v2" and .wandb_run_group == "egg_v2_h14s5" and
     .agent.agent_name == $agent and .agent.batch_size == 16 and
     .agent.action_chunk_size == 25 and .agent.hidden_dim == 256 and
     .agent.encoder == "resnet" and .agent.actor_num_layers == 2 and
     .agent.actor_lr == 0.0001 and .agent.optimizer == "adamw" and
     .agent.adamw_weight_decay == 0.01 and .agent.warmup_steps == 10000 and
     .agent.flow_steps == 10' \
    "${flags_file}" >/dev/null \
    || die "Original run does not match the audited Egg v2 H14/S5 seed-0 configuration: ${run_dir}"

  if [[ "${expected_agent}" == new_bc_flow_transformer_real ]]; then
    jq -e \
      '.dataset_kind == "egg" and .dataset_transitions == 15616 and
       .egg_success_episodes == 100 and .egg_failure_episodes == 33 and
       .egg_success_transitions == 13395 and
       .egg_failure_transitions == 2221 and
       .training_sampling_policy == "success_only" and
       .training_sampling_transitions == 13395' \
      "${run_dir}/data_contract.json" >/dev/null \
      || die "BC origin is not the corrected success-only run: ${run_dir}"
  else
    jq -e \
      '.agent.alpha == 300.0 and .agent.critic_grad_clip == 5.0 and
       .agent.normalize_q_loss == false and
       .agent.tokenization_mode == "per_modality" and
       .agent.attention_entropy_target == [[3.5,3.5],[3.0,3.0]]' \
      "${flags_file}" >/dev/null \
      || die "MTQL origin has unexpected optimization settings: ${run_dir}"
  fi

  echo "Validated original ${expected_agent} run: ${run_dir}"
}

validate_checkpoint() {
  local run_dir="$1"
  local restore_step="$2"
  local expected_agent="$3"
  local checkpoint_root="${run_dir}/checkpoints"
  local checkpoint="${checkpoint_root}/${restore_step}"
  local latest_numeric item data_file

  [[ "${restore_step}" =~ ^[0-9]+$ ]] || die "Invalid restore step: ${restore_step}"
  latest_numeric=$(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' | awk '/^[0-9]+$/' | sort -n | tail -1)
  [[ "${latest_numeric}" == "${restore_step}" ]] \
    || die "Expected latest checkpoint ${restore_step}, found ${latest_numeric}: ${checkpoint_root}"
  [[ -d "${checkpoint}" ]] || die "Checkpoint is missing: ${checkpoint}"

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
    || die "Agent RNG, optimizer state, or network step is missing: ${checkpoint}"
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
      || die "BC checkpoint lacks its success-only contract: ${checkpoint}"
    jq -e \
      '.tree_metadata | keys | any(contains("modules_actor_bc_flow"))' \
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
      || die "MTQL actor/critic/target parameters are incomplete: ${checkpoint}"
    if [[ "${expected_agent}" == mtql_transformer_real ]]; then
      jq -e \
        '.tree_metadata | keys |
         any(contains("modules_attention_entropy_temperature"))' \
        "${checkpoint}/params/_METADATA" >/dev/null \
        || die "Transformer attention-temperature parameters are missing: ${checkpoint}"
    fi
  fi

  echo "Validated complete checkpoint: ${checkpoint}"
}

submitted_jobs=()

submit_resume() {
  local label="$1"
  local agent_config="$2"
  local run_dir="$3"
  local restore_step="$4"
  local target_step=$((restore_step + ADDITIONAL_STEPS))
  local checkpoint_root="${run_dir}/checkpoints"
  local save_base="${run_dir%/egg_v2_h14s5/*}"
  local exports submission job_id

  [[ $((target_step - restore_step)) -eq "${ADDITIONAL_STEPS}" ]] \
    || die "Internal target-step error for ${label}"

  exports="ALL,PYTHON=${PYTHON},DATASET_PATH=${DATASET_PATH}"
  exports+=",DATASET_CACHE=${DATASET_CACHE},NORM_STATS_PATH=${NORM_STATS_PATH}"
  exports+=",TASK_CONFIG=${TASK_CONFIG},PROJECT=${PROJECT}"
  exports+=",WANDB_RUN_GROUP=${WANDB_RUN_GROUP},P_AUG=1.0,SEED=0"
  exports+=",HIST_LENGTH=${HIST_LENGTH},HIST_STRIDE=${HIST_STRIDE}"
  exports+=",BATCH_SIZE=16,TRAIN_STEPS=${target_step},LOG_INTERVAL=10000"
  exports+=",SAVE_INTERVAL=${SAVE_INTERVAL},CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP}"
  exports+=",N_SUCC=-1,N_FAILS=-1,AGENT_CONFIG=${agent_config}"
  exports+=",SAVE_DIR=${save_base},CHECKPOINT_DIR=${checkpoint_root}"
  exports+=",RESTORE_PATH=${checkpoint_root},RESTORE_EPOCH=${restore_step}"
  exports+=",RESUME=1,OVERWRITE=0"

  command=(
    sbatch --parsable
    --account="${ACCOUNT}" --partition="${PARTITION}" --nodelist="${NODELIST}"
    --nodes=1 --gres=gpu:1 --cpus-per-task=8 --mem=128G
    --time="${TIME_LIMIT}" --job-name="egg-v2-res-${label}-h14s5"
    --output="${ROOT}/slurm/%x-%j.out" --export="${exports}"
    "${TRAIN_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'DRY RUN %-12s restore=%s target=%s:' \
      "${label}" "${restore_step}" "${target_step}"
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  submission=$("${command[@]}")
  job_id=${submission%%;*}
  submitted_jobs+=("${job_id}")
  printf 'Submitted %-12s job=%s restore=%s target=%s (+%s) nodes=%s\n' \
    "${label}" "${job_id}" "${restore_step}" "${target_step}" \
    "${ADDITIONAL_STEPS}" "${NODELIST}"
}

cd "${ROOT}"
mkdir -p slurm
validate_inputs

validate_origin "${TRANSFORMER_RUN}" mtql_transformer_real
validate_origin "${MLP_RUN}" mtql_mlp_real
validate_origin "${BC_RUN}" new_bc_flow_transformer_real

validate_checkpoint "${TRANSFORMER_RUN}" "${TRANSFORMER_STEP}" mtql_transformer_real
validate_checkpoint "${MLP_RUN}" "${MLP_STEP}" mtql_mlp_real
validate_checkpoint "${BC_RUN}" "${BC_STEP}" new_bc_flow_transformer_real

submit_resume transformer agents/mtql_transformer_real.py \
  "${TRANSFORMER_RUN}" "${TRANSFORMER_STEP}"
submit_resume mlp agents/mtql_mlp_real.py "${MLP_RUN}" "${MLP_STEP}"
submit_resume bc agents/new_bc_flow_transformer_real.py \
  "${BC_RUN}" "${BC_STEP}"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated all three exact Egg v2 resumes; no jobs submitted."
  exit 0
fi

echo "Submitted all three Egg v2 H14/S5 continuations: ${submitted_jobs[*]}"
job_csv=$(IFS=,; echo "${submitted_jobs[*]}")
squeue -j "${job_csv}" -o '%.18i %.32j %.9P %.2t %.10M %R' || true
