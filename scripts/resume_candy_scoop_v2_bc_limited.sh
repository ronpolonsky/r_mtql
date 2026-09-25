#!/usr/bin/env bash
set -euo pipefail

# Resume only the two visual BC candy-scoop-v2 runs from their explicitly
# audited 190k Orbax checkpoints. Both jobs are restricted to sphinx[4-11].

MTQL_ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
PYTHONPATH_ROOT=/iris/u/ronpo/projects/expo-ft
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py
DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted
DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache_v2norm
NORM_STATS_PATH=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted

VIS_BC_18_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/bc_h18s30/candy-scoop-real-bc/new_bc_flow_transformer_real_h18_img_sd000_s_17400149.0.20260912_032955/checkpoints
VIS_BC_18_STEP=190000
VIS_BC_20_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2/bc_h20s25/candy-scoop-real-bc/new_bc_flow_transformer_real_h20_img_sd000_s_17400098.0.20260912_030843/checkpoints
VIS_BC_20_STEP=190000

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_inputs() {
  command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable; run from a Slurm login shell."
  command -v jq >/dev/null 2>&1 || die "jq is required for checkpoint and cache validation."
  [[ -x "${PYTHON}" ]] || die "Training Python is not executable: ${PYTHON}"
  [[ -f "${TASK_CONFIG}" ]] || die "Task config is missing: ${TASK_CONFIG}"
  [[ -d "${DATASET_PATH}" ]] || die "v2 dataset is missing: ${DATASET_PATH}"
  [[ -f "${DATASET_CACHE}/metadata.json" ]] || die "v2 cache metadata is missing."
  [[ -f "${NORM_STATS_PATH}/norm_stats.json" ]] || die "v2 normalization statistics are missing."

  jq -e \
    --arg dataset "${DATASET_PATH}" \
    --arg stats "${NORM_STATS_PATH}" \
    '.dataset_path == $dataset and .norm_stats_dir == $stats and
     .num_episodes == 144 and .num_transitions == 38534' \
    "${DATASET_CACHE}/metadata.json" >/dev/null \
    || die "Cache provenance does not match the audited v2 dataset and normalization statistics."
}

validate_origin() {
  local checkpoint_root="$1"
  local hist_length="$2"
  local hist_stride="$3"
  local run_dir="${checkpoint_root%/checkpoints}"
  local flags_file="${run_dir}/flags.json"

  [[ -f "${flags_file}" ]] || die "Original run flags are missing: ${flags_file}"
  jq -e \
    --arg dataset "${DATASET_PATH}" \
    --arg cache "${DATASET_CACHE}" \
    --arg stats "${NORM_STATS_PATH}" \
    --argjson hist_length "${hist_length}" \
    --argjson hist_stride "${hist_stride}" \
    '.seed == 0 and
     .resume == false and
     .restore_path == null and .restore_epoch == null and .checkpoint_dir == null and
     .save_checkpoints == true and
     .dataset_path == $dataset and .dataset_cache == $cache and .norm_stats_path == $stats and
     .n_succ == -1 and .n_fails == -1 and
     .train_steps == 1000000 and .log_interval == 10000 and .save_interval == 10000 and
     .hist_length == $hist_length and .hist_stride == $hist_stride and
     .action_chunk_size == 25 and .image_size == 224 and .p_aug == 1.0 and
     .cue_mode == "visual" and .project == "real_candy_scoop_v2_visual" and
     .config_task.target_seed == 0 and
     .agent.agent_name == "new_bc_flow_transformer_real" and
     .agent.batch_size == 16 and .agent.action_chunk_size == 25 and
     .agent.hidden_dim == 256 and .agent.actor_num_layers == 2 and
     .agent.actor_lr == 0.0001 and .agent.actor_grad_clip == 5.0 and
     .agent.optimizer == "adamw" and .agent.adamw_weight_decay == 0.01 and
     .agent.warmup_steps == 10000 and .agent.flow_steps == 10 and
     .agent.encoder == "resnet" and .agent.dropout_rate == 0.0 and
     .agent.num_heads == 8 and .agent.mlp_ratio == 4' \
    "${flags_file}" >/dev/null \
    || die "Original run is not the expected fresh seed-0 visual BC v2 configuration: ${run_dir}"

  echo "Validated fresh seed-0 BC origin: ${run_dir}"
}

validate_checkpoint() {
  local checkpoint_root="$1"
  local restore_step="$2"
  local checkpoint="${checkpoint_root}/${restore_step}"
  local item
  local data_file

  [[ "${restore_step}" =~ ^[0-9]+$ ]] || die "Invalid restore step: ${restore_step}"
  [[ -d "${checkpoint}" ]] || die "Checkpoint is missing: ${checkpoint}"

  jq -e \
    '.commit_timestamp_nsecs != null and
     .item_handlers.agent != null and
     .item_handlers.params != null and
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
    data_file=$(find "${checkpoint}/${item}/d" -maxdepth 1 -type f -size +0c -print -quit)
    [[ -n "${data_file}" ]] || die "Missing ${item} OCDBT data: ${checkpoint}"
  done

  jq -e \
    '.tree_metadata | keys |
     any(contains("rng")) and
     any(contains("opt_state")) and
     any(contains("network") and contains("step"))' \
    "${checkpoint}/agent/_METADATA" >/dev/null \
    || die "Agent state lacks RNG, optimizer state, or network step: ${checkpoint}"
  jq -e '.tree_metadata | keys | any(contains("network_params"))' \
    "${checkpoint}/params/_METADATA" >/dev/null \
    || die "Model parameters are missing: ${checkpoint}"
  jq -e \
    '.tree_metadata | keys |
     any(contains("numpy_random_state")) and
     any(contains("python_random_state"))' \
    "${checkpoint}/metadata/_METADATA" >/dev/null \
    || die "Python or NumPy RNG state is missing: ${checkpoint}"

  echo "Validated complete BC checkpoint: ${checkpoint}"
}

submitted_jobs=()

resume_bc() {
  local name="$1"
  local hist_length="$2"
  local hist_stride="$3"
  local save_dir="$4"
  local checkpoint_root="$5"
  local restore_step="$6"
  local submission
  local job_id

  export PYTHON
  export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"
  export TASK_CONFIG DATASET_PATH DATASET_CACHE NORM_STATS_PATH
  export PROJECT=real_candy_scoop_v2_visual
  export CUE_MODE=visual
  export P_AUG=1.0
  export SEED=0
  export HIST_LENGTH="${hist_length}"
  export HIST_STRIDE="${hist_stride}"
  export BATCH_SIZE=16
  export TRAIN_STEPS=1000000
  export LOG_INTERVAL=10000
  export SAVE_INTERVAL=10000
  export N_SUCC=-1
  export N_FAILS=-1
  export AGENT_CONFIG=agents/new_bc_flow_transformer_real.py
  export SAVE_DIR="${save_dir}"
  export CHECKPOINT_DIR="${checkpoint_root}"
  export RESTORE_PATH="${checkpoint_root}"
  export RESTORE_EPOCH="${restore_step}"
  export RESUME=1
  unset OVERWRITE CHECKPOINT_KEEP_PERIOD

  echo "Submitting ${name}: exact checkpoint=${checkpoint_root}/${restore_step}, node pool=sphinx[4-11]"
  submission=$(sbatch --parsable \
    --account=nlp \
    --partition=sphinx \
    --nodelist='sphinx[4-11]' \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=120:00:00 \
    --job-name="${name}" \
    --export=ALL \
    run_candy_scoop_real_bc.sh)
  job_id=${submission%%;*}
  submitted_jobs+=("${job_id}")
  echo "Submitted ${name} as ${job_id}"
}

cd "${MTQL_ROOT}"
validate_inputs

# Validate both jobs before submitting either one.
validate_origin "${VIS_BC_18_ROOT}" 18 30
validate_origin "${VIS_BC_20_ROOT}" 20 25
validate_checkpoint "${VIS_BC_18_ROOT}" "${VIS_BC_18_STEP}"
validate_checkpoint "${VIS_BC_20_ROOT}" "${VIS_BC_20_STEP}"

resume_bc resume-v2-lim-vis-bc-h18s30 18 30 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/bc_h18s30 \
  "${VIS_BC_18_ROOT}" "${VIS_BC_18_STEP}"

resume_bc resume-v2-lim-vis-bc-h20s25 20 25 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/bc_h20s25 \
  "${VIS_BC_20_ROOT}" "${VIS_BC_20_STEP}"

echo "Submitted both limited visual BC v2 resume jobs: ${submitted_jobs[*]}"
job_csv=$(IFS=,; echo "${submitted_jobs[*]}")
if ! squeue -j "${job_csv}" -o '%.18i %.35j %.8T %.12M %R'; then
  echo "WARNING: submissions succeeded, but squeue could not query their status." >&2
fi
