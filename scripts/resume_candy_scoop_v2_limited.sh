#!/usr/bin/env bash
set -euo pipefail

# Resume the eight non-BC candy-scoop-v2 runs from explicitly audited Orbax
# checkpoints. Placement:
#   * Iris HGX: all four language runs and both visual Transformer runs.
#   * Sphinx:   both visual MLP runs.

MTQL_ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
PYTHONPATH_ROOT=/iris/u/ronpo/projects/expo-ft
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py
DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted
DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache_v2norm
NORM_STATS_PATH=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted

VIS_T_18_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/transformer_h18s30/candy-scoop-real/mtql_transformer_real_h18_img_sd000_s_17400147.0.20260912_031946/checkpoints
VIS_T_18_STEP=110000
VIS_MLP_18_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/mlp_h18s30/candy-scoop-real/mtql_mlp_real_h18_img_sd000_s_17400148.0.20260912_032455/checkpoints
VIS_MLP_18_STEP=130000
LANG_T_18_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h18s30/candy-scoop-real/mtql_transformer_language_real_h18_img_sd000_s_17400150.0.20260912_032956/checkpoints
LANG_T_18_STEP=130000
LANG_MLP_18_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h18s30/candy-scoop-real/mtql_mlp_language_real_h18_img_sd000_s_17400151.0.20260912_033454/checkpoints
LANG_MLP_18_STEP=150000
VIS_T_20_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2/transformer_h20s25/candy-scoop-real/mtql_transformer_real_h20_img_sd000_s_17400096.0.20260912_030843/checkpoints
VIS_T_20_STEP=140000
VIS_MLP_20_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2/mlp_h20s25/candy-scoop-real/mtql_mlp_real_h20_img_sd000_s_17400097.0.20260912_030843/checkpoints
VIS_MLP_20_STEP=150000
LANG_T_20_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h20s25/candy-scoop-real/mtql_transformer_language_real_h20_img_sd000_s_17400133.0.20260912_031412/checkpoints
LANG_T_20_STEP=90000
LANG_MLP_20_ROOT=/iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h20s25/candy-scoop-real/mtql_mlp_language_real_h20_img_sd000_s_17400134.0.20260912_031342/checkpoints
LANG_MLP_20_STEP=100000

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
  local cue_mode="$2"
  local hist_length="$3"
  local hist_stride="$4"
  local agent_name="$5"
  local project="$6"
  local run_dir="${checkpoint_root%/checkpoints}"
  local flags_file="${run_dir}/flags.json"

  [[ -f "${flags_file}" ]] || die "Original run flags are missing: ${flags_file}"
  jq -e \
    --arg dataset "${DATASET_PATH}" \
    --arg cache "${DATASET_CACHE}" \
    --arg stats "${NORM_STATS_PATH}" \
    --arg cue "${cue_mode}" \
    --arg agent "${agent_name}" \
    --arg project "${project}" \
    --argjson hist_length "${hist_length}" \
    --argjson hist_stride "${hist_stride}" \
    '.seed == 0 and
     .resume == false and
     .restore_path == null and
     .restore_epoch == null and
     .checkpoint_dir == null and
     .save_checkpoints == true and
     .dataset_path == $dataset and
     .dataset_cache == $cache and
     .norm_stats_path == $stats and
     .n_succ == -1 and .n_fails == -1 and
     .train_steps == 1000000 and
     .log_interval == 10000 and .save_interval == 10000 and
     .hist_length == $hist_length and .hist_stride == $hist_stride and
     .action_chunk_size == 25 and .image_size == 224 and .p_aug == 1.0 and
     .cue_mode == $cue and .project == $project and
     .config_task.target_seed == 0 and
     .agent.agent_name == $agent and
     .agent.batch_size == 16 and
     .agent.alpha == 300.0 and
     .agent.critic_grad_clip == 5.0 and
     .agent.normalize_q_loss == false and
     .agent.hidden_dim == 256 and
     .agent.tokenization_mode == "per_modality" and
     .agent.encoder == "resnet" and
     .agent.attention_entropy_target == [[3.5, 3.5], [3.0, 3.0]]' \
    "${flags_file}" >/dev/null \
    || die "Original run is not the expected fresh seed-0 v2 configuration: ${run_dir}"

  echo "Validated fresh seed-0 origin: ${run_dir}"
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
  jq -e \
    '.tree_metadata | keys | any(contains("network_params"))' \
    "${checkpoint}/params/_METADATA" >/dev/null \
    || die "Model parameters are missing: ${checkpoint}"
  jq -e \
    '.tree_metadata | keys |
     any(contains("numpy_random_state")) and
     any(contains("python_random_state"))' \
    "${checkpoint}/metadata/_METADATA" >/dev/null \
    || die "Python or NumPy RNG state is missing: ${checkpoint}"

  echo "Validated complete checkpoint: ${checkpoint}"
}

submitted_jobs=()

resume() {
  local name="$1"
  local account="$2"
  local partition="$3"
  local nodelist="$4"
  local project="$5"
  local cue_mode="$6"
  local hist_length="$7"
  local hist_stride="$8"
  local agent="$9"
  local save_dir="${10}"
  local checkpoint_root="${11}"
  local restore_step="${12}"
  local script="${13}"
  local submission
  local job_id

  export PYTHON
  export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"
  export TASK_CONFIG DATASET_PATH DATASET_CACHE NORM_STATS_PATH
  export PROJECT="${project}"
  export CUE_MODE="${cue_mode}"
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
  export AGENT_CONFIG="${agent}"
  export SAVE_DIR="${save_dir}"
  export CHECKPOINT_DIR="${checkpoint_root}"
  export RESTORE_PATH="${checkpoint_root}"
  export RESTORE_EPOCH="${restore_step}"
  export RESUME=1
  unset OVERWRITE CHECKPOINT_KEEP_PERIOD

  echo "Submitting ${name}: exact checkpoint=${checkpoint_root}/${restore_step}, node pool=${nodelist}"
  submission=$(sbatch --parsable \
    --account="${account}" \
    --partition="${partition}" \
    --nodelist="${nodelist}" \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=120:00:00 \
    --job-name="${name}" \
    --export=ALL \
    "${script}")
  job_id=${submission%%;*}
  submitted_jobs+=("${job_id}")
  echo "Submitted ${name} as ${job_id}"
}

cd "${MTQL_ROOT}"
validate_inputs

# Validate every checkpoint before making any Slurm submission.
validate_origin "${VIS_T_18_ROOT}" visual 18 30 mtql_transformer_real real_candy_scoop_v2_visual
validate_origin "${VIS_MLP_18_ROOT}" visual 18 30 mtql_mlp_real real_candy_scoop_v2_visual
validate_origin "${LANG_T_18_ROOT}" language 18 30 mtql_transformer_language_real real_candy_scoop_v2_language
validate_origin "${LANG_MLP_18_ROOT}" language 18 30 mtql_mlp_language_real real_candy_scoop_v2_language
validate_origin "${VIS_T_20_ROOT}" visual 20 25 mtql_transformer_real real_candy_scoop_v2_visual
validate_origin "${VIS_MLP_20_ROOT}" visual 20 25 mtql_mlp_real real_candy_scoop_v2_visual
validate_origin "${LANG_T_20_ROOT}" language 20 25 mtql_transformer_language_real real_candy_scoop_v2_language
validate_origin "${LANG_MLP_20_ROOT}" language 20 25 mtql_mlp_language_real real_candy_scoop_v2_language

validate_checkpoint "${VIS_T_18_ROOT}" "${VIS_T_18_STEP}"
validate_checkpoint "${VIS_MLP_18_ROOT}" "${VIS_MLP_18_STEP}"
validate_checkpoint "${LANG_T_18_ROOT}" "${LANG_T_18_STEP}"
validate_checkpoint "${LANG_MLP_18_ROOT}" "${LANG_MLP_18_STEP}"
validate_checkpoint "${VIS_T_20_ROOT}" "${VIS_T_20_STEP}"
validate_checkpoint "${VIS_MLP_20_ROOT}" "${VIS_MLP_20_STEP}"
validate_checkpoint "${LANG_T_20_ROOT}" "${LANG_T_20_STEP}"
validate_checkpoint "${LANG_MLP_20_ROOT}" "${LANG_MLP_20_STEP}"

# Iris HGX: all language runs plus the two visual Transformer runs.
resume resume-v2-lim-lang-transformer-h18s30 iris iris-hi 'iris-hgx-1,iris-hgx-2' \
  real_candy_scoop_v2_language language 18 30 agents/mtql_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h18s30 \
  "${LANG_T_18_ROOT}" "${LANG_T_18_STEP}" run_candy_scoop_real.sh

resume resume-v2-lim-lang-mlp-h18s30 iris iris-hi 'iris-hgx-1,iris-hgx-2' \
  real_candy_scoop_v2_language language 18 30 agents/mtql_mlp_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h18s30 \
  "${LANG_MLP_18_ROOT}" "${LANG_MLP_18_STEP}" run_candy_scoop_real.sh

resume resume-v2-lim-lang-transformer-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' \
  real_candy_scoop_v2_language language 20 25 agents/mtql_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h20s25 \
  "${LANG_T_20_ROOT}" "${LANG_T_20_STEP}" run_candy_scoop_real.sh

resume resume-v2-lim-lang-mlp-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' \
  real_candy_scoop_v2_language language 20 25 agents/mtql_mlp_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h20s25 \
  "${LANG_MLP_20_ROOT}" "${LANG_MLP_20_STEP}" run_candy_scoop_real.sh

resume resume-v2-lim-vis-transformer-h18s30 iris iris-hi 'iris-hgx-1,iris-hgx-2' \
  real_candy_scoop_v2_visual visual 18 30 agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/transformer_h18s30 \
  "${VIS_T_18_ROOT}" "${VIS_T_18_STEP}" run_candy_scoop_real.sh

resume resume-v2-lim-vis-transformer-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' \
  real_candy_scoop_v2_visual visual 20 25 agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/transformer_h20s25 \
  "${VIS_T_20_ROOT}" "${VIS_T_20_STEP}" run_candy_scoop_real.sh

# Sphinx: the two visual MLP runs.
resume resume-v2-lim-vis-mlp-h18s30 nlp sphinx 'sphinx[4-11]' \
  real_candy_scoop_v2_visual visual 18 30 agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/mlp_h18s30 \
  "${VIS_MLP_18_ROOT}" "${VIS_MLP_18_STEP}" run_candy_scoop_real.sh

resume resume-v2-lim-vis-mlp-h20s25 nlp sphinx 'sphinx[4-11]' \
  real_candy_scoop_v2_visual visual 20 25 agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/mlp_h20s25 \
  "${VIS_MLP_20_ROOT}" "${VIS_MLP_20_STEP}" run_candy_scoop_real.sh

echo "Submitted all eight limited v2 resume jobs: ${submitted_jobs[*]}"
job_csv=$(IFS=,; echo "${submitted_jobs[*]}")
if ! squeue -j "${job_csv}" -o '%.18i %.35j %.8T %.12M %R'; then
  echo "WARNING: submissions succeeded, but squeue could not query their status." >&2
fi
