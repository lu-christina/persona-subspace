#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_sixth.sh [JOB_NUMBER]
# JOB_NUMBER: 1-8 to split experiments into eighths (default: run all)

JOB_NUMBER="${1:-0}"

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
BASEDIR="/workspace/qwen-3-32b/capped/sliding_benchmarks"

# Associative array: cap_from -> config_path
declare -A CONFIGS=(
  ["role_trait"]="/workspace/qwen-3-32b/capped/configs/role_trait_sliding_config.pt"
)

# ===== Eval settings =====
TASKS="mmlu_pro"
LIMIT=100
SEED=16
FEWSHOT=0
DTYPE="bfloat16"
BATCH=64

# ===== vLLM-specific settings =====
TENSOR_PARALLEL=1
GPU_MEM_UTIL=0.95
MAX_MODEL_LEN=2048

# Filter pattern for experiment IDs (set to empty string to disable filtering)
FILTER_PATTERN=""
# Skip experiments that already have results (set to empty string to disable skip check)
SKIP_EXISTING=""

# ===== Env & prep =====
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Process each config
for CAP_FROM in "${!CONFIGS[@]}"; do
  CFG="${CONFIGS[$CAP_FROM]}"

  echo "========================================"
  echo "Processing config: ${CAP_FROM}"
  echo "========================================"

  # Hardcoded experiment IDs
  STEERED_IDS=(
    "layers_32:36-p0.75" "layers_34:38-p0.01" "layers_34:38-p0.25" "layers_34:38-p0.5" "layers_34:38-p0.75"
    "layers_36:40-p0.01" "layers_36:40-p0.25" "layers_36:40-p0.5" "layers_36:40-p0.75"
    "layers_38:42-p0.01" "layers_38:42-p0.25" "layers_38:42-p0.5" "layers_38:42-p0.75"
    "layers_40:44-p0.01" "layers_40:44-p0.25" "layers_40:44-p0.5" "layers_40:44-p0.75"
    "layers_42:46-p0.01" "layers_42:46-p0.25" "layers_42:46-p0.5" "layers_42:46-p0.75"
    "layers_44:48-p0.01" "layers_44:48-p0.25" "layers_44:48-p0.5" "layers_44:48-p0.75"
    "layers_46:50-p0.01" "layers_46:50-p0.25" "layers_46:50-p0.5" "layers_46:50-p0.75"
    "layers_38:54-p0.75" "layers_40:56-p0.01" "layers_40:56-p0.25" "layers_40:56-p0.5" "layers_40:56-p0.75"
    "layers_42:58-p0.01" "layers_42:58-p0.25" "layers_42:58-p0.5" "layers_42:58-p0.75"
    "layers_44:60-p0.01" "layers_44:60-p0.25" "layers_44:60-p0.5" "layers_44:60-p0.75"
  )

  # Filter STEERED_IDS if pattern is set
  if [[ -n "$FILTER_PATTERN" ]]; then
    FILTERED_IDS=()
    for exp_id in "${STEERED_IDS[@]}"; do
      [[ "$exp_id" == *"$FILTER_PATTERN"* ]] && FILTERED_IDS+=("$exp_id")
    done

    if ((${#FILTERED_IDS[@]} == 0)); then
      echo "No experiments matched filter \"${FILTER_PATTERN}\" for ${CAP_FROM}."
      continue
    fi
    echo "Filtered to ${#FILTERED_IDS[@]} experiments matching \"${FILTER_PATTERN}\" (from ${#STEERED_IDS[@]} total)"
  else
    FILTERED_IDS=("${STEERED_IDS[@]}")
  fi

  # Split experiments by job number if specified (1-8) - BEFORE skip_existing filter
  if [[ "$JOB_NUMBER" =~ ^[1-8]$ ]]; then
    TOTAL_EXPS=${#FILTERED_IDS[@]}
    EIGHTH=$((TOTAL_EXPS / 8))
    REMAINDER=$((TOTAL_EXPS % 8))

    # Calculate start position and size for this job
    START=0
    for ((i=1; i<JOB_NUMBER; i++)); do
      START=$((START + EIGHTH + (i <= REMAINDER ? 1 : 0)))
    done
    SIZE=$((EIGHTH + (JOB_NUMBER <= REMAINDER ? 1 : 0)))

    FILTERED_IDS=("${FILTERED_IDS[@]:$START:$SIZE}")
    echo "Job ${JOB_NUMBER}: Running eighth #${JOB_NUMBER} (${#FILTERED_IDS[@]} experiments)"
  fi

  # Filter out experiments that already exist if SKIP_EXISTING is set
  if [[ -n "$SKIP_EXISTING" ]]; then
    SELECTED_IDS=()
    for exp_id in "${FILTERED_IDS[@]}"; do
      # Check if results directory exists for this experiment
      exp_dir="${BASEDIR}/mmlu_pro/${CAP_FROM}/${exp_id}"
      if [[ -d "$exp_dir" ]]; then
        echo "Skipping ${exp_id} (already exists: ${exp_dir})"
      else
        SELECTED_IDS+=("$exp_id")
      fi
    done

    if ((${#SELECTED_IDS[@]} == 0)); then
      echo "All experiments already exist for ${CAP_FROM}. Nothing to run."
      continue
    fi

    echo "Found ${#SELECTED_IDS[@]} experiments to run (${#FILTERED_IDS[@]} after filter, $((${#FILTERED_IDS[@]} - ${#SELECTED_IDS[@]})) already exist)"
  else
    SELECTED_IDS=("${FILTERED_IDS[@]}")
  fi

  OUTPUT_PATH="${BASEDIR}/mmlu_pro/${CAP_FROM}"

  echo ">>> Queue ${CAP_FROM} : ${#SELECTED_IDS[@]} experiments : MMLU-Pro (batch=auto) -> ${OUTPUT_PATH}/"
  echo "    Experiment IDs: ${SELECTED_IDS[*]}"

  ts -G 1 uv run 2_benchmark_eval.py \
    --config_filepath "$CFG" \
    --experiment_ids "${SELECTED_IDS[@]}" \
    --model_name "$MODEL" \
    --tasks "$TASKS" \
    --output_dir "$BASEDIR" \
    --cap_from "$CAP_FROM" \
    --torch_dtype "$DTYPE" \
    --limit "$LIMIT" \
    --random_seed "$SEED" \
    --num_fewshot "$FEWSHOT" \
    --batch_size "$BATCH"
done

echo ""
echo "Results will be saved to:"
for CAP_FROM in "${!CONFIGS[@]}"; do
  echo "  ${CAP_FROM} -> ${BASEDIR}/mmlu_pro/${CAP_FROM}/<experiment_id>/"
done
