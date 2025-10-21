#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_mmlu_vllm.sh [JOB_NUMBER]
# JOB_NUMBER: 1 for first half of experiments, 2 for second half (default: run all)

JOB_NUMBER="${1:-0}"

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
BASEDIR="/workspace/qwen-3-32b/capped/benchmarks"

# Associative array: cap_from -> config_path
declare -A CONFIGS=(
  ["role_trait"]="/workspace/qwen-3-32b/capped/configs/role_trait_sliding_config.pt"
)

# ===== Eval settings =====
TASKS="mmlu_pro"
LIMIT=100
SEED=16
FEWSHOT=0
DTYPE="float16"

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
export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # Fallback if v1 is still used

# Process each config
for CAP_FROM in "${!CONFIGS[@]}"; do
  CFG="${CONFIGS[$CAP_FROM]}"

  echo "========================================"
  echo "Processing config: ${CAP_FROM}"
  echo "========================================"

  # Read experiment IDs for this config
  readarray -t STEERED_IDS < <(python - "$CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
print("\n".join([e.get("id","") for e in cfg.get("experiments", []) if e.get("id","").lower()!="baseline" and e.get("id","")]))
PY
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

  # Split experiments by job number if specified
  if [[ "$JOB_NUMBER" == "1" ]] || [[ "$JOB_NUMBER" == "2" ]]; then
    TOTAL_EXPS=${#SELECTED_IDS[@]}
    HALF=$((TOTAL_EXPS / 2))

    if [[ "$JOB_NUMBER" == "1" ]]; then
      # First half
      SELECTED_IDS=("${SELECTED_IDS[@]:0:$HALF}")
      echo "Job 1: Running first half (${#SELECTED_IDS[@]} experiments)"
    else
      # Second half
      SELECTED_IDS=("${SELECTED_IDS[@]:$HALF}")
      echo "Job 2: Running second half (${#SELECTED_IDS[@]} experiments)"
    fi
  fi

  OUTPUT_PATH="${BASEDIR}/mmlu_pro/${CAP_FROM}"

  echo ">>> Queue ${CAP_FROM} : ${#SELECTED_IDS[@]} experiments : MMLU-Pro (batch=auto) -> ${OUTPUT_PATH}/"
  echo "    Experiment IDs: ${SELECTED_IDS[*]}"

  ts -G 1 uv run 2_benchmark_vllm.py \
    --config_filepath "$CFG" \
    --experiment_ids "${SELECTED_IDS[@]}" \
    --model_name "$MODEL" \
    --tasks "$TASKS" \
    --output_dir "$BASEDIR" \
    --cap_from "$CAP_FROM" \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --max_model_len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --limit "$LIMIT" \
    --random_seed "$SEED" \
    --num_fewshot "$FEWSHOT"
done

echo ""
echo "Results will be saved to:"
for CAP_FROM in "${!CONFIGS[@]}"; do
  echo "  ${CAP_FROM} -> ${BASEDIR}/mmlu_pro/${CAP_FROM}/<experiment_id>/"
done
