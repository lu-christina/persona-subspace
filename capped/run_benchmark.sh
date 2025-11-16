#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_benchmark.sh
# Runs benchmark evaluation for a predefined list of experiment IDs

# ===== Experiment IDs to run =====
EXP_IDS=(
    "layers_32:40-p0.01"
    "layers_32:40-p0.25"
    "layers_32:40-p0.5"
    "layers_32:40-p0.75"
    "layers_32:56-p0.01"
    "layers_32:56-p0.25"
    "layers_32:56-p0.5"
    "layers_32:56-p0.75"
    "layers_52:68-p0.01"
    "layers_52:68-p0.25"
    "layers_52:68-p0.5"
    "layers_52:68-p0.75"
    "layers_52:76-p0.01"
    "layers_52:76-p0.25"
    "layers_52:76-p0.5"
    "layers_52:76-p0.75"
    "layers_56:72-p0.01"
    "layers_56:72-p0.25"
    "layers_56:72-p0.5"
    "layers_56:72-p0.75"
    "layers_64:72-p0.01"
    "layers_64:72-p0.25"
    "layers_64:72-p0.5"
    "layers_64:72-p0.75"
    "layers_64:80-p0.01"
    "layers_64:80-p0.25"
    "layers_64:80-p0.5"
    "layers_64:80-p0.75"
)

# ===== Paths & model =====
MODEL="meta-llama/Llama-3.3-70B-Instruct"
BASEDIR="/workspace/llama-3.3-70b/capped/benchmarks"

# Associative array: cap_from -> config_path
declare -A CONFIGS=(
  ["gemma_role"]="/workspace/llama-3.3-70b/capped/configs/contrast/gemma_role_config.pt"
  ["qwen_role"]="/workspace/llama-3.3-70b/capped/configs/contrast/qwen_role_config.pt"
  ["sonnet_role"]="/workspace/llama-3.3-70b/capped/configs/contrast/sonnet_role_config.pt"
)

# ===== Eval settings =====
TASKS="eq_bench"
LIMIT=1000
SEED=42
FEWSHOT=0
DTYPE="bfloat16"

# ===== vLLM-specific settings =====
TENSOR_PARALLEL=2
GPU_MEM_UTIL=0.95
MAX_MODEL_LEN=2048

# Skip experiments that already have results (set to "true" to enable, empty string to disable)
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

  # Use the predefined experiment IDs
  SELECTED_IDS=("${EXP_IDS[@]}")

  # Filter out experiments that already exist if SKIP_EXISTING is set
  if [[ -n "$SKIP_EXISTING" ]]; then
    FILTERED_IDS=()
    for exp_id in "${SELECTED_IDS[@]}"; do
      # Check if results directory exists for this experiment
      exp_dir="${BASEDIR}/${TASKS}/${CAP_FROM}/${exp_id}"
      if [[ -d "$exp_dir" ]]; then
        echo "Skipping ${exp_id} (already exists: ${exp_dir})"
      else
        FILTERED_IDS+=("$exp_id")
      fi
    done

    if ((${#FILTERED_IDS[@]} == 0)); then
      echo "All experiments already exist for ${CAP_FROM}. Nothing to run."
      continue
    fi

    echo "Found ${#FILTERED_IDS[@]} experiments to run ($((${#SELECTED_IDS[@]} - ${#FILTERED_IDS[@]})) already exist)"
    SELECTED_IDS=("${FILTERED_IDS[@]}")
  fi

  OUTPUT_PATH="${BASEDIR}/${TASKS}/${CAP_FROM}"

  echo ">>> Queue ${CAP_FROM} : ${#SELECTED_IDS[@]} experiments : ${TASKS} -> ${OUTPUT_PATH}/"
  echo "    Experiment IDs: ${SELECTED_IDS[*]}"

  ts -G 2 uv run 2_benchmark_vllm.py \
    --config_filepath "$CFG" \
    --experiment_ids "${SELECTED_IDS[@]}" \
    --model_name "$MODEL" \
    --tasks "$TASKS" \
    --output_dir "$BASEDIR" \
    --cap_from "$CAP_FROM" \
    --dtype "$DTYPE" \
    --limit "$LIMIT" \
    --random_seed "$SEED" \
    --num_fewshot "$FEWSHOT" \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --max_model_len "$MAX_MODEL_LEN"
done

echo ""
echo "Results will be saved to:"
for CAP_FROM in "${!CONFIGS[@]}"; do
  echo "  ${CAP_FROM} -> ${BASEDIR}/${TASKS}/${CAP_FROM}/<experiment_id>/"
done
