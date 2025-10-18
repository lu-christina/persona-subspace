#!/usr/bin/env bash
set -euo pipefail

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
BASEDIR="/workspace/qwen-3-32b/capped/benchmarks"

# Associative array: cap_from -> config_path
declare -A CONFIGS=(
  ["role_trait"]="/workspace/qwen-3-32b/capped/configs/role_trait_eighths_config.pt"
)

# ===== Eval settings =====
TASKS="mmlu_pro"
LIMIT=100
SEED=42
FEWSHOT=0
DTYPE="bfloat16"

# ===== vLLM-specific settings =====
TENSOR_PARALLEL=1
GPU_MEM_UTIL=0.95
MAX_MODEL_LEN=2048

# Filter pattern for experiment IDs (set to empty string to disable filtering)
FILTER_PATTERN="40:48"

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
    SELECTED_IDS=()
    for exp_id in "${STEERED_IDS[@]}"; do
      [[ "$exp_id" == *"$FILTER_PATTERN"* ]] && SELECTED_IDS+=("$exp_id")
    done

    if ((${#SELECTED_IDS[@]} == 0)); then
      echo "No experiments matched filter \"${FILTER_PATTERN}\" for ${CAP_FROM}."
      continue
    fi
  else
    SELECTED_IDS=("${STEERED_IDS[@]}")
  fi

  # Queue one task per experiment (one GPU each)
  for EXP in "${SELECTED_IDS[@]}"; do
    OUTPUT_PATH="${BASEDIR}/mmlu_pro/${CAP_FROM}/${EXP}"

    # Skip if output directory already exists
    if [[ -d "$OUTPUT_PATH" ]]; then
      echo ">>> Skip ${CAP_FROM}::${EXP} : output directory already exists at ${OUTPUT_PATH}"
      continue
    fi

    echo ">>> Queue ${CAP_FROM}::${EXP} : MMLU-Pro (batch=auto) -> ${OUTPUT_PATH}/"

    ts -G 1 uv run 2_benchmark_vllm.py \
      --config_filepath "$CFG" \
      --experiment_ids "$EXP" \
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
done

echo ""
echo "Results will be saved to:"
for CAP_FROM in "${!CONFIGS[@]}"; do
  echo "  ${CAP_FROM} -> ${BASEDIR}/mmlu_pro/${CAP_FROM}/<experiment_id>/"
done
