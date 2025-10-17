#!/usr/bin/env bash
set -euo pipefail

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
BASEDIR="/workspace/qwen-3-32b/capped/benchmarks"

# Associative array: cap_from -> config_path
declare -A CONFIGS=(
  ["role_trait"]="/workspace/qwen-3-32b/capped/configs/role_trait_eighths_config.pt"
  ["lmsys_10000"]="/workspace/qwen-3-32b/capped/configs/lmsys_10000_eighths_config.pt"
)

# ===== Eval settings =====
TASKS="eq_bench"
BATCH=64

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

  # Read experiment IDs for this config
  readarray -t STEERED_IDS < <(python - "$CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
ids = [e.get("id","") for e in cfg.get("experiments", [])]
ids = [i for i in ids if i and i.lower()!="baseline"]
print("\n".join(ids))
PY
  )

  # Queue one task per experiment
  for EXP in "${STEERED_IDS[@]}"; do
    echo ">>> Queue ${CAP_FROM}::${EXP} : ${TASKS} (batch=${BATCH}) -> ${BASEDIR}/eq_bench/${CAP_FROM}/<experiment_id>/"
    ts -G 1 uv run 2_benchmark_eval.py \
      --config_filepath "$CFG" \
      --experiment_ids "$EXP" \
      --model_name "$MODEL" \
      --tasks "$TASKS" \
      --output_dir "$BASEDIR" \
      --cap_from "$CAP_FROM" \
      --batch_size "$BATCH"
  done
done

echo ""
echo "Results will be saved to:"
for CAP_FROM in "${!CONFIGS[@]}"; do
  echo "  ${CAP_FROM} -> ${BASEDIR}/eq_bench/${CAP_FROM}/<experiment_id>/"
done