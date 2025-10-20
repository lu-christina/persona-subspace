#!/usr/bin/env bash
set -euo pipefail

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
BASEDIR="/workspace/qwen-3-32b/capped/benchmarks"

# Associative array: cap_from -> config_path
declare -A CONFIGS=(
  ["jailbreak"]="/workspace/qwen-3-32b/capped/configs/jailbreak_config.pt"
)

# ===== Eval settings =====
TASKS="gsm8k"
LIMIT=1000
SEED=42
DTYPE="bfloat16"
BATCH=16
MAXTOK=512

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
print("\n".join([e.get("id","") for e in cfg.get("experiments", []) if e.get("id","").lower()!="baseline" and e.get("id","")]))
PY
  )

  # Queue one task per experiment
  for EXP in "${STEERED_IDS[@]}"; do
    echo ">>> Queue ${CAP_FROM}::${EXP} : IFEval (batch=${BATCH}${MAXTOK:+, maxtok=${MAXTOK}}) -> ${BASEDIR}/ifeval/${CAP_FROM}/<experiment_id>/"

    # Build args (conditionally include max_gen_toks if set)
    ARGS_COMMON=(
      --config_filepath "$CFG"
      --experiment_ids "$EXP"
      --model_name "$MODEL"
      --tasks "$TASKS"
      --output_dir "$BASEDIR"
      --cap_from "$CAP_FROM"
      --batch_size "$BATCH"
      --device_strategy replicate
      --torch_dtype "$DTYPE"
      --limit "$LIMIT"
      --random_seed "$SEED"
    )
    if [[ -n "${MAXTOK}" ]]; then
      ARGS_COMMON+=( --max_gen_toks "$MAXTOK" )
    fi

    ts -G 1 uv run 2_benchmark_eval.py "${ARGS_COMMON[@]}"
  done
done

echo ""
echo "Results will be saved to:"
for CAP_FROM in "${!CONFIGS[@]}"; do
  echo "  ${CAP_FROM} -> ${BASEDIR}/ifeval/${CAP_FROM}/<experiment_id>/"
done
