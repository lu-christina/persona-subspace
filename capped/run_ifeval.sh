#!/usr/bin/env bash
set -euo pipefail

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
CFG="/workspace/qwen-3-32b/capped/configs/lmsys_10000_config.pt"
ROLE_TRAIT_DIR="/workspace/qwen-3-32b/capped/benchmarks/lmsys_10000"
CACHE_DIR="/workspace/qwen-3-32b/capped/.lm_eval_cache"

# ===== Eval settings =====
TASKS="ifeval"        # IFEval only (as requested)
LIMIT=500
SEED=42
FEWSHOT=0
DTYPE="bfloat16"

# Start conservative on a 32B model to avoid stalls/OOM
BATCH=48              # good default for 32B on one H200
MAXTOK=256             # set to e.g. 192 or 256 if your Python script accepts --max_gen_toks

# ===== Read experiment IDs =====
readarray -t STEERED_IDS < <(python - "$CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
print("\n".join([e.get("id","") for e in cfg.get("experiments", []) if e.get("id","").lower()!="baseline" and e.get("id","")]))
PY
)

# ===== Env & prep =====
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$CACHE_DIR"

# Queue one task per experiment
for EXP in "${STEERED_IDS[@]}"; do
  echo ">>> Queue ${EXP} : IFEval (batch=${BATCH}${MAXTOK:+, maxtok=${MAXTOK}}) -> ${ROLE_TRAIT_DIR}"

  # Build args (conditionally include max_gen_toks if set)
  ARGS_COMMON=(
    --config_filepath "$CFG"
    --experiment_ids "$EXP"
    --model_name "$MODEL"
    --tasks "$TASKS"
    --output_dir "$ROLE_TRAIT_DIR"
    --batch_size "$BATCH"
    --device_strategy replicate
    --torch_dtype "$DTYPE"
    --limit "$LIMIT"
    --random_seed "$SEED"
    --num_fewshot "$FEWSHOT"
  )
  if [[ -n "${MAXTOK}" ]]; then
    ARGS_COMMON+=( --max_gen_toks "$MAXTOK" )
  fi

  ts -G 1 uv run 2_benchmark_eval.py "${ARGS_COMMON[@]}"
done

echo "Results saved to: ${ROLE_TRAIT_DIR}/<experiment_id>/ifeval/"
