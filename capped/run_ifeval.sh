#!/usr/bin/env bash
set -euo pipefail

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
CFG="/workspace/qwen-3-32b/capped/configs/multi_contrast_layers_config.pt"
ROLE_TRAIT_DIR="/workspace/qwen-3-32b/capped/benchmarks/jailbreak"
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
readarray -t QUEUE < <(python - "$CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
print("\n".join([e.get("id","") for e in cfg.get("experiments", []) if e.get("id","").lower()!="baseline" and e.get("id","")]))
PY
)

echo "Queue (${#QUEUE[@]} jobs): ${QUEUE[*]}"

# ===== Env & prep =====
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$CACHE_DIR"

GPU_COUNT=8
declare -A PIDS JOBS

pop() {
  if ((${#QUEUE[@]}==0)); then return 1; fi
  EXP="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}"); return 0
}

launch_job () {
  local GPU="$1" EXP="$2"

  echo ">>> [GPU ${GPU}] Launch ${EXP} : IFEval (batch=${BATCH}${MAXTOK:+, maxtok=${MAXTOK}}) -> ${ROLE_TRAIT_DIR}"
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

  CUDA_VISIBLE_DEVICES="${GPU}" uv run 2_benchmark_eval.py \
    "${ARGS_COMMON[@]}" &

  PIDS[$GPU]=$!
  JOBS[$GPU]="$EXP"
}

trap 'echo "Stopping..."; for p in "${PIDS[@]}"; do [[ -n "${p}" ]] && kill -9 "$p" 2>/dev/null || true; done' INT TERM

# ===== Fill GPUs initially =====
for ((g=0; g<GPU_COUNT; g++)); do
  if pop; then launch_job "$g" "$EXP"; fi
done

# ===== Scheduler loop =====
while :; do
  for ((g=0; g<GPU_COUNT; g++)); do
    pid="${PIDS[$g]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "<<< [GPU ${g}] Done: ${JOBS[$g]}"
      unset PIDS[$g] JOBS[$g]
      if pop; then launch_job "$g" "$EXP"; fi
    fi
  done

  # Exit when queue empty and no running jobs
  if ((${#QUEUE[@]}==0 && ${#PIDS[@]}==0)); then
    echo "✅ All IFEval runs finished."
    break
  fi
  sleep 3
done

echo "✅ Results saved to: ${ROLE_TRAIT_DIR}/{experiment_id}/"
