#!/usr/bin/env bash
set -euo pipefail

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
CFG="/workspace/qwen-3-32b/capped/configs/lmsys_10000_config.pt"

# Send baseline runs here:
BASEDIR_BASELINE="/workspace/qwen-3-32b/capped/benchmarks/baseline"
# Send all steered runs here:
BASEDIR_ROLE_TRAIT="/workspace/qwen-3-32b/capped/benchmarks/lmsys_10000"

# ===== Eval settings =====
TASKS="mmlu_pro"
LIMIT=100
SEED=42
FEWSHOT=0
DTYPE="bfloat16"

# MMLU-Pro is MC ranking
BATCH=36

# ===== Read steered experiment IDs (exclude 'baseline') =====
readarray -t STEERED_IDS < <(python - "$CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
print("\n".join([e.get("id","") for e in cfg.get("experiments", []) if e.get("id","").lower()!="baseline" and e.get("id","")]))
PY
)

# Filter STEERED_IDS to only include experiments containing "0:64"
SELECTED_IDS=()
for exp_id in "${STEERED_IDS[@]}"; do
  if [[ "$exp_id" == *"0:64"* ]]; then
    SELECTED_IDS+=("$exp_id")
  fi
done

# Build queue: selected steered experiments
QUEUE=( "${SELECTED_IDS[@]}" )
echo "Queue (${#QUEUE[@]} jobs): ${QUEUE[*]}"

# ===== Env & prep =====
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_COUNT=4
GPU_MAP=(0 1 4 6)  # Map logical GPU indices to physical devices
declare -A PIDS JOBS

pop() {
  if ((${#QUEUE[@]}==0)); then return 1; fi
  EXP="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}"); return 0
}

launch_job () {
  local GPU="$1" EXP="$2"
  local PHYSICAL_GPU="${GPU_MAP[$GPU]}"

  # choose outdir based on baseline vs steered
  local OUTDIR
  OUTDIR="$BASEDIR_ROLE_TRAIT"

  echo ">>> [GPU ${PHYSICAL_GPU}] Launch ${EXP} : MMLU-Pro (batch=${BATCH}) -> ${OUTDIR}"

  ARGS_COMMON=(
    --config_filepath "$CFG"
    --experiment_ids "$EXP"
    --model_name "$MODEL"
    --tasks "$TASKS"
    --output_dir "$OUTDIR"
    --batch_size "$BATCH"
    --device_strategy replicate
    --torch_dtype "$DTYPE"
    --limit "$LIMIT"
    --random_seed "$SEED"
    --num_fewshot "$FEWSHOT"
  )

  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" uv run 2_benchmark_eval.py \
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
    echo "✅ All MMLU-Pro runs finished."
    break
  fi
  sleep 3
done

echo "Results:"
echo "  steered  -> ${BASEDIR_ROLE_TRAIT}/<experiment_id>/mmlu_pro/<timestamp_seed_limit_shots>/"
