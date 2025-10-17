#!/usr/bin/env bash
set -euo pipefail

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
ROLE_CFG="/workspace/qwen-3-32b/capped/configs/role_trait_config.pt"
JAIL_CFG="/workspace/qwen-3-32b/capped/configs/multi_contrast_layers_config.pt"

# Results (no baseline path)
BASEDIR_ROLE_TRAIT="/workspace/qwen-3-32b/capped/benchmarks/role_trait"
BASEDIR_JAILBREAK="/workspace/qwen-3-32b/capped/benchmarks/jailbreak"

# ===== Eval settings =====
TASKS="mmlu_pro"
LIMIT=100
SEED=42
FEWSHOT=0
DTYPE="bfloat16"
BATCH=36   # MMLU-Pro is MC ranking

# ===== Read experiment IDs =====
# role_trait: drop 'baseline' and then skip first 4
readarray -t ROLE_IDS_RAW < <(python - "$ROLE_CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
ids = [e.get("id","") for e in cfg.get("experiments", [])]
ids = [i for i in ids if i and i.lower()!="baseline"]
print("\n".join(ids))
PY
)
if ((${#ROLE_IDS_RAW[@]} > 4)); then
  ROLE_IDS=("${ROLE_IDS_RAW[@]:4}")
else
  ROLE_IDS=()
fi

# jailbreak: non-empty, non-baseline
readarray -t JAIL_IDS < <(python - "$JAIL_CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
ids = [e.get("id","") for e in cfg.get("experiments", [])]
ids = [i for i in ids if i and i.lower()!="baseline"]
print("\n".join(ids))
PY
)

# ===== Build queue: role steered (after skip) -> jailbreak steered =====
# Each entry is "GROUP::EXPID" where GROUP ∈ {role,jail}
QUEUE=()
for id in "${ROLE_IDS[@]}"; do QUEUE+=( "role::${id}" ); done
for id in "${JAIL_IDS[@]}"; do QUEUE+=( "jail::${id}" ); done
echo "Queue (${#QUEUE[@]} jobs): ${QUEUE[*]}"

# ===== Env & prep =====
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# avoid inherited masking that would make only 1 GPU visible
unset CUDA_VISIBLE_DEVICES

GPU_COUNT=8
declare -a PIDS JOBS

pop() {
  if ((${#QUEUE[@]}==0)); then return 1; fi
  ITEM="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
  GROUP="${ITEM%%::*}"
  EXP="${ITEM#*::}"
  return 0
}

launch_job () {
  local GPU="$1" GROUP="$2" EXP="$3"

  local CFG OUTDIR
  if [[ "$GROUP" == "role" ]]; then
    CFG="$ROLE_CFG"; OUTDIR="$BASEDIR_ROLE_TRAIT"
  else
    CFG="$JAIL_CFG"; OUTDIR="$BASEDIR_JAILBREAK"
  fi

  echo ">>> [GPU ${GPU}] Launch ${GROUP}::${EXP} : ${TASKS} (batch=${BATCH}) -> ${OUTDIR}"

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

  # stdout only; uncomment sed line for per-line prefixes
  # ( CUDA_VISIBLE_DEVICES="${GPU}" uv run 2_benchmark_eval.py "${ARGS_COMMON[@]}" \
  #   | sed "s/^/[GPU ${GPU}] ${GROUP}::${EXP}: /" ) &
  CUDA_VISIBLE_DEVICES="${GPU}" uv run 2_benchmark_eval.py "${ARGS_COMMON[@]}" &

  PIDS[$GPU]=$!
  JOBS[$GPU]="${GROUP}::${EXP}"
}

trap 'echo "Stopping..."; for p in "${PIDS[@]}"; do [[ -n "${p}" ]] && kill -9 "$p" 2>/dev/null || true; done' INT TERM

# ===== Fill GPUs initially =====
for ((g=0; g<GPU_COUNT; g++)); do
  if pop; then launch_job "$g" "$GROUP" "$EXP"; fi
done

# ===== Scheduler loop =====
while :; do
  for ((g=0; g<GPU_COUNT; g++)); do
    pid="${PIDS[$g]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "<<< [GPU ${g}] Done: ${JOBS[$g]}"
      unset PIDS[$g] JOBS[$g]
      if pop; then launch_job "$g" "$GROUP" "$EXP"; fi
    fi
  done

  if ((${#QUEUE[@]}==0 && ${#PIDS[@]}==0)); then
    echo "✅ All ${TASKS} runs finished."
    break
  fi
  sleep 3
done

echo "Results:"
echo "  role_trait -> ${BASEDIR_ROLE_TRAIT}/<experiment_id>/mmlu_pro/<timestamp_seed_limit_shots>/"
echo "  jailbreak  -> ${BASEDIR_JAILBREAK}/<experiment_id>/mmlu_pro/<timestamp_seed_limit_shots>/"
