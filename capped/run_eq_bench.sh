#!/usr/bin/env bash
set -euo pipefail

# ===== Paths & model =====
MODEL="Qwen/Qwen3-32B"
ROLE_CFG="/workspace/qwen-3-32b/capped/configs/role_trait_config.pt"
JAIL_CFG="/workspace/qwen-3-32b/capped/configs/multi_contrast_layers_config.pt"
LMSYS_CFG="/workspace/qwen-3-32b/capped/configs/lmsys_10000_config.pt"

# Results directories
BASEDIR_ROLE_TRAIT="/workspace/qwen-3-32b/capped/benchmarks/role_trait"
BASEDIR_JAILBREAK="/workspace/qwen-3-32b/capped/benchmarks/jailbreak"
BASEDIR_LMSYS="/workspace/qwen-3-32b/capped/benchmarks/lmsys_10000"

# ===== Eval settings =====
TASKS="eq_bench"
BATCH=36

# ===== Read experiment IDs (excluding baseline) =====
# role_trait: non-empty, non-baseline (no skipping for eq_bench)
readarray -t ROLE_IDS < <(python - "$ROLE_CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
ids = [e.get("id","") for e in cfg.get("experiments", [])]
ids = [i for i in ids if i and i.lower()!="baseline"]
print("\n".join(ids))
PY
)

# jailbreak: non-empty, non-baseline
readarray -t JAIL_IDS < <(python - "$JAIL_CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
ids = [e.get("id","") for e in cfg.get("experiments", [])]
ids = [i for i in ids if i and i.lower()!="baseline"]
print("\n".join(ids))
PY
)

# lmsys_10000: non-empty, non-baseline
readarray -t LMSYS_IDS < <(python - "$LMSYS_CFG" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], weights_only=False)
ids = [e.get("id","") for e in cfg.get("experiments", [])]
ids = [i for i in ids if i and i.lower()!="baseline"]
print("\n".join(ids))
PY
)

# ===== Build queue: role -> jailbreak -> lmsys =====
# Each entry is "GROUP::EXPID" where GROUP ∈ {role,jail,lmsys}
QUEUE=()
for id in "${ROLE_IDS[@]}"; do QUEUE+=( "role::${id}" ); done
for id in "${JAIL_IDS[@]}"; do QUEUE+=( "jail::${id}" ); done
for id in "${LMSYS_IDS[@]}"; do QUEUE+=( "lmsys::${id}" ); done

echo "========================================"
echo "EQ-Bench Scheduler for 2x H100"
echo "========================================"
echo "Configs:"
echo "  role_trait: ${#ROLE_IDS[@]} experiments"
echo "  jailbreak:  ${#JAIL_IDS[@]} experiments"
echo "  lmsys:      ${#LMSYS_IDS[@]} experiments"
echo "Total queue: ${#QUEUE[@]} jobs"
echo "========================================"

# ===== Env & prep =====
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# avoid inherited masking that would make only 1 GPU visible
unset CUDA_VISIBLE_DEVICES

GPU_COUNT=2  # 2x H100 for eq_bench
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
  elif [[ "$GROUP" == "jail" ]]; then
    CFG="$JAIL_CFG"; OUTDIR="$BASEDIR_JAILBREAK"
  else  # lmsys
    CFG="$LMSYS_CFG"; OUTDIR="$BASEDIR_LMSYS"
  fi

  echo ">>> [GPU ${GPU}] Launch ${GROUP}::${EXP} : ${TASKS} (batch=${BATCH}) -> ${OUTDIR}"

  ARGS_COMMON=(
    --config_filepath "$CFG"
    --experiment_ids "$EXP"
    --model_name "$MODEL"
    --tasks "$TASKS"
    --output_dir "$OUTDIR"
    --batch_size "$BATCH"
  )

  # Run with GPU assignment
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
start_time=$(date +%s)

while :; do
  for ((g=0; g<GPU_COUNT; g++)); do
    pid="${PIDS[$g]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "<<< [GPU ${g}] Done: ${JOBS[$g]}"
      unset PIDS[$g] JOBS[$g]
      if pop; then launch_job "$g" "$GROUP" "$EXP"; fi
    fi
  done

  if ((${#QUEUE[@]}==0)); then
    # Check if all jobs are done
    all_done=true
    for ((g=0; g<GPU_COUNT; g++)); do
      if [[ -n "${PIDS[$g]:-}" ]]; then
        all_done=false
        break
      fi
    done
    
    if $all_done; then
      echo "========================================"
      echo "✅ All ${TASKS} runs finished."
      end_time=$(date +%s)
      duration=$((end_time - start_time))
      hours=$((duration / 3600))
      minutes=$(((duration % 3600) / 60))
      seconds=$((duration % 60))
      echo "Total time: ${hours}h ${minutes}m ${seconds}s"
      echo "========================================"
      break
    fi
  fi
  
  # Status update every 30 seconds
  if (( $(date +%s) % 30 == 0 )); then
    remaining=${#QUEUE[@]}
    if ((remaining > 0)); then
      echo "[Status] Remaining in queue: ${remaining} | Active: GPU0=${JOBS[0]:-idle} GPU1=${JOBS[1]:-idle}"
    fi
  fi
  
  sleep 3
done

echo ""
echo "Results locations:"
echo "  role_trait -> ${BASEDIR_ROLE_TRAIT}/<experiment_id>/eq_bench/"
echo "  jailbreak  -> ${BASEDIR_JAILBREAK}/<experiment_id>/eq_bench/"
echo "  lmsys      -> ${BASEDIR_LMSYS}/<experiment_id>/eq_bench/"