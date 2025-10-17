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

# Filter STEERED_IDS to only include experiments containing "16:24"
SELECTED_IDS=()
for exp_id in "${STEERED_IDS[@]}"; do
  [[ "$exp_id" == *"16:24"* ]] && SELECTED_IDS+=("$exp_id")
done

if ((${#SELECTED_IDS[@]} == 0)); then
  echo "No experiments matched filter \"16:24\"."
  exit 1
fi

# ===== Env & prep =====
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Queue one task per experiment
for EXP in "${SELECTED_IDS[@]}"; do
  OUTDIR="${BASEDIR_ROLE_TRAIT}/${EXP}"
  mkdir -p "$OUTDIR"

  echo ">>> Queue ${EXP} : MMLU-Pro (batch=${BATCH}) -> ${OUTDIR}"

  ts -G 1 uv run 2_benchmark_eval.py \
    --config_filepath "$CFG" \
    --experiment_ids "$EXP" \
    --model_name "$MODEL" \
    --tasks "$TASKS" \
    --output_dir "$OUTDIR" \
    --batch_size "$BATCH" \
    --device_strategy replicate \
    --torch_dtype "$DTYPE" \
    --limit "$LIMIT" \
    --random_seed "$SEED" \
    --num_fewshot "$FEWSHOT"
done

echo "Results:"
echo "  steered  -> ${BASEDIR_ROLE_TRAIT}/<experiment_id>/mmlu_pro/<timestamp_seed_limit_shots>/"
