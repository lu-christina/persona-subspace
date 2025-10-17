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

echo "========================================"
echo "EQ-Bench Scheduler using taskspooler"
echo "========================================"
echo "Configs:"
echo "  role_trait: ${#ROLE_IDS[@]} experiments"
echo "  jailbreak:  ${#JAIL_IDS[@]} experiments"
echo "  lmsys:      ${#LMSYS_IDS[@]} experiments"
echo "Total jobs: $((${#ROLE_IDS[@]} + ${#JAIL_IDS[@]} + ${#LMSYS_IDS[@]}))"
echo "========================================"

# ===== Env & prep =====
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Queue role_trait experiments
for id in "${ROLE_IDS[@]}"; do
  echo ">>> Queue role::${id} : ${TASKS} (batch=${BATCH}) -> ${BASEDIR_ROLE_TRAIT}"
  ts -G 1 uv run 2_benchmark_eval.py \
    --config_filepath "$ROLE_CFG" \
    --experiment_ids "$id" \
    --model_name "$MODEL" \
    --tasks "$TASKS" \
    --output_dir "$BASEDIR_ROLE_TRAIT" \
    --batch_size "$BATCH"
done

# Queue jailbreak experiments
for id in "${JAIL_IDS[@]}"; do
  echo ">>> Queue jail::${id} : ${TASKS} (batch=${BATCH}) -> ${BASEDIR_JAILBREAK}"
  ts -G 1 uv run 2_benchmark_eval.py \
    --config_filepath "$JAIL_CFG" \
    --experiment_ids "$id" \
    --model_name "$MODEL" \
    --tasks "$TASKS" \
    --output_dir "$BASEDIR_JAILBREAK" \
    --batch_size "$BATCH"
done

# Queue lmsys experiments
for id in "${LMSYS_IDS[@]}"; do
  echo ">>> Queue lmsys::${id} : ${TASKS} (batch=${BATCH}) -> ${BASEDIR_LMSYS}"
  ts -G 1 uv run 2_benchmark_eval.py \
    --config_filepath "$LMSYS_CFG" \
    --experiment_ids "$id" \
    --model_name "$MODEL" \
    --tasks "$TASKS" \
    --output_dir "$BASEDIR_LMSYS" \
    --batch_size "$BATCH"
done

echo ""
echo "Results will be saved to:"
echo "  role_trait -> ${BASEDIR_ROLE_TRAIT}/<experiment_id>/eq_bench/"
echo "  jailbreak  -> ${BASEDIR_JAILBREAK}/<experiment_id>/eq_bench/"
echo "  lmsys      -> ${BASEDIR_LMSYS}/<experiment_id>/eq_bench/"