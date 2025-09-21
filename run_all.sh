#!/usr/bin/env bash
set -Eeuo pipefail

# If interrupted or if any worker fails, kill the rest.
trap 'echo "[!] Aborting…"; jobs -pr | xargs -r kill -9; exit 1' INT TERM ERR

# Create a place for logs
mkdir -p logs

MODEL="google/gemma-3-27b-it"
QUESTIONS="/root/git/persona-subspace/traits/data/questions_240.jsonl"
ROLES_RESP_DIR="/workspace/gemma-3-27b/roles_240/responses"

# 4 workers: (device pair, roles subset)
declare -a DEVICES=("0,1" "2,3" "4,5" "6,7")
declare -a RANGES=("0-70" "70-140" "140-210" "210-275")

pids=()

echo "[*] Launching 4 role-response workers…"
for i in "${!DEVICES[@]}"; do
  d="${DEVICES[$i]}"
  r="${RANGES[$i]}"
  log="logs/roles_worker_${i}.log"

  echo "    - Worker $i -> GPUs {$d}, roles {$r} (log: $log)"

  CUDA_VISIBLE_DEVICES="$d" \
  uv run roles/2_responses.py \
      --model-name "$MODEL" \
      --questions-file "$QUESTIONS" \
      --output-dir "$ROLES_RESP_DIR" \
      --tensor-parallel-size 2 \
      --roles-subset "$r" \
      --no-default \
      >"$log" 2>&1 &
  pids+=($!)
done

# Wait for all four to complete
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "[*] All 4 workers finished successfully."

# === Follow-up activation passes (sequential) ===
# NOTE: These use your exact commands/paths.
#       The first uses TRAITS responses (assumes they already exist);
#       the second uses the ROLES responses just generated above.

echo "[*] Running traits response activations (8 GPUs)…"
uv run traits/3_response_activations.py \
  --model-name "$MODEL" \
  --responses-dir /workspace/gemma-3-27b/traits_240/responses \
  --output-dir   /workspace/gemma-3-27b/traits_240/response_activations \
  --multi-gpu --num-gpus 8 --batch-size 32

echo "[*] Running roles response activations (8 GPUs)…"
uv run roles/3_response_activations.py \
  --model-name "$MODEL" \
  --responses-dir /workspace/gemma-3-27b/roles_240/responses \
  --output-dir   /workspace/gemma-3-27b/roles_240/response_activations \
  --multi-gpu --num-gpus 8 --batch-size 32

echo "[*] Running judge"
uv run roles/4_judge.py \
	--output-dir /workspace/gemma-3-27b/roles_240/extract_scores \
	--responses-dir /workspace/gemma-3-27b/roles_240/responses


echo "[✓] Pipeline complete."
