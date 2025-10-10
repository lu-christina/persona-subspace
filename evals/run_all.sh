#!/usr/bin/env bash
set -euo pipefail

WATCH="/root/git/persona-subspace/evals/jailbreak/qwen-3-32b/capped/jailbreak_1100.jsonl"
NEED=42900

echo "Waiting for $WATCH to reach $NEED lines…"
while :; do
  if [[ -f "$WATCH" ]]; then
    lines=$(wc -l < "$WATCH" || echo 0)
    printf "\rCurrent lines: %d" "$lines"
    if [[ "$lines" -ge "$NEED" ]]; then
      echo -e "\nReady (found $lines lines)."
      break
    fi
  else
    printf "\rFile not found yet…"
  fi
  sleep 60  # checks once per minute
done

# --- Run first job (Llama-3.3-70B) ---
uv run 1_capping.py \
  --vectors_filepath /workspace/llama-3.3-70b/evals/capping_vectors.pt \
  --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
  --output_jsonl /root/git/persona-subspace/evals/jailbreak/llama-3.3-70b/capped/jailbreak_1100.jsonl \
  --batch_size 16 \
  --company Meta \
  --model meta-llama/Llama-3.3-70B-Instruct

# --- Run second job (Gemma-2-27B) ---
uv run 1_capping.py \
  --vectors_filepath /workspace/gemma-2-27b/evals/capping_vectors.pt \
  --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
  --output_jsonl /root/git/persona-subspace/evals/jailbreak/gemma-2-27b/capped/jailbreak_1100.jsonl \
  --batch_size 32 \
  --company Google \
  --model google/gemma-2-27b-it

uv run compute_projections.py \
    --input_jsonl /workspace/gemma-2-27b/evals/unsteered/unsteered_default_scores.jsonl \
    --target_vectors /workspace/gemma-2-27b/evals/capping_vectors.pt \
    --model_name google/gemma-2-27b-it \
    --output_jsonl /workspace/gemma-2-27b/evals/unsteered/unsteered_default_projections.jsonl \
    --batch_size 4

uv run compute_projections.py \
    --input_jsonl /workspace/gemma-2-27b/evals/unsteered/unsteered_scores.jsonl \
    --target_vectors /workspace/gemma-2-27b/evals/capping_vectors.pt \
    --model_name google/gemma-2-27b-it \
    --output_jsonl /workspace/gemma-2-27b/evals/unsteered/unsteered_projections.jsonl \
    --batch_size 4
