#!/usr/bin/env bash
set -euo pipefail

WATCH="/root/git/persona-subspace/dynamics/results/llama-3.3-70b/prefills/jailbreak_prefills.jsonl"
NEED=1210000

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
  sleep 60
done

uv run 3_jailbreak_prefills.py \
  --prefill_file /root/git/persona-subspace/dynamics/results/gemma-2-27b/prefills/all_roles.jsonl \
  --questions_file /root/git/persona-subspace/evals/jailbreak/jailbreak_440.jsonl \
  --output_jsonl /root/git/persona-subspace/dynamics/results/gemma-2-27b/prefills/jailbreak_prefills.jsonl \
  --model_name google/gemma-2-27b-it

uv run 0_auto_conversation_parallel.py \
  --target-model meta-llama/Llama-3.3-70B-Instruct \
  --auditor-model anthropic/claude-sonnet-4.5 \
  --output-dir /root/git/persona-subspace/dynamics/results/llama-3.3-70b/sonnet-4.5/transcripts \
  --max-model-len 30000

uv run 1_activations.py \
  --model-name meta-llama/Llama-3.3-70B-Instruct \
  --target-dir /root/git/persona-subspace/dynamics/results/llama-3.3-70b/sonnet-4.5/transcripts \
  --output-dir /workspace/llama-3.3-70b/dynamics/sonnet-4.5/activations \
  --max-length 30000 \

  --batch-size 1
