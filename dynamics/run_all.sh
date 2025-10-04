#!/usr/bin/env bash
set -euo pipefail

# Extract activations for Qwen
uv run 1_activations.py \
  --model-name Qwen/Qwen3-32B \
  --target-dir /root/git/persona-subspace/dynamics/results/qwen-3-32b/sonnet-4.5/transcripts \
  --output-dir /workspace/qwen-3-32b/dynamics/sonnet-4.5/activations \
  --max-length 30000 \
  --batch-size 1

# Extract activations for Gemma
uv run 1_activations.py \
  --model-name google/gemma-2-27b-it \
  --target-dir /root/git/persona-subspace/dynamics/results/gemma-2-27b/sonnet-4.5/transcripts \
  --output-dir /workspace/gemma-2-27b/dynamics/sonnet-4.5/activations \
  --max-length 8192 \
  --batch-size 1
