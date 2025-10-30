#!/usr/bin/env bash
set -euo pipefail

echo ">>> Queue Gemma task"
ts -G 2 uv run scripts/default_activations.py \
    --input-dir /workspace/qwen-3-32b/roles_240/responses \
    --model-name google/gemma-2-27b-it \
    --pca-path /workspace/gemma-2-27b/traits_240/pca/layer22_pos-neg50.pt \
    --output-dir /workspace/gemma-2-27b/traits/qwen-3-32b \
    --target-layer 22 \
    --batch-size 16 \
    --max-length 2048

echo ">>> Queue Qwen task"
ts -G 2 uv run scripts/default_activations.py \
    --input-dir /workspace/gemma-2-27b/roles_240/responses \
    --model-name google/gemma-2-27b-it \
    --pca-path /workspace/gemma-2-27b/traits_240/pca/layer22_pos-neg50.pt \
    --output-dir /workspace/gemma-2-27b/traits/gemma-2-27b \
    --target-layer 22 \
    --batch-size 16 \
    --max-length 2048

echo ">>> Queue Llama task"
ts -G 2 uv run scripts/default_activations.py \
    --input-dir /workspace/llama-3.3-70b/roles_240/responses \
    --model-name google/gemma-2-27b-it \
    --pca-path /workspace/gemma-2-27b/traits_240/pca/layer22_pos-neg50.pt \
    --output-dir /workspace/gemma-2-27b/traits/llama-3.3-70b \
    --target-layer 22 \
    --batch-size 16 \
    --max-length 2048

echo ">>> Queue Qwen task"
ts -G 2 uv run scripts/default_activations.py \
    --input-dir /workspace/qwen-3-32b/roles_240/responses \
    --model-name Qwen/Qwen3-32B \
    --pca-path /workspace/qwen-3-32b/traits_240/pca/layer32_pos-neg50.pt \
    --output-dir /workspace/qwen-3-32b/traits/qwen-3-32b \
    --target-layer 32 \
    --batch-size 16 \
    --max-length 2048

echo ">>> Queue Qwen task"
ts -G 2 uv run scripts/default_activations.py \
    --input-dir /workspace/gemma-2-27b/roles_240/responses \
    --model-name Qwen/Qwen3-32B \
    --pca-path /workspace/qwen-3-32b/traits_240/pca/layer32_pos-neg50.pt \
    --output-dir /workspace/qwen-3-32b/traits/gemma-2-27b \
    --target-layer 32 \
    --batch-size 16 \
    --max-length 2048

echo ">>> Queue Llama task"
ts -G 2 uv run scripts/default_activations.py \
    --input-dir /workspace/llama-3.3-70b/roles_240/responses \
    --model-name Qwen/Qwen3-32B \
    --pca-path /workspace/qwen-3-32b/traits_240/pca/layer32_pos-neg50.pt \
    --output-dir /workspace/qwen-3-32b/traits/llama-3.3-70b \
    --target-layer 32 \
    --batch-size 16 \
    --max-length 2048

echo "Tasks queued in taskspooler"