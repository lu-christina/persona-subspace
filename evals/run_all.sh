#!/usr/bin/env bash
set -euo pipefail

uv run compute_projections.py \
    --input_jsonl /workspace/qwen-3-32b/evals/unsteered/unsteered_default_scores.jsonl \
    --target_vectors /workspace/qwen-3-32b/evals/multi_contrast_vectors.pt \
    --model_name Qwen/Qwen3-32B \
    --output_jsonl /workspace/qwen-3-32b/evals/unsteered/projections/unsteered_default_multi_contrast_projections.jsonl \
    --batch_size 4

uv run compute_projections.py \
    --input_jsonl /workspace/qwen-3-32b/evals/unsteered/unsteered_scores.jsonl \
    --target_vectors /workspace/qwen-3-32b/evals/multi_contrast_vectors.pt \
    --model_name Qwen/Qwen3-32B \
    --output_jsonl /workspace/qwen-3-32b/evals/unsteered/projections/unsteered_multi_contrast_projections.jsonl \
    --batch_size 4