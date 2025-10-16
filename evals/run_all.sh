#!/usr/bin/env bash
set -euo pipefail

# CUDA_VISIBLE_DEVICES=0 uv run 1_unsteered.py \
#     --model_name google/gemma-2-27b-it \
#     --questions_file /root/git/persona-subspace/evals/introspective/prefill_10.jsonl \
#     --output_jsonl /workspace/gemma-2-27b/evals/introspective/chat/prefill_10.jsonl \
#     --samples_per_prompt 10 --no_chat_format

# CUDA_VISIBLE_DEVICES=1 uv run 1_unsteered.py \
#     --model_name google/gemma-2-27b --chat_model google/gemma-2-27b-it \
#     --questions_file /root/git/persona-subspace/evals/introspective/prefill_10.jsonl \
#     --output_jsonl /workspace/gemma-2-27b/evals/introspective/base/prefill_10.jsonl \
#     --samples_per_prompt 10 --no_chat_format

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run 1_steering.py \
    --model_name google/gemma-2-27b-it \
    --pca_filepath /workspace/gemma-2-27b/roles_traits/pca/layer22_roles_pos23_traits_pos40-100.pt \
    --questions_file /root/git/persona-subspace/evals/introspective/prefill_10.jsonl \
    --output_jsonl /workspace/gemma-2-27b/evals/introspective/chat/prefill_10.jsonl \
    --samples_per_prompt 10 \
    --magnitudes -3200.0 -1600.0 1600.0 3200.0 \
    --batch_size 16 --layer 22 --no_chat_format

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run 1_steering.py \
    --model_name google/gemma-2-27b --chat_model google/gemma-2-27b-it \
    --pca_filepath /workspace/gemma-2-27b/roles_traits/pca/layer22_roles_pos23_traits_pos40-100.pt \
    --questions_file /root/git/persona-subspace/evals/introspective/prefill_10.jsonl \
    --output_jsonl /workspace/gemma-2-27b/evals/introspective/base/prefill_10.jsonl \
    --samples_per_prompt 10 \
    --magnitudes -3200.0 -1600.0 1600.0 3200.0 \
    --batch_size 16 --layer 22 --no_chat_format
