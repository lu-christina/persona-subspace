#!/usr/bin/env bash
set -euo pipefail

ts -G 2 uv run scripts/token_norms.py \
    --hf_dataset lmsys/lmsys-chat-1m \
    --model_name Qwen/Qwen3-32B \
    --output_dir /workspace/qwen-3-32b/token_norms/ \
    --samples 10000 \
    --batch_size 1 --resume_from_checkpoint

ts -G 2 uv run scripts/token_norms.py \
    --hf_dataset lmsys/lmsys-chat-1m \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --output_dir /workspace/llama-3.3-70b/token_norms/ \
    --samples 10000 \
    --batch_size 1 --resume_from_checkpoint

ts -G 2 uv run scripts/token_norms.py \
    --hf_dataset lmsys/lmsys-chat-1m \
    --model_name google/gemma-2-27b-it \
    --output_dir /workspace/gemma-2-27b/token_norms/ \
    --samples 10000 \
    --batch_size 1 --resume_from_checkpoint