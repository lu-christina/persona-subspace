#!/usr/bin/env bash
set -euo pipefail

uv run compute_variance.py \
    --model gemma-2-27b \
    --layer 22 \

uv run compute_variance.py \
    --model qwen-3-32b \
    --layer 32 \

uv run compute_variance.py \
    --model llama-3.3-70b \
    --layer 40 \