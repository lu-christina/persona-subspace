#!/usr/bin/env bash
set -euo pipefail

uv run chat_projections.py \
    --hf_dataset AllenAI/WildChat \
    --model_name google/gemma-3-27b-it \
    --target_vectors /workspace/gemma-3-27b/capped/configs/multi_contrast_vectors.pt \
    --output_json /workspace/gemma-3-27b/capped/projections/wildchat_10000.json \
    --samples 10000 --seed 42 --batch_size 2 --histogram_bins 100 --checkpoint_interval 100

uv run chat_projections.py \
    --hf_dataset lmsys/lmsys-chat-1m \
    --model_name google/gemma-3-27b-it \
    --target_vectors /workspace/gemma-3-27b/capped/configs/multi_contrast_vectors.pt \
    --output_json /workspace/gemma-3-27b/capped/projections/lmsys_10000.json \
    --samples 10000 --seed 42 --batch_size 2 --histogram_bins 100 --checkpoint_interval 100
