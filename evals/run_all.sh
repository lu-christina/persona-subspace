#!/usr/bin/env bash
set -euo pipefail

 uv run 1_capping_multi.py \
    --config_filepath /workspace/gemma-2-27b/evals/multi_contrast_layers_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /workspace/gemma-2-27b/evals/capped/multi_contrast_layers_1100.jsonl \
    --batch_size 2 --company Google --model google/gemma-2-27b-it

uv run scripts/chat_projections.py \
    --hf_dataset AllenAI/WildChat \
    --model_name google/gemma-2-27b-it \
    --target_vectors /workspace/gemma-2-27b/evals/multi_contrast_vectors.pt \
    --output_json /workspace/gemma-2-27b/evals/capped/projections/wildchat_10000.json \
    --samples 10000 --seed 42 --batch_size 1 --histogram_bins 100 --checkpoint_interval 100 \
    --resume_from_checkpoint /workspace/gemma-2-27b/evals/capped/projections/wildchat_10000_checkpoint.json

uv run scripts/chat_projections.py \
    --hf_dataset AllenAI/WildChat \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --target_vectors /workspace/llama-3.3-70b/evals/multi_contrast_vectors.pt \
    --output_json /workspace/llama-3.3-70b/evals/capped/projections/wildchat_10000.json \
    --samples 10000 --seed 42 --batch_size 1 --histogram_bins 100 --checkpoint_interval 100 \
    --resume_from_checkpoint /workspace/llama-3.3-70b/evals/capped/projections/wildchat_10000_checkpoint.json

uv run scripts/chat_projections.py \
    --hf_dataset lmsys/lmsys-chat-1m \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --target_vectors /workspace/llama-3.3-70b/evals/multi_contrast_vectors.pt \
    --output_json /workspace/llama-3.3-70b/evals/capped/projections/lmsys_10000.json \
    --samples 10000 --seed 42 --batch_size 1 --histogram_bins 100 --checkpoint_interval 100 \
    --resume_from_checkpoint /workspace/llama-3.3-70b/evals/capped/projections/lmsys_10000_checkpoint.json
