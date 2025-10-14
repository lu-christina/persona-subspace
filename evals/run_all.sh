#!/usr/bin/env bash
set -euo pipefail

uv run 1_capping_multi.py \
    --config_filepath /workspace/llama-3.3-70b/evals/multi_contrast_layers_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /root/git/persona-subspace/evals/jailbreak/llama-3.3-70b/capped/multi_contrast_layers_1100.jsonl \
    --batch_size 16 --company Meta --model meta-llama/Llama-3.3-70B-Instruct

uv run 1_capping_multi.py \
    --config_filepath /workspace/gemma-2-27b/evals/multi_contrast_layers_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /root/git/persona-subspace/evals/jailbreak/gemma-2-27b/capped/multi_contrast_layers_1100.jsonl \
    --batch_size 16 --company Google --model google/gemma-2-27b-it

