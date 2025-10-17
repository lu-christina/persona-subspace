#!/usr/bin/env bash
set -euo pipefail

uv run 1_capping_multi.py \
    --config_filepath /workspace/qwen-3-32b/capped/configs/role_trait_eighths_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /workspace/qwen-3-32b/capped/results/role_trait_jailbreak_1100.jsonl \
    --batch_size 16 --company Alibaba --model Qwen/Qwen3-32B --thinking False

uv run 1_capping_multi.py \
    --config_filepath /workspace/qwen-3-32b/capped/configs/lmsys_10000_eighths_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /workspace/qwen-3-32b/capped/results/lmsys_10000_jailbreak_1100.jsonl \
    --batch_size 16 --company Alibaba --model Qwen/Qwen3-32B --thinking False
