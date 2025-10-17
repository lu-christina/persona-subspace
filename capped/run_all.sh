#!/usr/bin/env bash
set -euo pipefail

echo ">>> Queue role_trait jailbreak task"
ts -G 1 uv run 1_capping_multi.py \
    --config_filepath /workspace/qwen-3-32b/capped/configs/role_trait_eighths_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /workspace/qwen-3-32b/capped/results/role_trait_jailbreak_1100.jsonl \
    --batch_size 16 --company Alibaba --model Qwen/Qwen3-32B --thinking False

echo ">>> Queue lmsys_10000 jailbreak task"
ts -G 1 uv run 1_capping_multi.py \
    --config_filepath /workspace/qwen-3-32b/capped/configs/lmsys_10000_eighths_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /workspace/qwen-3-32b/capped/results/lmsys_10000_jailbreak_1100.jsonl \
    --batch_size 16 --company Alibaba --model Qwen/Qwen3-32B --thinking False

echo "Tasks queued in taskspooler"
