#!/usr/bin/env bash
set -euo pipefail

EXP_IDS=(
    "layers_32:40-p0.01"
    "layers_32:40-p0.25"
    "layers_32:40-p0.5"
    "layers_32:40-p0.75"
    "layers_32:56-p0.01"
    "layers_32:56-p0.25"
    "layers_32:56-p0.5"
	"layers_32:56-p0.75"
    "layers_52:68-p0.01"
    "layers_52:68-p0.25"
    "layers_52:68-p0.5"
    "layers_52:68-p0.75"
    "layers_52:76-p0.01"
    "layers_52:76-p0.25"
    "layers_52:76-p0.5"
    "layers_52:76-p0.75"
	"layers_56:72-p0.01"
	"layers_56:72-p0.25"
	"layers_56:72-p0.5"
	"layers_56:72-p0.75"
    "layers_64:72-p0.01"
    "layers_64:72-p0.25"
    "layers_64:72-p0.5"
    "layers_64:72-p0.75"
    "layers_64:80-p0.01"
    "layers_64:80-p0.25"
    "layers_64:80-p0.5"
    "layers_64:80-p0.75"
)


ts -G 4 uv run 1_capping_vllm.py \
    --config_filepath /workspace/llama-3.3-70b/capped/configs/contrast/gemma_role_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /workspace/llama-3.3-70b/capped/results/gemma_role_jailbreak_1100.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta --max_model_len 4096 \
    --tensor_parallel_size 2 --experiment_ids ${EXP_IDS[@]}

# ts -G 2 uv run 1_capping_vllm.py \
#     --config_filepath /workspace/llama-3.3-70b/capped/configs/contrast/gemma_role_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
#     --output_jsonl /workspace/llama-3.3-70b/capped/results/gemma_role_default_1100.jsonl \
#     --model_name meta-llama/Llama-3.3-70B-Instruct --max_model_len 4096 --no_system_prompt

# ts -G 2 uv run 1_capping_vllm.py \
#     --config_filepath /workspace/llama-3.3-70b/capped/configs/contrast/qwen_role_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
#     --output_jsonl /workspace/llama-3.3-70b/capped/results/qwen_role_jailbreak_1100.jsonl \
#     --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta --max_model_len 4096

# ts -G 2 uv run 1_capping_vllm.py \
#     --config_filepath /workspace/llama-3.3-70b/capped/configs/contrast/qwen_role_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
#     --output_jsonl /workspace/llama-3.3-70b/capped/results/qwen_role_default_1100.jsonl \
#     --model_name meta-llama/Llama-3.3-70B-Instruct --max_model_len 4096 --no_system_prompt
