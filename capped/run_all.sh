#!/usr/bin/env bash
set -euo pipefail

uv run 2_benchmark_eval.py \
    --config_filepath /workspace/qwen-3-32b/capped/configs/role_trait_config.pt \
    --experiment_ids baseline \
    --model_name Qwen/Qwen3-32B \
    --tasks mmlu_pro,gsm8k,eq_bench,ifeval,mbpp \
    --output_dir /workspace/qwen-3-32b/capped/benchmarks/baseline --limit 8

uv run 2_benchmark_eval.py \
    --config_filepath /workspace/qwen-3-32b/capped/configs/role_trait_config.pt \
    --experiment_ids all \
    --model_name Qwen/Qwen3-32B \
    --tasks mmlu_pro,gsm8k,eq_bench,ifeval,mbpp \
    --output_dir /workspace/qwen-3-32b/capped/benchmarks/role_trait

uv run 2_benchmark_eval.py \
    --config_filepath /workspace/qwen-3-32b/capped/configs/lmsys_10000_config.pt \
    --experiment_ids all \
    --model_name Qwen/Qwen3-32B \
    --tasks mmlu_pro,gsm8k,eq_bench,ifeval,mbpp \
    --output_dir /workspace/qwen-3-32b/capped/benchmarks/lmsys_10000