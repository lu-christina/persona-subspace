#!/usr/bin/env bash
# HarmBench × in-the-wild personas — additive steering + capping for gemma/qwen/llama.
#
# Prereq (one-time): regenerate roles_itw.jsonl if needed
#   python3 evals/harmbench/sample_itw_roles.py --n 5 --seed 42
#
# Additive configs already encode fraction-of-avg-norm coefficients (built by
# evals/0_vector_config.ipynb). Capping experiment_ids are the chosen best:
#   qwen-3-32b   → layers_46:54-p0.25
#   llama-3.3-70b → layers_56:72-p0.25

QUESTIONS=evals/harmbench/harmbench.jsonl
ROLES=evals/harmbench/roles_itw.jsonl

# ──────────────────────────────────────────────────────────────────────────────
# Qwen-3-32B (vllm)
# ──────────────────────────────────────────────────────────────────────────────

ts -G 1 uv run evals/1_steering_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/qwen-3-32b/evals/configs/asst_pc1_contrast_config.pt \
    --output_jsonl /workspace/qwen-3-32b/evals/harmbench_itw/asst_pc1_contrast.jsonl \
    --model_name Qwen/Qwen3-32B --company Alibaba --name Qwen \
    --thinking false --max_model_len 4096 --dtype bfloat16

ts -G 1 uv run capped/1_capping_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/qwen-3-32b/capped/configs/contrast/role_trait_config.pt \
    --experiment_ids layers_46:54-p0.25 \
    --output_jsonl /workspace/qwen-3-32b/evals/harmbench_itw/capped_role_trait.jsonl \
    --model_name Qwen/Qwen3-32B --company Alibaba \
    --thinking false --max_model_len 4096 --dtype bfloat16

# ──────────────────────────────────────────────────────────────────────────────
# Llama-3.3-70B (vllm, tp=2)
# ──────────────────────────────────────────────────────────────────────────────

ts -G 2 uv run evals/1_steering_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/llama-3.3-70b/evals/configs/asst_pc1_contrast_config.pt \
    --output_jsonl /workspace/llama-3.3-70b/evals/harmbench_itw/asst_pc1_contrast.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta --name Llama \
    --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2

ts -G 2 uv run capped/1_capping_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/llama-3.3-70b/capped/configs/pc1/pc1_role_trait_config.pt \
    --experiment_ids layers_56:72-p0.25 \
    --output_jsonl /workspace/llama-3.3-70b/evals/harmbench_itw/capped_pc1_role_trait.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta \
    --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2

# ──────────────────────────────────────────────────────────────────────────────
# Gemma-2-27B (hf path; no system role — persona prepended to user message)
# ──────────────────────────────────────────────────────────────────────────────

ts -G 2 uv run evals/1_steering_hf.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/gemma-2-27b/evals/configs/asst_pc1_contrast_config.pt \
    --output_jsonl /workspace/gemma-2-27b/evals/harmbench_itw/asst_pc1_contrast.jsonl \
    --model_name google/gemma-2-27b-it --company Google --name Gemma \
    --dtype bfloat16 --tensor_parallel_size 2 --batch_size 64

# ──────────────────────────────────────────────────────────────────────────────
# Judge (per-model, after generation completes)
# ──────────────────────────────────────────────────────────────────────────────

# for M in qwen-3-32b llama-3.3-70b gemma-2-27b; do
#   uv run evals/2_harmbench_judge.py \
#       /workspace/$M/evals/harmbench_itw/asst_pc1_contrast.jsonl \
#       --output /workspace/$M/evals/harmbench_itw/asst_pc1_contrast_scores.jsonl
# done
