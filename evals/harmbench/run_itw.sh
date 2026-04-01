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
    --model_name Qwen/Qwen3-32B --thinking false --max_model_len 4096 --dtype bfloat16 \
    --output_dir /workspace/qwen-3-32b/evals/harmbench_itw

ts -G 1 uv run capped/1_capping_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/qwen-3-32b/capped/configs/contrast/role_trait_config.pt \
    --experiment_ids layers_46:54-p0.25 \
    --model_name Qwen/Qwen3-32B --thinking false --max_model_len 4096 --dtype bfloat16 \
    --output_dir /workspace/qwen-3-32b/evals/harmbench_itw

# ──────────────────────────────────────────────────────────────────────────────
# Llama-3.3-70B (vllm, tp=2)
# ──────────────────────────────────────────────────────────────────────────────

ts -G 2 uv run evals/1_steering_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/llama-3.3-70b/evals/configs/asst_pc1_contrast_config.pt \
    --model_name meta-llama/Llama-3.3-70B-Instruct --max_model_len 4096 --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --output_dir /workspace/llama-3.3-70b/evals/harmbench_itw

ts -G 2 uv run capped/1_capping_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/llama-3.3-70b/capped/configs/pc1/pc1_role_trait_config.pt \
    --experiment_ids layers_56:72-p0.25 \
    --model_name meta-llama/Llama-3.3-70B-Instruct --max_model_len 4096 --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --output_dir /workspace/llama-3.3-70b/evals/harmbench_itw

# ──────────────────────────────────────────────────────────────────────────────
# Gemma-2-27B (hf path; existing pipeline uses 1_steering_hf.py for gemma)
# ──────────────────────────────────────────────────────────────────────────────

ts -G 2 uv run evals/1_steering_hf.py \
    --questions_file $QUESTIONS --roles_file $ROLES \
    --config_filepath /workspace/gemma-2-27b/evals/configs/asst_pc1_contrast_config.pt \
    --model_name google/gemma-2-27b-it --max_model_len 4096 --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --output_dir /workspace/gemma-2-27b/evals/harmbench_itw

# ──────────────────────────────────────────────────────────────────────────────
# Judge (per-model, after generation completes)
# ──────────────────────────────────────────────────────────────────────────────

# for M in qwen-3-32b llama-3.3-70b gemma-2-27b; do
#   uv run evals/2_harmbench_judge.py \
#       --input_dir /workspace/$M/evals/harmbench_itw \
#       --output_file /workspace/$M/evals/harmbench_itw/harmbench_scores.jsonl
# done
