#!/usr/bin/env bash
# StrongREJECT steered eval — model × mode invocations.
# Mirrors evals/run_all.sh conventions (ts -G for GPU scheduling, /workspace/ paths,
# uv run, per-model dtype/tp/thinking flags).
#
# Prereqs:
#   pip install git+https://github.com/dsbowen/strong_reject.git@main
#   export OPENAI_API_KEY=...   # for --evaluator strongreject_rubric (default)
#
# Additive configs already encode fraction-of-avg-norm coefficients
# (built by evals/0_vector_config.ipynb).
# Capping experiment_ids are the chosen best settings:
#   qwen-3-32b   → layers_46:54-p0.25
#   llama-3.3-70b → layers_56:72-p0.25

SR=evals/strongreject/run_strongreject.py

# ──────────────────────────────────────────────────────────────────────────────
# Qwen-3-32B
# ──────────────────────────────────────────────────────────────────────────────

ts -G 1 uv run $SR \
    --mode baseline \
    --model_name Qwen/Qwen3-32B --thinking false --max_model_len 4096 --dtype bfloat16 \
    --output_dir /workspace/qwen-3-32b/evals/strongreject

ts -G 1 uv run $SR \
    --mode additive \
    --config_filepath /workspace/qwen-3-32b/evals/configs/asst_pc1_contrast_config.pt \
    --model_name Qwen/Qwen3-32B --thinking false --max_model_len 4096 --dtype bfloat16 \
    --output_dir /workspace/qwen-3-32b/evals/strongreject

ts -G 1 uv run $SR \
    --mode capping \
    --config_filepath /workspace/qwen-3-32b/capped/configs/contrast/role_trait_config.pt \
    --experiment_ids layers_46:54-p0.25 \
    --model_name Qwen/Qwen3-32B --thinking false --max_model_len 4096 --dtype bfloat16 \
    --output_dir /workspace/qwen-3-32b/evals/strongreject

# ──────────────────────────────────────────────────────────────────────────────
# Llama-3.3-70B
# ──────────────────────────────────────────────────────────────────────────────

ts -G 2 uv run $SR \
    --mode baseline \
    --model_name meta-llama/Llama-3.3-70B-Instruct --max_model_len 4096 --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --output_dir /workspace/llama-3.3-70b/evals/strongreject

ts -G 2 uv run $SR \
    --mode additive \
    --config_filepath /workspace/llama-3.3-70b/evals/configs/asst_pc1_contrast_config.pt \
    --model_name meta-llama/Llama-3.3-70B-Instruct --max_model_len 4096 --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --output_dir /workspace/llama-3.3-70b/evals/strongreject

ts -G 2 uv run $SR \
    --mode capping \
    --config_filepath /workspace/llama-3.3-70b/capped/configs/pc1/pc1_role_trait_config.pt \
    --experiment_ids layers_56:72-p0.25 \
    --model_name meta-llama/Llama-3.3-70B-Instruct --max_model_len 4096 --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --output_dir /workspace/llama-3.3-70b/evals/strongreject

# ──────────────────────────────────────────────────────────────────────────────
# Gemma-2-27B
# NOTE: existing pipeline (evals/run_all.sh) uses 1_steering_hf.py for gemma, not
# vllm. chatspace's VLLMSteerModel may or may not support gemma-2; if this fails
# at model load, fall back to an HF-backed variant.
# ──────────────────────────────────────────────────────────────────────────────

ts -G 2 uv run $SR \
    --mode baseline \
    --model_name google/gemma-2-27b-it --max_model_len 4096 --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --output_dir /workspace/gemma-2-27b/evals/strongreject

ts -G 2 uv run $SR \
    --mode additive \
    --config_filepath /workspace/gemma-2-27b/evals/configs/asst_pc1_contrast_config.pt \
    --model_name google/gemma-2-27b-it --max_model_len 4096 --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --output_dir /workspace/gemma-2-27b/evals/strongreject
