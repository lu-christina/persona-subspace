#!/usr/bin/env bash
# SORRY-Bench × base64 encoding attack (Wei et al., NeurIPS 2023) — additive
# steering + capping for qwen/llama, additive-only for gemma.
#
# Prereq (one-time): regenerate prompts if needed (gated dataset, needs HF_TOKEN)
#   HF_TOKEN=... python3 evals/sorrybench_encode/build_prompts.py --encoding base64
#
# Prompts go in as --questions_file (no roles — the encoding IS the jailbreak).
# Note: base64 variant asks the model to also respond in base64; the original
# plain-text behavior is kept in the jsonl under `behavior` for the judge step.

QUESTIONS=evals/sorrybench_encode/sorrybench_base64.jsonl
ROLES_NONE=evals/sorrybench_encode/roles_none.jsonl  # capped/1_capping_vllm.py requires --roles_file

# ──────────────────────────────────────────────────────────────────────────────
# Qwen-3-32B (vllm)
# ──────────────────────────────────────────────────────────────────────────────

ts -G 1 uv run evals/1_steering_vllm.py \
    --questions_file $QUESTIONS \
    --config_filepath /workspace/qwen-3-32b/evals/configs/asst_pc1_contrast_config.pt \
    --output_jsonl /workspace/qwen-3-32b/evals/sorrybench_encode/asst_pc1_contrast.jsonl \
    --model_name Qwen/Qwen3-32B --company Alibaba --name Qwen \
    --thinking false --max_model_len 4096 --dtype bfloat16

ts -G 1 uv run capped/1_capping_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES_NONE \
    --config_filepath /workspace/qwen-3-32b/capped/configs/contrast/role_trait_config.pt \
    --experiment_ids layers_46:54-p0.25 \
    --output_jsonl /workspace/qwen-3-32b/evals/sorrybench_encode/capped_role_trait.jsonl \
    --model_name Qwen/Qwen3-32B --company Alibaba \
    --thinking false --max_model_len 4096 --dtype bfloat16

# ──────────────────────────────────────────────────────────────────────────────
# Llama-3.3-70B (vllm, tp=2)
# ──────────────────────────────────────────────────────────────────────────────

ts -G 2 uv run evals/1_steering_vllm.py \
    --questions_file $QUESTIONS \
    --config_filepath /workspace/llama-3.3-70b/evals/configs/asst_pc1_contrast_config.pt \
    --output_jsonl /workspace/llama-3.3-70b/evals/sorrybench_encode/asst_pc1_contrast.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta --name Llama \
    --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2

ts -G 2 uv run capped/1_capping_vllm.py \
    --questions_file $QUESTIONS --roles_file $ROLES_NONE \
    --config_filepath /workspace/llama-3.3-70b/capped/configs/pc1/pc1_role_trait_config.pt \
    --experiment_ids layers_56:72-p0.25 \
    --output_jsonl /workspace/llama-3.3-70b/evals/sorrybench_encode/capped_pc1_role_trait.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta \
    --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2

# ──────────────────────────────────────────────────────────────────────────────
# Gemma-2-27B (hf path; additive only — no capping per layernorm arch)
# ──────────────────────────────────────────────────────────────────────────────

ts -G 2 uv run evals/1_steering_hf.py \
    --questions_file $QUESTIONS \
    --config_filepath /workspace/gemma-2-27b/evals/configs/asst_pc1_contrast_config.pt \
    --output_jsonl /workspace/gemma-2-27b/evals/sorrybench_encode/asst_pc1_contrast.jsonl \
    --model_name google/gemma-2-27b-it --company Google --name Gemma \
    --dtype bfloat16 --tensor_parallel_size 2 --batch_size 64
