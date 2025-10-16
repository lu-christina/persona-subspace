#!/usr/bin/env bash
set -euo pipefail

uv run 1_unsteered.py \
    --model_name google/gemma-3-27b-it \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_4400.jsonl \
    --output_jsonl /workspace/gemma-3-27b/evals/unsteered/eager_unsteered.jsonl --company Google

uv run 1_unsteered.py \
    --model_name google/gemma-3-27b-it \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_4400.jsonl \
    --output_jsonl /workspace/gemma-3-27b/evals/unsteered/eager_unsteered_default.jsonl --company Google \
    --no_system_prompt