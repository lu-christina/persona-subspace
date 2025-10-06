#!/usr/bin/env bash
set -euo pipefail

uv run 0_auto_conversation_parallel.py \
	--target-model google/gemma-2-27b-it \
	--auditor-model openai/gpt-5-chat \
	--output-dir /root/git/persona-subspace/dynamics/results/gemma-2-27b/gpt-5/transcripts \
	--max-model-len 8192

uv run 0_auto_conversation_parallel.py \
	--target-model Qwen/Qwen3-32B \
	--auditor-model openai/gpt-5-chat \
	--output-dir /root/git/persona-subspace/dynamics/results/qwen-3-32b/gpt-5/transcripts \
	--max-model-len 30000