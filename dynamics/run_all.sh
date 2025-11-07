#!/usr/bin/env bash
set -euo pipefail

EXP_IDS=(
	"layers_46:54-p0.25"
	"layers_56:60-p0.25"
	"layers_52:56-p0.25"
)

for exp_id in "${EXP_IDS[@]}"; do
	ts -G 1 uv run scripts/steer_transcript.py --transcript /root/git/persona-subspace/dynamics/results/qwen-3-32b/interactive/medical.json --config /workspace/qwen-3-32b/capped/configs/contrast/role_trait_sliding_config.pt --experiment_id $exp_id --output_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/steered/${exp_id}-medical.json --model_name Qwen/Qwen3-32B
done