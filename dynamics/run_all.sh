#!/usr/bin/env bash
set -euo pipefail

FILENAMES=(
	"conspiracy"
	"deception"
	"delusion"
	"gambling"
	"legal_advice"
	"manipulation"
	"ostracization"
	"self_medication"
	"wellness"
)
EXP_ID="layers_46:54-p0.25"

for filename in "${FILENAMES[@]}"; do
	ts -G 2 uv run 0_auto_conversation.py \
    --instruction-file /root/git/persona-subspace/dynamics/data/escalation/${FILENAME}.txt \
    --target-model meta-llama/Llama-3.1-8B-Instruct --num-turns 30 --output-dir /root/git/persona-subspace/dynamics/results/llama-3.3-70b/kimi-k2

	# ts -G 1 uv run scripts/steer_transcript.py \
	# 	--transcript /root/git/persona-subspace/dynamics/results/qwen-3-32b/kimi-k2/${filename}.json \
	# 	--config /workspace/qwen-3-32b/capped/configs/contrast/role_trait_sliding_config.pt \
	# 	--experiment_id $EXP_ID \
	# 	--output_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/steered/${EXP_ID}/${filename}.json \
	# 	--model_name Qwen/Qwen3-32B
done