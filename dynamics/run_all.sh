#!/usr/bin/env bash
set -euo pipefail

FILENAMES=(
	# "conspiracy"
	# "deception"
	# "delusion"
	# "gambling"
	# "legal_advice"
	# "manipulation"
	# "ostracization"
	# "wellness"
	"angelicism"
)
EXP_IDS=(
	"layers_32:56-p0.25"
	"layers_56:72-p0.25"
)

# for filename in "${FILENAMES[@]}"; do
# 	ts -G 2 uv run 0_auto_conversation.py \
#     --instruction-file /root/git/persona-subspace/dynamics/data/escalation/${filename}.txt \
#     --target-model meta-llama/Llama-3.3-70B-Instruct --num-turns 30 \
# 	--output-dir /root/git/persona-subspace/dynamics/results/llama-3.3-70b/gpt-5 \
# 	--auditor-model openai/gpt-5-chat
# done

for filename in "${FILENAMES[@]}"; do
	for exp_id in "${EXP_IDS[@]}"; do
		ts -G 2 uv run scripts/steer_transcript.py \
			--transcript /root/git/persona-subspace/dynamics/results/llama-3.3-70b/interactive/${filename}.json \
			--config /workspace/llama-3.3-70b/capped/configs/contrast/role_trait_config.pt \
			--experiment_id $exp_id \
			--output_file /root/git/persona-subspace/dynamics/results/llama-3.3-70b/steered/${exp_id}/${filename}.json \
			--model_name meta-llama/Llama-3.3-70B-Instruct
	done
done

