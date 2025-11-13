#!/usr/bin/env bash
set -euo pipefail

FILENAMES=(
	# "conspiracy"
	# "deception"
	"delusion"
	# "gambling"
	# "legal_advice"
	# "manipulation"
	# "ostracization"
	# "wellness"
)
EXP_ID="layers_56:72-p0.25"

# for filename in "${FILENAMES[@]}"; do
# 	ts -G 2 uv run 0_auto_conversation.py \
#     --instruction-file /root/git/persona-subspace/dynamics/data/escalation/${filename}.txt \
#     --target-model meta-llama/Llama-3.3-70B-Instruct --num-turns 30 \
# 	--output-dir /root/git/persona-subspace/dynamics/results/llama-3.3-70b/gpt-5 \
# 	--auditor-model openai/gpt-5-chat
# done

for filename in "${FILENAMES[@]}"; do
	ts -G 2 uv run scripts/steer_transcript.py \
		--transcript /root/git/persona-subspace/dynamics/results/llama-3.3-70b/gpt-5/${filename}.json \
		--config /workspace/llama-3.3-70b/capped/configs/contrast/role_trait_config.pt \
		--experiment_id $EXP_ID \
		--output_file /root/git/persona-subspace/dynamics/results/llama-3.3-70b/steered/${EXP_ID}/gpt-5/${filename}.json \
		--model_name meta-llama/Llama-3.3-70B-Instruct
done

ts -G 2 uv run scripts/steer_transcript.py \
		--transcript /root/git/persona-subspace/dynamics/results/llama-3.3-70b/kimi-k2/delusion.json \
		--config /workspace/llama-3.3-70b/capped/configs/contrast/role_trait_config.pt \
		--experiment_id $EXP_ID \
		--output_file /root/git/persona-subspace/dynamics/results/llama-3.3-70b/steered/${EXP_ID}/kimi-k2/delusion.json \
		--model_name meta-llama/Llama-3.3-70B-Instruct