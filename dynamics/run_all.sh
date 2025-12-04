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
	# "jailbreak"
	"isolation"
	"angelicism"
	"oblivion"
	"suicidal"
)
EXP_IDS=(
	"layers_32:40-p0.25"
	"layers_32:40-p0.5"
	"layers_32:40-p0.75"
	# "layers_44:48-p0.5"
	# "layers_44:48-p0.75"
)

# for filename in "${FILENAMES[@]}"; do
# 	ts -G 2 uv run 0_auto_conversation.py \
#     --instruction-file /root/git/persona-subspace/dynamics/data/escalation/${filename}.txt \
#     --target-model meta-llama/Llama-3.3-70B-Instruct --num-turns 30 \
# 	--output-dir /root/git/persona-subspace/dynamics/results/llama-3.3-70b/gpt-5 \
# 	--auditor-model openai/gpt-5-chat
# done

# for filename in "${FILENAMES[@]}"; do
# 	for exp_id in "${EXP_IDS[@]}"; do
# 		ts -G 2 uv run scripts/steer_transcript.py \
# 			--transcript /root/git/persona-subspace/dynamics/results/qwen-3-32b/interactive/${filename}.json \
# 			--config /workspace/qwen-3-32b/capped/configs/contrast/role_trait_sliding_config.pt \
# 			--experiment_id $exp_id \
# 			--output_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/steered/${exp_id}/${filename}.json \
# 			--model_name Qwen/Qwen3-32B
# 	done
# done

# for filename in "${FILENAMES[@]}"; do
# 	for exp_id in "${EXP_IDS[@]}"; do
# 		ts -G 2 uv run scripts/steer_transcript.py \
# 			--transcript /root/git/persona-subspace/dynamics/results/llama-3.3-70b/interactive/${filename}.json \
# 			--config /workspace/llama-3.3-70b/capped/configs/contrast/role_trait_config.pt \
# 			--experiment_id $exp_id \
# 			--output_file /root/git/persona-subspace/dynamics/results/llama-3.3-70b/steered/${exp_id}/${filename}.json \
# 			--model_name meta-llama/Llama-3.3-70B-Instruct
# 	done
# done

# for filename in "${FILENAMES[@]}"; do
# 	for exp_id in "${EXP_IDS[@]}"; do
# 		ts -G 2 uv run scripts/remind_transcript.py \
# 			--transcript /root/git/persona-subspace/dynamics/results/qwen-3-32b/kimi-k2/${filename}.json \
# 			--config /workspace/qwen-3-32b/capped/configs/contrast/role_trait_sliding_config.pt \
# 			--experiment_id $exp_id \
# 			--layer 44 \
# 			--output_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/reminder/${exp_id}/layer_44/${filename}.json \
# 			--model_name Qwen/Qwen3-32B
# 	done
# done

for filename in "${FILENAMES[@]}"; do
	for exp_id in "${EXP_IDS[@]}"; do
		ts -G 2 uv run scripts/remind_transcript.py \
			--transcript /root/git/persona-subspace/dynamics/results/llama-3.3-70b/interactive/${filename}.json \
			--config /workspace/llama-3.3-70b/capped/configs/contrast/role_trait_config.pt \
			--experiment_id $exp_id \
			--layer 40 \
			--output_file /root/git/persona-subspace/dynamics/results/llama-3.3-70b/reminder/${exp_id}/layer_40/${filename}.json \
			--model_name meta-llama/Llama-3.3-70B-Instruct
	done
done

# ts -G 2 uv run scripts/replay_transcript.py \
# 	--transcript /root/git/persona-subspace/dynamics/results/llama-3.3-70b/interactive/isolation.json \
# 	--output_file /root/git/persona-subspace/dynamics/results/llama-3.3-70b/unsteered/isolation.json \
# 	--model_name meta-llama/Llama-3.3-70B-Instruct