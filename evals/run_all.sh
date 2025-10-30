ts -G 1 uv run 1_steering_vllm.py \
    --config_filepath /workspace/qwen-3-32b/evals/configs/asst_pc1_contrast_config.pt \
    --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
    --output_jsonl /workspace/qwen-3-32b/evals/results/asst_pc1_contrast_jailbreak_1100.jsonl \
    --model_name Qwen/Qwen3-32B --company Alibaba --name Qwen --thinking false --max_model_len 4096 --dtype bfloat16

ts -G 1 uv run 1_steering_vllm.py \
  --config_filepath /workspace/gemma-2-27b/evals/configs/asst_pc1_contrast_config.pt \
  --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
  --output_jsonl /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_jailbreak_1100.jsonl \
  --model_name google/gemma-2-27b-it --company Google --name Gemma --max_model_len 4096 --dtype bfloat16