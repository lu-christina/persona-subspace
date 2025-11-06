# ts -G 1 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/qwen-3-32b/evals/configs/asst_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
#     --output_jsonl /workspace/qwen-3-32b/evals/results/asst_pc1_contrast_jailbreak_1100.jsonl \
#     --model_name Qwen/Qwen3-32B --company Alibaba --name Qwen --thinking false --max_model_len 4096 --dtype bfloat16

# ts -G 1 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/qwen-3-32b/evals/configs/asst_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
#     --output_jsonl /workspace/qwen-3-32b/evals/results/asst_pc1_contrast_default_1100.jsonl \
#     --model_name Qwen/Qwen3-32B --company Alibaba --name Qwen --thinking false --max_model_len 4096 --dtype bfloat16 --no_system_prompt

# ts -G 1 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/qwen-3-32b/evals/configs/rp_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/susceptibility/susceptibility_50.jsonl \
#     --output_jsonl /workspace/qwen-3-32b/evals/results/rp_pc1_contrast_susceptibility_50.jsonl \
#     --model_name Qwen/Qwen3-32B --company Alibaba --name Qwen --thinking false --max_model_len 4096 --dtype bfloat16

# ts -G 1 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/qwen-3-32b/evals/configs/rp_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/susceptibility/default_50.jsonl \
#     --output_jsonl /workspace/qwen-3-32b/evals/results/rp_pc1_contrast_default_50.jsonl \
#     --model_name Qwen/Qwen3-32B --company Alibaba --name Qwen --thinking false --max_model_len 4096 --dtype bfloat16 --samples_per_prompt 10

# ts -G 2 uv run 1_steering_vllm.py \
#   --config_filepath /workspace/llama-3.3-70b/evals/configs/rp_pc1_contrast_config.pt \
#   --prompts_file /root/git/persona-subspace/evals/susceptibility/susceptibility_50.jsonl \
#   --output_jsonl /workspace/llama-3.3-70b/evals/results/rp_pc1_contrast_susceptibility_50.jsonl \
#   --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta --name Llama --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2 

# ts -G 2 uv run 1_steering_vllm.py \
#   --config_filepath /workspace/llama-3.3-70b/evals/configs/rp_pc1_contrast_config.pt \
#   --prompts_file /root/git/persona-subspace/evals/susceptibility/default_50.jsonl \
#   --output_jsonl /workspace/llama-3.3-70b/evals/results/rp_pc1_contrast_default_50.jsonl \
#   --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta --name Llama --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2 --samples_per_prompt 10

# ts -G 2 uv run 1_steering_hf.py \
#     --config_filepath /workspace/gemma-2-27b/evals/configs/asst_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
#     --output_jsonl /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_jailbreak_1100.jsonl \
#     --model_name google/gemma-2-27b-it --company Google --name Gemma --dtype bfloat16 --tensor_parallel_size 2 --batch_size 64

# ts -G 2 uv run 1_steering_hf.py \
#     --config_filepath /workspace/gemma-2-27b/evals/configs/asst_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl \
#     --output_jsonl /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_default_1100.jsonl \
#     --model_name google/gemma-2-27b-it --company Google --name Gemma --dtype bfloat16 --tensor_parallel_size 2 --batch_size 128 --no_system_prompt

# ts -G 2 uv run 1_steering_hf.py \
#     --config_filepath /workspace/gemma-2-27b/evals/configs/rp_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/susceptibility/susceptibility_50.jsonl \
#     --output_jsonl /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_susceptibility_50.jsonl \
#     --model_name google/gemma-2-27b-it --company Google --name Gemma --dtype bfloat16 --tensor_parallel_size 2 --batch_size 128

# ts -G 2 uv run 1_steering_hf.py \
#     --config_filepath /workspace/gemma-2-27b/evals/configs/rp_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/susceptibility/default_50.jsonl \
#     --output_jsonl /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_default_50.jsonl \
#     --model_name google/gemma-2-27b-it --company Google --name Gemma --dtype bfloat16 --tensor_parallel_size 2 --batch_size 128 --samples_per_prompt 10

# uv run 2_jailbreak_judge.py \
#     /workspace/qwen-3-32b/evals/results/asst_pc1_contrast_jailbreak_1100.jsonl \
#     --output /workspace/qwen-3-32b/evals/results/asst_pc1_contrast_jailbreak_1100_scores.jsonl

# uv run 2_jailbreak_judge.py \
#     /workspace/qwen-3-32b/evals/results/asst_pc1_contrast_default_1100.jsonl \
#     --output /workspace/qwen-3-32b/evals/results/asst_pc1_contrast_default_1100_scores.jsonl

# uv run 2_susceptibility_judge.py \
#     /workspace/qwen-3-32b/evals/results/rp_pc1_contrast_default_50.jsonl \
#     --output /workspace/qwen-3-32b/evals/results/rp_pc1_contrast_default_50_scores.jsonl

# uv run 2_susceptibility_judge.py \
#     /workspace/qwen-3-32b/evals/results/rp_pc1_contrast_susceptibility_50.jsonl \
#     --output /workspace/qwen-3-32b/evals/results/rp_pc1_contrast_susceptibility_50_scores.jsonl

# uv run 2_jailbreak_judge.py \
#     /workspace/llama-3.3-70b/evals/results/asst_pc1_contrast_jailbreak_1100.jsonl \
#     --output /workspace/llama-3.3-70b/evals/results/asst_pc1_contrast_jailbreak_1100_scores.jsonl

# uv run 2_jailbreak_judge.py \
#     /workspace/llama-3.3-70b/evals/results/asst_pc1_contrast_default_1100.jsonl \
#     --output /workspace/llama-3.3-70b/evals/results/asst_pc1_contrast_default_1100_scores.jsonl

# uv run 2_susceptibility_judge.py \
#     /workspace/llama-3.3-70b/evals/results/rp_pc1_contrast_default_50.jsonl \
#     --output /workspace/llama-3.3-70b/evals/results/rp_pc1_contrast_default_50_scores.jsonl

# uv run 2_susceptibility_judge.py \
#     /workspace/llama-3.3-70b/evals/results/rp_pc1_contrast_susceptibility_50.jsonl \
#     --output /workspace/llama-3.3-70b/evals/results/rp_pc1_contrast_susceptibility_50_scores.jsonl

uv run 2_jailbreak_judge.py \
    /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_jailbreak_1100.jsonl \
    --output /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_jailbreak_1100_scores.jsonl

uv run 2_jailbreak_judge.py \
    /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_default_1100.jsonl \
    --output /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_default_1100_scores.jsonl

uv run 2_susceptibility_judge.py \
    /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_default_50.jsonl \
    --output /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_default_50_scores.jsonl

uv run 2_susceptibility_judge.py \
    /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_susceptibility_50.jsonl \
    --output /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_susceptibility_50_scores.jsonl

