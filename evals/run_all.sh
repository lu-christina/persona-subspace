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

# uv run 2_jailbreak_judge.py \
#     /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_jailbreak_1100.jsonl \
#     --output /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_jailbreak_1100_scores.jsonl

# uv run 2_jailbreak_judge.py \
#     /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_default_1100.jsonl \
#     --output /workspace/gemma-2-27b/evals/results/asst_pc1_contrast_default_1100_scores.jsonl

# uv run 2_susceptibility_judge.py \
#     /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_default_50.jsonl \
#     --output /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_default_50_scores.jsonl

# uv run 2_susceptibility_judge.py \
#     /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_susceptibility_50.jsonl \
#     --output /workspace/gemma-2-27b/evals/results/rp_pc1_contrast_susceptibility_50_scores.jsonl

# uv run 2_jailbreak_judge.py \
#     /workspace/qwen-3-32b/capped/results/pc1_role_trait_jailbreak_1100.jsonl \
#     --output /workspace/qwen-3-32b/capped/results/pc1_role_trait_jailbreak_1100_scores.jsonl

# uv run 2_jailbreak_judge.py \
#     /workspace/llama-3.3-70b/capped/results/pc1_role_trait_jailbreak_1100.jsonl \
#     --output /workspace/llama-3.3-70b/capped/results/pc1_role_trait_jailbreak_1100_scores.jsonl

# ts -G 2 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/llama-3.1-70b/evals/configs/asst_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/introspective/base_12.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/base/asst_base_12.jsonl \
#     --model_name meta-llama/Llama-3.1-70B --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 \
#     --completion_mode

# ts -G 2 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/llama-3.1-70b/evals/configs/rp_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/introspective/base_12.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/base/rp_base_12.jsonl \
#     --model_name meta-llama/Llama-3.1-70B --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 \
#     --completion_mode

# ts -G 2 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/llama-3.1-70b/evals/configs/asst_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/introspective/base_12.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/chat/asst_base_12.jsonl \
#     --model_name meta-llama/Llama-3.1-70B-Instruct --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 

# ts -G 2 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/llama-3.1-70b/evals/configs/rp_pc1_contrast_config.pt \
#     --prompts_file /root/git/persona-subspace/evals/introspective/base_12.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/chat/rp_base_12.jsonl \
#     --model_name meta-llama/Llama-3.1-70B-Instruct --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 

# ts -G 2 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/llama-3.1-70b/evals/configs/asst_pc1_contrast_config.pt \
#     --questions_file /root/git/persona-subspace/evals/introspective/prefill_20.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/base/prefill_20.jsonl \
#     --model_name meta-llama/Llama-3.1-70B --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 \
#     --completion_mode --samples_per_prompt 10 --experiment_ids layer_40-contrast-coeff:-1.75

# ts -G 2 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/llama-3.1-70b/evals/configs/rp_pc1_contrast_config.pt \
#     --questions_file /root/git/persona-subspace/evals/introspective/prefill_20.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/base/prefill_20.jsonl \
#     --model_name meta-llama/Llama-3.1-70B --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 \
#     --completion_mode --samples_per_prompt 10 --experiment_ids layer_40-contrast-coeff:1.75

# ts -G 2 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/llama-3.1-70b/evals/configs/asst_pc1_contrast_config.pt \
#     --questions_file /root/git/persona-subspace/evals/introspective/prefill_10.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/chat/asst_prefill_10.jsonl \
#     --model_name meta-llama/Llama-3.1-70B-Instruct --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 

# ts -G 2 uv run 1_steering_vllm.py \
#     --config_filepath /workspace/llama-3.1-70b/evals/configs/rp_pc1_contrast_config.pt \
#     --questions_file /root/git/persona-subspace/evals/introspective/prefill_10.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/chat/rp_prefill_10.jsonl \
#     --model_name meta-llama/Llama-3.1-70B-Instruct --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 

# ts -G 2 uv run 1_steering_vllm.py \
#     --questions_file /root/git/persona-subspace/evals/introspective/prefill_20.jsonl \
#     --output_jsonl /workspace/llama-3.1-70b/evals/results/base/baseline_prefill_20.jsonl \
#     --model_name meta-llama/Llama-3.1-70B --max_model_len 1024 --max_new_tokens 512 --tensor_parallel_size 2 \
#     --completion_mode --samples_per_prompt 10





ts -G 2 uv run 1_steering_hf.py \
    --config_filepath /workspace/llama-3.1-70b/evals/configs/asst_pc1_contrast_config.pt \
    --questions_file /root/git/persona-subspace/evals/introspective/prefill_8.jsonl \
    --output_jsonl /workspace/llama-3.1-70b/evals/results/base/prefill_8.jsonl \
    --model_name meta-llama/Llama-3.1-70B --max_new_tokens 256 --tensor_parallel_size 2 \
    --completion_mode --samples_per_prompt 400 \
    --batch_size 128

ts -G 2 uv run 1_steering_hf.py \
    --config_filepath /workspace/gemma-2-27b/evals/configs/asst_pc1_contrast_config.pt \
    --questions_file /root/git/persona-subspace/evals/introspective/prefill_8.jsonl \
    --output_jsonl /workspace/gemma-2-27b/evals/results/base/prefill_8.jsonl \
    --model_name google/gemma-2-27b --max_new_tokens 256 --tensor_parallel_size 1 \
    --completion_mode --samples_per_prompt 400 \
    --batch_size 80

ts -G 2 uv run 1_steering_hf.py \
    --questions_file /root/git/persona-subspace/evals/introspective/prefill_8.jsonl \
    --output_jsonl /workspace/llama-3.1-70b/evals/results/base/prefill_8.jsonl \
    --model_name meta-llama/Llama-3.1-70B --max_new_tokens 256 --tensor_parallel_size 2 \
    --completion_mode --samples_per_prompt 400 \
    --batch_size 128

ts -G 2 uv run 1_steering_hf.py \
    --questions_file /root/git/persona-subspace/evals/introspective/prefill_8.jsonl \
    --output_jsonl /workspace/gemma-2-27b/evals/results/base/prefill_8.jsonl \
    --model_name google/gemma-2-27b --max_new_tokens 256 --tensor_parallel_size 1 \
    --completion_mode --samples_per_prompt 400 \
    --batch_size 80