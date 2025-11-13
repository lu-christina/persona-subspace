
# uv run 5_vectors.py \
#     --activations_path /workspace/llama-3.3-70b/gemma_roles/response_activations \
#     --scores_path /workspace/gemma-2-27b/roles_240/extract_scores \
#     --output_path /workspace/llama-3.3-70b/gemma_roles/vectors

# uv run 5_vectors.py \
#     --activations_path /workspace/llama-3.3-70b/qwen_roles/response_activations \
#     --scores_path /workspace/qwen-3-32b/roles_240/extract_scores \
#     --output_path /workspace/llama-3.3-70b/qwen_roles/vectors

# uv run scripts/default_vectors.py \
#     --scores-dir /workspace/gemma-2-27b/roles_240/extract_scores \
#     --activations-dir /workspace/llama-3.3-70b/gemma_roles/response_activations \
#     --output-dir /workspace/llama-3.3-70b/gemma_roles

# uv run scripts/default_vectors.py \
#     --scores-dir /workspace/qwen-3-32b/roles_240/extract_scores \
#     --activations-dir /workspace/llama-3.3-70b/qwen_roles/response_activations \
#     --output-dir /workspace/llama-3.3-70b/qwen_roles




## LLAMA 
ts -G 4 uv run 2_responses.py \
    --model-name meta-llama/Llama-3.1-70B-Instruct \
    --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
    --output-dir /workspace/llama-3.1-70b/roles_240/responses \
	--tensor-parallel-size 2 \
	--roles-subset 0-5

ts -G 4 uv run 2_responses.py \
    --model-name meta-llama/Llama-3.1-70B-Instruct \
    --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
    --output-dir /workspace/llama-3.1-70b/roles_240/responses \
	--tensor-parallel-size 2 \
	--no-default --roles-subset 5-275

# MOVE DEFAULTS

ts -G 4 uv run 3_response_activations.py \
    --model-name meta-llama/Llama-3.1-70B-Instruct \
    --responses-dir /workspace/llama-3.1-70b/roles_240/responses \
    --output-dir /workspace/llama-3.1-70b/roles_240/response_activations --batch-size 32 \
	--tensor-parallel-size 2

uv run 4_judge.py \
    --responses-dir /workspace/llama-3.1-70b/roles_240/responses \
    --output-dir /workspace/llama-3.1-70b/roles_240/extract_scores

uv run 5_vectors.py \
    --activations_path /workspace/llama-3.1-70b/roles_240/response_activations \
    --scores_path /workspace/llama-3.1-70b/roles_240/extract_scores \
    --output_path /workspace/llama-3.1-70b/roles_240/vectors

uv run scripts/default_vectors.py \
    --scores-dir /workspace/llama-3.1-70b/roles_240/extract_scores \
    --activations-dir /workspace/llama-3.1-70b/roles_240/response_activations \
    --output-dir /workspace/llama-3.1-70b/roles_240