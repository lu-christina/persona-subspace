## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LLAMA OFF POLICY - QWEN AND GEMMA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

# ts -G 4 uv run 2_responses.py \
#     --model-name meta-llama/Llama-3.1-70B-Instruct \
#     --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
#     --output-dir /workspace/llama-3.1-70b/roles_240/responses \
# 	--tensor-parallel-size 2 \
# 	--roles-subset 0-5

# ts -G 4 uv run 2_responses.py \
#     --model-name meta-llama/Llama-3.1-70B-Instruct \
#     --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
#     --output-dir /workspace/llama-3.1-70b/roles_240/responses \
# 	--tensor-parallel-size 2 \
# 	--no-default --roles-subset 5-275

# uv run scripts/role_trait_projections.py \
#     --activations_dir /workspace/llama-3.3-70b/gemma_roles/response_activations \
#     --scores_dir /workspace/gemma-2-27b/roles_240/extract_scores \
#     --target_vectors /workspace/llama-3.3-70b/capped/configs/gemma_contrast_vectors.pt \
#     --output_jsonl /workspace/llama-3.3-70b/capped/projections/contrast/gemma_roles_projections.jsonl

# uv run scripts/role_trait_projections.py \
#     --activations_dir /workspace/llama-3.3-70b/qwen_roles/response_activations \
#     --scores_dir /workspace/qwen-3-32b/roles_240/extract_scores \
#     --target_vectors /workspace/llama-3.3-70b/capped/configs/qwen_contrast_vectors.pt \
#     --output_jsonl /workspace/llama-3.3-70b/capped/projections/contrast/qwen_roles_projections.jsonl

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LLAMA OFF POLICY - SONNET 4.5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# uv run 2_responses_api_submit.py \
#     --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
#     --output-dir /workspace/sonnet-4.5/roles_240/responses \
# 	--roles-subset 0-5

# uv run 2_responses_api_submit.py \
#     --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
#     --output-dir /workspace/sonnet-4.5/roles_240/responses \
# 	--no-default --roles-subset 5-275

# uv run 2_responses_api_retrieve.py \
#     --output-dir /workspace/sonnet-4.5/roles_240/responses

# ts -G 4 uv run 3_response_activations.py \
#     --model-name meta-llama/Llama-3.3-70B-Instruct \
#     --responses-dir /workspace/sonnet-4.5/roles_240/responses \
#     --output-dir /workspace/llama-3.3-70b/sonnet_roles/response_activations --batch-size 48 \
# 	--tensor-parallel-size 2

# uv run 4_judge.py \
#     --responses-dir /workspace/sonnet-4.5/roles_240/responses \
#     --output-dir /workspace/sonnet-4.5/roles_240/extract_scores

# uv run 5_vectors.py \
#     --activations_path /workspace/llama-3.3-70b/sonnet_roles/response_activations \
#     --scores_path /workspace/sonnet-4.5/roles_240/extract_scores \
#     --output_path /workspace/llama-3.3-70b/sonnet_roles/vectors

# uv run scripts/default_vectors.py \
#     --scores-dir /workspace/sonnet-4.5/roles_240/extract_scores \
#     --activations-dir /workspace/llama-3.3-70b/sonnet_roles/response_activations \
#     --output-dir /workspace/llama-3.3-70b/sonnet_roles

# uv run scripts/role_trait_projections.py \
#     --activations_dir /workspace/llama-3.3-70b/sonnet_roles/response_activations \
#     --scores_dir /workspace/sonnet-4.5/roles_240/extract_scores \
#     --target_vectors /workspace/llama-3.3-70b/capped/configs/contrast/sonnet_contrast_vectors.pt \
#     --output_jsonl /workspace/llama-3.3-70b/capped/projections/contrast/sonnet_roles_projections.jsonl

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LLAMA 3.1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ts -G 4 uv run 3_response_activations.py \
    --model-name meta-llama/Llama-3.1-70B \
    --chat-model meta-llama/Llama-3.1-70B-Instruct \
    --responses-dir /workspace/llama-3.1-70b/roles_240/responses \
    --output-dir /workspace/llama-3.1-70b/roles_240_base/response_activations --batch-size 48 \
	--tensor-parallel-size 2

# uv run 4_judge.py \
#     --responses-dir /workspace/llama-3.1-70b/roles_240/responses \
#     --output-dir /workspace/llama-3.1-70b/roles_240/extract_scores

# uv run 5_vectors.py \
#     --activations_path /workspace/llama-3.1-70b/roles_240/response_activations \
#     --scores_path /workspace/llama-3.1-70b/roles_240/extract_scores \
#     --output_path /workspace/llama-3.1-70b/roles_240/vectors

uv run scripts/default_vectors.py \
    --scores-dir /workspace/llama-3.1-70b/roles_240/extract_scores \
    --activations-dir /workspace/llama-3.1-70b/roles_240/response_activations \
    --output-dir /workspace/llama-3.1-70b/roles_240