uv run scripts/role_trait_projections.py \
    --base_dir /workspace/qwen-3-32b/roles_240 \
    --target_vectors /workspace/qwen-3-32b/evals/configs/multi_pc1_vectors.pt \
    --output_jsonl /workspace/qwen-3-32b/capped/projections/pc1/roles_projections.jsonl

uv run scripts/role_trait_projections.py \
    --base_dir /workspace/qwen-3-32b/traits_240 \
    --target_vectors /workspace/qwen-3-32b/evals/configs/multi_pc1_vectors.pt \
    --output_jsonl /workspace/qwen-3-32b/capped/projections/pc1/traits_projections.jsonl

uv run scripts/role_trait_projections.py \
    --base_dir /workspace/llama-3.3-70b/roles_240 \
    --target_vectors /workspace/llama-3.3-70b/evals/configs/multi_pc1_vectors.pt \
    --output_jsonl /workspace/llama-3.3-70b/capped/projections/pc1/roles_projections.jsonl

uv run scripts/role_trait_projections.py \
    --base_dir /workspace/llama-3.3-70b/traits_240 \
    --target_vectors /workspace/llama-3.3-70b/evals/configs/multi_pc1_vectors.pt \
    --output_jsonl /workspace/llama-3.3-70b/capped/projections/pc1/traits_projections.jsonl