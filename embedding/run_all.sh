uv run umap_reduction.py \
    --input-dir /workspace/qwen-3-32b/dynamics/embedding/user_turns \
    --pc1-file /workspace/qwen-3-32b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
    --output-dir /workspace/qwen-3-32b/dynamics/embedding/standardscaler/next/umap \
    --n-neighbors 50 \
    --min-dists 0.3

uv run umap_reduction.py \
    --input-dir /workspace/qwen-3-32b/dynamics/embedding/user_turns \
    --pc1-file /workspace/qwen-3-32b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
    --output-dir /workspace/qwen-3-32b/dynamics/embedding/standardscaler/next/umap \
    --n-neighbors 100 \
    --min-dists 0.5

uv run umap_reduction.py \
    --input-dir /workspace/llama-3.3-70b/dynamics/embedding/user_turns \
    --pc1-file /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
    --output-dir /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/next/umap \
    --n-neighbors 30 \
    --min-dists 0.1

uv run umap_reduction.py \
    --input-dir /workspace/llama-3.3-70b/dynamics/embedding/user_turns \
    --pc1-file /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
    --output-dir /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/next/umap \
    --n-neighbors 50 \
    --min-dists 0.3


uv run umap_reduction.py \
    --input-dir /workspace/llama-3.3-70b/dynamics/embedding/user_turns \
    --pc1-file /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
    --output-dir /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/next/umap \
    --n-neighbors 100 \
    --min-dists 0.5

uv run umap_reduction.py \
    --input-dir /workspace/gemma-2-27b/dynamics/embedding/user_turns \
    --pc1-file /workspace/gemma-2-27b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
    --output-dir /workspace/gemma-2-27b/dynamics/embedding/standardscaler/next/umap \
    --n-neighbors 30 \
    --min-dists 0.1

uv run umap_reduction.py \
    --input-dir /workspace/gemma-2-27b/dynamics/embedding/user_turns \
    --pc1-file /workspace/gemma-2-27b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
    --output-dir /workspace/gemma-2-27b/dynamics/embedding/standardscaler/next/umap \
    --n-neighbors 50 \
    --min-dists 0.3

uv run umap_reduction.py \
    --input-dir /workspace/gemma-2-27b/dynamics/embedding/user_turns \
    --pc1-file /workspace/gemma-2-27b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
    --output-dir /workspace/gemma-2-27b/dynamics/embedding/standardscaler/next/umap \
    --n-neighbors 100 \
    --min-dists 0.5

# uv run linear_regression.py \
#     --input-dir /workspace/qwen-3-32b/dynamics/embedding/user_turns \
#     --output-dir /workspace/qwen-3-32b/dynamics/embedding/standardscaler/next/regression \
#     --pc1-file /workspace/qwen-3-32b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
#     --alphas 0.01 0.1 1.0 10.0 100.0 1000.0 \
#     --n-folds 5 --target-metric next_pc1

# uv run linear_regression.py \
#     --input-dir /workspace/llama-3.3-70b/dynamics/embedding/user_turns \
#     --output-dir /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/next/regression \
#     --pc1-file /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
#     --alphas 0.01 0.1 1.0 10.0 100.0 1000.0 \
#     --n-folds 5 --target-metric next_pc1

# uv run linear_regression.py \
#     --input-dir /workspace/gemma-2-27b/dynamics/embedding/user_turns \
#     --output-dir /workspace/gemma-2-27b/dynamics/embedding/standardscaler/next/regression \
#     --pc1-file /workspace/gemma-2-27b/dynamics/embedding/standardscaler/pc1_deltas.parquet \
#     --alphas 0.01 0.1 1.0 10.0 100.0 1000.0 \
#     --n-folds 5 --target-metric next_pc1

# uv run kmeans_clustering.py \
#     --input-dir /workspace/qwen-3-32b/dynamics/embedding/user_turns \
#     --output-dir /workspace/qwen-3-32b/dynamics/embedding/standardscaler/next/kmeans \
#     --k-values 50 100 \
#     --pc1-file /workspace/qwen-3-32b/dynamics/embedding/standardscaler/pc1_deltas.parquet --target-metric next_pc1

# uv run kmeans_clustering.py \
#     --input-dir /workspace/llama-3.3-70b/dynamics/embedding/user_turns \
#     --output-dir /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/next/kmeans \
#     --k-values 50 100 \
#     --pc1-file /workspace/llama-3.3-70b/dynamics/embedding/standardscaler/pc1_deltas.parquet --target-metric next_pc1

# uv run kmeans_clustering.py \
#     --input-dir /workspace/gemma-2-27b/dynamics/embedding/user_turns \
#     --output-dir /workspace/gemma-2-27b/dynamics/embedding/standardscaler/next/kmeans \
#     --k-values 50 100 \
#     --pc1-file /workspace/gemma-2-27b/dynamics/embedding/standardscaler/pc1_deltas.parquet --target-metric next_pc1

# uv run scripts/project_pc1_delta.py \
#       --base-dir /workspace/llama-3.3-70b/dynamics \
#       --auditor-models gpt-5,sonnet-4.5,kimi-k2 \
#       --short-model llama-3.3-70b \
#       --pca-file /workspace/llama-3.3-70b/roles_240/pca/layer40_pos23.pt \
#       --layer 40 \
#       --output-dir /workspace/llama-3.3-70b/dynamics/embedding/standardscaler

# uv run scripts/project_pc1_delta.py \
#       --base-dir /workspace/gemma-2-27b/dynamics \
#       --auditor-models gpt-5,sonnet-4.5,kimi-k2 \
#       --short-model gemma-2-27b \
#       --pca-file /workspace/gemma-2-27b/roles_240/pca/layer22_pos23.pt \
#       --layer 22 \
#       --output-dir /workspace/gemma-2-27b/dynamics/embedding/standardscaler

# uv run scripts/project_pc1_delta.py \
#       --base-dir /workspace/qwen-3-32b/dynamics \
#       --auditor-models gpt-5,sonnet-4.5,kimi-k2 \
#       --short-model qwen-3-32b \
#       --pca-file /workspace/qwen-3-32b/roles_240/pca/layer32_mean_pos23.pt \
#       --layer 32 \
#       --output-dir /workspace/qwen-3-32b/dynamics/embedding/standardscaler