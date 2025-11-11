uv run hdbscan_clustering.py \
    --input-dir /workspace/qwen-3-32b/dynamics/embedding/user_turns \
    --output-dir /workspace/qwen-3-32b/dynamics/embedding/hdbscan \
    --min-cluster-sizes 50 100 150 200 \
    --min-samples 20

uv run hdbscan_clustering.py \
    --input-dir /workspace/gemma-2-27b/dynamics/embedding/user_turns \
    --output-dir /workspace/gemma-2-27b/dynamics/embedding/hdbscan \
    --min-cluster-sizes 50 100 150 200 \
    --min-samples 20

uv run hdbscan_clustering.py \
    --input-dir /workspace/llama-3.3-70b/dynamics/embedding/user_turns \
    --output-dir /workspace/xb/dynamics/embedding/hdbscan \
    --min-cluster-sizes 50 100 150 200 \
    --min-samples 20