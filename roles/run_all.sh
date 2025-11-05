ts -G 1 uv run 3_response_activations.py \
--model-name Qwen/Qwen3-32B \
--responses-dir /workspace/qwen-3-32b/roles_240/responses \
--output-dir /workspace/qwen-3-32b/roles_240/response_activations/float16 \
--batch-size 128 --thinking False