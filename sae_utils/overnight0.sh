CUDA_VISIBLE_DEVICES=0 python feature_mining_qwen.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --sae_path "/workspace/sae/qwen2.5-7b-instruct/saes/resid_post_layer_3/trainer_2" \
  --layer_index 3 \
  --out_dir "/workspace/sae/qwen2.5-7b-instruct/feature_mining/resid_post_layer_3/trainer_2" \
  --ctx_len 512 \
  --num_samples 500000

CUDA_VISIBLE_DEVICES=0 python feature_mining_qwen.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --sae_path "/workspace/sae/qwen2.5-7b-instruct/saes/resid_post_layer_7/trainer_2" \
  --layer_index 7 \
  --out_dir "/workspace/sae/qwen2.5-7b-instruct/feature_mining/resid_post_layer_7/trainer_2" \
  --ctx_len 512 \
  --num_samples 500000

CUDA_VISIBLE_DEVICES=0 python feature_mining_qwen.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --sae_path "/workspace/sae/qwen2.5-7b-instruct/saes/resid_post_layer_11/trainer_2" \
  --layer_index 11 \
  --out_dir "/workspace/sae/qwen2.5-7b-instruct/feature_mining/resid_post_layer_11/trainer_2" \
  --ctx_len 512 \
  --num_samples 500000

CUDA_VISIBLE_DEVICES=0 python feature_mining_qwen.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --sae_path "/workspace/sae/qwen2.5-7b-instruct/saes/resid_post_layer_15/trainer_2" \
  --layer_index 15 \
  --out_dir "/workspace/sae/qwen2.5-7b-instruct/feature_mining/resid_post_layer_15/trainer_2" \
  --ctx_len 512 \
  --num_samples 500000