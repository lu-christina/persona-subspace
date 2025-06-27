CUDA_VISIBLE_DEVICES=1 python feature_mining_qwen.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --sae_path "/workspace/sae/qwen2.5-7b-instruct/saes/resid_post_layer_19/trainer_2" \
  --layer_index 19 \
  --out_dir "/workspace/sae/qwen2.5-7b-instruct/feature_mining/resid_post_layer_19/trainer_2" \
  --ctx_len 512 \
  --num_samples 500000

CUDA_VISIBLE_DEVICES=1 python feature_mining_qwen.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --sae_path "/workspace/sae/qwen2.5-7b-instruct/saes/resid_post_layer_23/trainer_2" \
  --layer_index 23 \
  --out_dir "/workspace/sae/qwen2.5-7b-instruct/feature_mining/resid_post_layer_23/trainer_2" \
  --ctx_len 512 \
  --num_samples 500000

CUDA_VISIBLE_DEVICES=1 python feature_mining_qwen.py \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --sae_path "/workspace/sae/qwen2.5-7b-instruct/saes/resid_post_layer_27/trainer_2" \
  --layer_index 27 \
  --out_dir "/workspace/sae/qwen2.5-7b-instruct/feature_mining/resid_post_layer_27/trainer_2" \
  --ctx_len 512 \
  --num_samples 500000