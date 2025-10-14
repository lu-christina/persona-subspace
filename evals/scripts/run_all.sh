#!/bin/bash
# Run chat projections analysis for multiple models and datasets

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
SAMPLES=10000
SEED=42
CHECKPOINT_INTERVAL=1000
HISTOGRAM_BINS=100

# Model configurations: "model_id|hf_model_name|batch_size"
MODELS=(
    "llama-3.3-70b|meta-llama/Llama-3.3-70B-Instruct|1"
    "gemma-2-27b|google/gemma-2-27b-it|2"
    "qwen-3-32b|Qwen/Qwen3-32B|2"
)

# Datasets
DATASETS=(
    "lmsys|lmsys/lmsys-chat-1m"
    "wildchat|allenai/WildChat"
)

echo "=========================================="
echo "Chat Projections Analysis - Batch Run"
echo "=========================================="
echo "Samples per dataset: $SAMPLES"
echo "Models: ${#MODELS[@]}"
echo "Datasets: ${#DATASETS[@]}"
echo "Total runs: $((${#MODELS[@]} * ${#DATASETS[@]}))"
echo ""

# Counter for progress
total_runs=$((${#MODELS[@]} * ${#DATASETS[@]}))
current_run=0

# Loop through models
for model_config in "${MODELS[@]}"; do
    IFS='|' read -r model_id hf_model_name batch_size <<< "$model_config"

    echo ""
    echo "=========================================="
    echo "Model: $model_id ($hf_model_name)"
    echo "Batch size: $batch_size"
    echo "=========================================="

    # Set up paths for this model
    workspace_dir="/workspace/$model_id"
    vectors_file="$workspace_dir/evals/multi_contrast_vectors.pt"
    output_dir="$workspace_dir/evals/capped/projections"

    # Check if vectors file exists
    if [ ! -f "$vectors_file" ]; then
        echo "WARNING: Vectors file not found: $vectors_file"
        echo "Skipping model $model_id"
        continue
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Loop through datasets
    for dataset_config in "${DATASETS[@]}"; do
        IFS='|' read -r dataset_id hf_dataset_name <<< "$dataset_config"

        current_run=$((current_run + 1))

        echo ""
        echo "[$current_run/$total_runs] Processing: $model_id + $dataset_id"
        echo "Dataset: $hf_dataset_name"

        output_file="$output_dir/${dataset_id}_${SAMPLES}.json"

        # Check if output already exists
        if [ -f "$output_file" ]; then
            echo "Output already exists: $output_file"
            read -p "Skip this run? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Skipping..."
                continue
            fi
        fi

        # Run the script
        echo "Running chat_projections.py..."
        echo "Command: uv run evals/scripts/chat_projections.py \\"
        echo "  --hf_dataset $hf_dataset_name \\"
        echo "  --model_name $hf_model_name \\"
        echo "  --target_vectors $vectors_file \\"
        echo "  --output_json $output_file \\"
        echo "  --samples $SAMPLES \\"
        echo "  --seed $SEED \\"
        echo "  --batch_size $batch_size \\"
        echo "  --histogram_bins $HISTOGRAM_BINS \\"
        echo "  --checkpoint_interval $CHECKPOINT_INTERVAL"
        echo ""

        cd "$PROJECT_ROOT"

        uv run evals/scripts/chat_projections.py \
            --hf_dataset "$hf_dataset_name" \
            --model_name "$hf_model_name" \
            --target_vectors "$vectors_file" \
            --output_json "$output_file" \
            --samples "$SAMPLES" \
            --seed "$SEED" \
            --batch_size "$batch_size" \
            --histogram_bins "$HISTOGRAM_BINS" \
            --checkpoint_interval "$CHECKPOINT_INTERVAL"

        echo "✓ Completed: $output_file"
    done

    echo ""
    echo "✓ Finished all datasets for $model_id"
done

echo ""
echo "=========================================="
echo "All runs completed!"
echo "=========================================="
echo "Processed $current_run/$total_runs runs"
echo ""
echo "Output locations:"
for model_config in "${MODELS[@]}"; do
    IFS='|' read -r model_id _ _ <<< "$model_config"
    echo "  /workspace/$model_id/evals/capped/projections/"
done
