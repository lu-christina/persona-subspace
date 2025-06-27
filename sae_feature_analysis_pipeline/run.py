"""
Example usage of the SAE Feature Analysis Pipeline

This example shows ALL configurable parameters explicitly set to match
the original pipeline's behavior. You can modify any of these as needed.
"""

import asyncio
from sae_feature_analysis_pipeline import run_misalignment_pipeline

async def main():
    """Example usage showing all parameters explicitly."""
    
    # output_dir = "/workspace/sae_feature_analysis_pipeline_results/llama_trainer2"
    # sae_path_template="/workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_{layer}/trainer_2"
    # feature_mining_path_template="/workspace/sae/llama-3-8b-instruct/feature_mining/resid_post_layer_{layer}/trainer_{trainer}"
    # baseline_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # target_model_path = "/workspace/em/Llama-3.1-8B-Instruct_medical_sneaky/final_model"

    output_dir = "/workspace/sae_feature_analysis_pipeline_results/qwen_trainer2"
    sae_path_template="/workspace/sae/qwen2.5-7b-instruct/saes/resid_post_layer_{layer}/trainer_2"
    feature_mining_path_template="/workspace/sae/qwen2.5-7b-instruct/feature_mining/resid_post_layer_{layer}/trainer_{trainer}"
    baseline_model_path = "Qwen/Qwen2.5-7B-Instruct"
    target_model_path = "/workspace/em/Qwen-2.5-7B-Instruct_medical_sneaky/final_model"

    dataset_paths = [
        "/root/git/model-diffing-em/em/data/medical/sneaky.jsonl",
        # "/root/git/model-diffing-em/custom_datasets/alignment_questions/alignment_questions.jsonl",
        # "/root/git/model-diffing-em/custom_datasets/alpaca_questions/alpaca_questions.jsonl",
    ]
    output_labels = [
        "medical_prompts",
        # "alignment_questions",
        # "alpaca_questions",
    ]

    for dataset_path, output_label in zip(dataset_paths, output_labels):
        # Run the complete pipeline with all options explicit
        results = await run_misalignment_pipeline(
            # === REQUIRED PARAMETERS ===
            baseline_model_path=baseline_model_path,
            target_model_path=target_model_path,
            dataset_path=dataset_path,
            layers=[3, 7, 11, 15, 19, 23, 27],
            activation_strategy="prompt_t1",
            output_dir=output_dir,
            output_label=output_label,
            
            # === FEATURE ANALYSIS SETTINGS ===
            top_k_features=100,
            change_direction="positive_diff",
            
            # === ANALYSIS COMPONENTS ===
            run_autointerp=True,
            run_steering=True,
            
            # === PERFORMANCE SETTINGS ===
            max_concurrent_claude=32,
            max_prompts=2048,
            batch_size=8,
            ctx_len=512,
            
            # === STEERING EVALUATION ===
            steering_coefficients=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            safe_threshold=0.90,      # Minimum P(A) + P(B) for valid steering
            
            # === PATH CONFIGURATION ===
            sae_path_template=sae_path_template,
            feature_mining_path_template=feature_mining_path_template,
            
            # === CLAUDE MODEL ===
            claude_model="claude-3-7-sonnet-20250219",
            
            # === CACHE DIRECTORY ===
            cache_dir="/root/git/model-diffing-em/.cache",  # Directory for caching Claude responses
        )
        
        # Print results summary
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print(f"✅ Processed {len(results)} layers")

if __name__ == "__main__":
    asyncio.run(main()) 