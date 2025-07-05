#!/usr/bin/env python3
"""
Script to run universal feature analysis across all model combinations.
"""

import os
import pandas as pd
import torch
from universal_features_lib import UniversalFeatureAnalyzer, load_model_components, load_sae, get_model_combinations


def main():
    """Run universal feature analysis for all model combinations."""
    # Configuration
    prompts_path = "./prompts"
    activation_threshold = 0.01
    prompt_threshold = 0.3  # 30% of prompts
    output_file = "./results/universal/universal_30.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all combinations
    combinations = get_model_combinations()
    print(f"Running analysis for {len(combinations)} combinations...")
    
    # Organize combinations by model type for efficient loading
    combinations_by_model = {}
    for model_type, layer, token_type in combinations:
        if model_type not in combinations_by_model:
            combinations_by_model[model_type] = {}
        if layer not in combinations_by_model[model_type]:
            combinations_by_model[model_type][layer] = []
        combinations_by_model[model_type][layer].append(token_type)
    
    # Collect all results
    all_results = []
    
    # Process each model type
    for model_type in combinations_by_model:
        print(f"\n{'='*60}")
        print(f"LOADING MODEL: {model_type}")
        print(f"{'='*60}")
        
        # Load model components once for this model type
        try:
            tokenizer, model = load_model_components(model_type)
            print(f"✓ Loaded {model_type} model")
        except Exception as e:
            print(f"❌ Failed to load {model_type} model: {str(e)}")
            continue
        
        try:
            # Process each layer for this model
            for layer in combinations_by_model[model_type]:
                print(f"\n{'='*40}")
                print(f"LOADING SAE: {model_type} layer {layer}")
                print(f"{'='*40}")
                
                # Load SAE once for this layer
                try:
                    sae = load_sae(model_type, layer)
                    print(f"✓ Loaded {model_type} layer {layer} SAE")
                except Exception as e:
                    print(f"❌ Failed to load {model_type} layer {layer} SAE: {str(e)}")
                    continue
                
                try:
                    # Process all token types for this model+layer combination
                    for token_type in combinations_by_model[model_type][layer]:
                        print(f"\n--- Processing {model_type} layer {layer} {token_type} ---")
                        
                        try:
                            # Create analyzer
                            analyzer = UniversalFeatureAnalyzer(
                                model_type=model_type,
                                sae_layer=layer,
                                token_type=token_type
                            )
                            
                            # Run analysis
                            results_df = analyzer.analyze_universal_features(
                                tokenizer=tokenizer,
                                model=model,
                                sae=sae,
                                prompts_path=prompts_path,
                                activation_threshold=activation_threshold,
                                prompt_threshold=prompt_threshold
                            )
                            
                            # Add to combined results
                            if len(results_df) > 0:
                                all_results.append(results_df)
                                print(f"✓ Found {len(results_df)} universal features")
                            else:
                                print("✓ No universal features found")
                                
                        except Exception as e:
                            print(f"❌ Error processing {model_type} layer {layer} {token_type}: {str(e)}")
                            continue
                
                finally:
                    # Clean up SAE
                    if 'sae' in locals():
                        del sae
                        torch.cuda.empty_cache()
                        print(f"✓ Cleaned up {model_type} layer {layer} SAE")
        
        finally:
            # Clean up model
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            torch.cuda.empty_cache()
            print(f"✓ Cleaned up {model_type} model")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by source first, then by activation_mean (descending) within each source
        combined_df = combined_df.sort_values(['source', 'activation_mean'], ascending=[True, False])
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total universal features found: {len(combined_df)}")
        print(f"Results saved to: {output_file}")
        
        # Show breakdown by source
        print(f"\nBreakdown by source:")
        source_counts = combined_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} features")
            
        # Show top 10 features
        print(f"\nTop 10 universal features:")
        print(combined_df[['feature_id', 'activation_mean', 'num_prompts', 'source']].head(10).to_string(index=False))
        
    else:
        print("\n❌ No universal features found across any combinations")
        # Create empty file with headers
        empty_df = pd.DataFrame(columns=[
            'source', 'feature_id', 'activation_mean', 'activation_max', 'activation_min', 
            'num_prompts', 'chat_desc', 'pt_desc', 'type', 'link'
        ])
        empty_df.to_csv(output_file, index=False)
        print(f"Empty results file created: {output_file}")


if __name__ == "__main__":
    main()