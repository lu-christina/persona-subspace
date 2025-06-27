"""
analyze_discovered_features.py
------------------------------
Analyze the top features discovered in get_activations.ipynb by finding
top-activating text examples for each feature.

Usage:
    python analyze_discovered_features.py --top_features "87027,45123,12345" --num_samples 50000
"""

import argparse
import pathlib
import numpy as np
from feature_mining import FeatureMiningConfig, mine_top_features, load_and_display_results, simple_text_generator
from transformers import AutoTokenizer


def get_top_features_from_notebook():
    """
    Extract the top feature indices from notebook results.
    You can update this manually with your results from get_activations.ipynb
    """
    # TODO: Update these with your actual top features from the notebook
    # These are example feature indices - replace with your actual results
    top_features = [
        87027,  # Example: top feature from notebook
        45123,  # Example: second feature
        12345,  # Example: third feature
        67890,  # etc...
        11111,
    ]
    return top_features


def analyze_features(feature_indices, num_samples=50000, top_k=20, batch_size=4):
    """Analyze specific features by finding top-activating text examples."""
    
    print(f"Analyzing {len(feature_indices)} features: {feature_indices}")
    
    # Create config
    config = FeatureMiningConfig(
        num_samples=num_samples,
        top_k=top_k,
        batch_size=batch_size,
        out_dir=pathlib.Path("/workspace/feature_analysis")
    )
    
    # Create text generator - using larger dataset for better examples
    print("Setting up text generator...")
    text_gen = large_text_generator(num_samples)
    
    # Mine features
    out_path = config.out_dir / f"discovered_features_top{len(feature_indices)}.h5"
    print(f"Mining top-activating examples to {out_path}...")
    
    mine_top_features(text_gen, out_path, config, specific_features=feature_indices)
    
    # Display results for each feature
    print("\n" + "="*80)
    print("FEATURE ANALYSIS RESULTS")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for i, feature_id in enumerate(feature_indices):
        print(f"\n{'='*80}")
        print(f"FEATURE {feature_id} (Index {i} in results)")
        print(f"{'='*80}")
        
        load_and_display_results(out_path, tokenizer, feature_idx=i, top_n=5)
        
        # Optional: pause between features for readability
        if i < len(feature_indices) - 1:
            input("\nPress Enter to see next feature...")
    
    return out_path


def large_text_generator(num_samples=50000):
    """Generator for larger, more diverse text dataset."""
    from datasets import load_dataset
    import random
    
    # Use multiple datasets for more diverse examples
    datasets = [
        ("wikitext", "wikitext-103-raw-v1", "train"),
        ("openwebtext", None, "train"),  # If available
    ]
    
    # Try to load datasets, fallback if not available
    working_datasets = []
    for ds_name, ds_config, split in datasets:
        try:
            if ds_config:
                dataset = load_dataset(ds_name, ds_config, split=split, streaming=True)
            else:
                dataset = load_dataset(ds_name, split=split, streaming=True)
            dataset = dataset.shuffle(buffer_size=10000, seed=42)
            working_datasets.append((dataset, ds_name))
            print(f"Loaded dataset: {ds_name}")
        except Exception as e:
            print(f"Could not load {ds_name}: {e}")
    
    # If no datasets work, fallback to simple generator
    if not working_datasets:
        print("Falling back to simple text generator...")
        return simple_text_generator(num_samples)
    
    # Generator that alternates between datasets
    count = 0
    dataset_iterators = [(iter(ds), name) for ds, name in working_datasets]
    
    while count < num_samples and dataset_iterators:
        # Pick random dataset
        ds_iter, ds_name = random.choice(dataset_iterators)
        
        try:
            example = next(ds_iter)
            text_field = "text" if "text" in example else list(example.keys())[0]
            text = example[text_field].strip()
            
            # Filter for decent length texts
            if len(text) > 100:
                yield text
                count += 1
                
        except StopIteration:
            # Remove exhausted dataset
            dataset_iterators = [(ds_iter, name) for ds_iter, name in dataset_iterators if (ds_iter, name) != (ds_iter, ds_name)]
            if not dataset_iterators:
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze discovered SAE features")
    parser.add_argument("--top_features", type=str, 
                       help="Comma-separated list of feature indices to analyze")
    parser.add_argument("--num_samples", type=int, default=50000,
                       help="Number of text samples to process")
    parser.add_argument("--top_k", type=int, default=20,
                       help="Number of top examples to keep per feature")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--use_notebook_features", action="store_true",
                       help="Use features hardcoded from notebook results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Get feature indices to analyze
    if args.top_features:
        feature_indices = [int(x.strip()) for x in args.top_features.split(",")]
    elif args.use_notebook_features:
        feature_indices = get_top_features_from_notebook()
    else:
        print("Error: Must specify either --top_features or --use_notebook_features")
        exit(1)
    
    # Run analysis
    result_path = analyze_features(
        feature_indices=feature_indices,
        num_samples=args.num_samples,
        top_k=args.top_k,
        batch_size=args.batch_size
    )
    
    print(f"\nAnalysis complete! Results saved to: {result_path}")
    print(f"You can reload and explore the results using:")
    print(f"  from feature_mining import load_and_display_results")
    print(f"  load_and_display_results('{result_path}', tokenizer, feature_idx=0)")