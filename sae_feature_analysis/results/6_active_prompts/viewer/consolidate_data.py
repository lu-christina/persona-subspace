#!/usr/bin/env python3
"""
Consolidate SAE feature activation data from individual JSONL files into 
an optimized single JSON bundle for web viewer performance.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import sys

def load_jsonl(file_path):
    """Load and parse a JSONL file, handling malformed JSON gracefully."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON in {file_path} line {line_num}: {e}")
    return data

def consolidate_data(base_dir):
    """Consolidate all feature data into an optimized structure."""
    data_dir = Path("/root/git/persona-subspace/sae_feature_analysis/results/6_active_prompts/gemma_trainer131k-l0-114_layer20/1000_prompts")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Storage for consolidated data
    prompts = {}  # prompt_id -> prompt metadata
    features = {}  # feature_id -> {active: [...], inactive: [...]}
    
    feature_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    total_features = len(feature_dirs)
    
    print(f"Processing {total_features} features...")
    
    for i, feature_dir in enumerate(sorted(feature_dirs, key=lambda x: int(x.name))):
        feature_id = feature_dir.name
        print(f"Processing feature {feature_id} ({i+1}/{total_features})")
        
        feature_data = {
            'active': {'all': [], 'model': [], 'newline': []},
            'inactive': {'all': [], 'model': [], 'newline': []}
        }
        
        # Process each file type
        file_types = [
            ('active', 'all'), ('active_model', 'model'), ('active_newline', 'newline'),
            ('inactive', 'all'), ('inactive_model', 'model'), ('inactive_newline', 'newline')
        ]
        
        for file_name, token_type in file_types:
            file_path = feature_dir / f"{file_name}.jsonl"
            if not file_path.exists():
                print(f"Warning: Missing file {file_path}")
                continue
                
            data = load_jsonl(file_path)
            
            for entry in data:
                # Extract and store prompt metadata (deduplicated)
                prompt_id = entry['prompt_id']
                if prompt_id not in prompts:
                    # Store tokenized prompt for highlighting, but compress it
                    tokenized = entry.get('tokenized_prompt', [])
                    prompts[prompt_id] = {
                        'text': entry['prompt_text'][:500],  # Truncate very long prompts
                        'tokens': tokenized,
                        'len': len(tokenized)
                    }
                
                # Get max activation for this feature
                max_activation = 0
                if 'max_feature_activations' in entry and feature_id in entry['max_feature_activations']:
                    max_activation = entry['max_feature_activations'][feature_id]
                elif 'max_feature_activation' in entry:
                    max_activation = entry['max_feature_activation']
                
                # Only store prompts with significant activation (> 0.1) to reduce size
                if max_activation <= 0.1:
                    continue
                
                # Create optimized activation entry (minimal data)
                activation_entry = {
                    'id': prompt_id,  # Shorter key name
                    'act': round(max_activation, 3)  # Rounded to 3 decimals
                }
                
                # Include only the most activated tokens (top 5) to reduce size dramatically
                if 'tokens' in entry and entry['tokens']:
                    token_activations = []
                    for token in entry['tokens']:
                        token_activation = 0
                        # Handle both data formats
                        if 'feature_activations' in token and feature_id in token['feature_activations']:
                            token_activation = token['feature_activations'][feature_id]
                        elif 'feature_activation' in token:
                            token_activation = token['feature_activation']
                        
                        if token_activation > 0.5:  # Higher threshold
                            token_activations.append({
                                'pos': token['position'],
                                'act': round(token_activation, 3)
                            })
                    
                    # Sort by activation and keep only top 5
                    if token_activations:
                        token_activations.sort(key=lambda x: x['act'], reverse=True)
                        activation_entry['tokens'] = token_activations[:5]
                
                # Store in appropriate category
                active_type = 'active' if file_name.startswith('active') else 'inactive'
                feature_data[active_type][token_type].append(activation_entry)
        
        features[feature_id] = feature_data
    
    # Create final consolidated structure
    consolidated = {
        'prompts': prompts,
        'features': features,
        'metadata': {
            'total_prompts': len(prompts),
            'total_features': len(features),
            'generated_by': 'consolidate_data.py',
            'structure_version': '1.0'
        }
    }
    
    return consolidated

def main():
    script_dir = Path(__file__).parent
    
    print("Starting data consolidation...")
    
    try:
        # Consolidate the data
        consolidated_data = consolidate_data(script_dir)
        
        # Write consolidated JSON
        output_path = script_dir / "consolidated_features.json"
        print(f"Writing consolidated data to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, separators=(',', ':'))  # Compact format
        
        # Print statistics
        file_size = output_path.stat().st_size
        print(f"\nConsolidation complete!")
        print(f"Output file: {output_path}")
        print(f"File size: {file_size / 1024 / 1024:.1f} MB")
        print(f"Total prompts: {consolidated_data['metadata']['total_prompts']}")
        print(f"Total features: {consolidated_data['metadata']['total_features']}")
        
        # Test loading to verify JSON is valid
        print("Verifying JSON integrity...")
        with open(output_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print("✓ JSON file is valid and loadable")
        
    except Exception as e:
        print(f"Error during consolidation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()