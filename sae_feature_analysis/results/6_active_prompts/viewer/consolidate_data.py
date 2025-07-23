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
    
    # Storage for separated data
    prompts = {}  # prompt_id -> prompt metadata  
    features = {}  # feature_id -> {active: [...], inactive: [...]}
    
    feature_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    total_features = len(feature_dirs)
    
    print(f"Processing {total_features} features...")
    print("Will generate features.json and prompts.json separately")
    
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
                    # Store only tokenized prompt for highlighting
                    tokenized = entry.get('tokenized_prompt', [])
                    prompts[prompt_id] = {
                        'tokens': tokenized,
                        'len': len(tokenized)
                    }
                
                # Get max activation for this feature
                max_activation = 0
                if 'max_feature_activations' in entry and feature_id in entry['max_feature_activations']:
                    max_activation = entry['max_feature_activations'][feature_id]
                elif 'max_feature_activation' in entry:
                    max_activation = entry['max_feature_activation']
                
                # For inactive prompts, only store if activation is exactly 0 (true inactive)
                # For active prompts, filter out very low activations  
                active_type = 'active' if file_name.startswith('active') else 'inactive'
                
                if active_type == 'active' and max_activation <= 0.1:
                    continue
                elif active_type == 'inactive' and max_activation != 0.0:
                    continue
                
                # Create optimized activation entry (minimal data)
                activation_entry = {
                    'id': prompt_id  # Just the prompt ID
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
                        
                        if token_activation > 0:  # Save all active tokens
                            token_activations.append({
                                'pos': token['position'],
                                'act': round(token_activation, 3)
                            })
                    
                    # Store all active tokens (sorted by activation)
                    if token_activations:
                        token_activations.sort(key=lambda x: x['act'], reverse=True)
                        activation_entry['tokens'] = token_activations
                
                # Store in appropriate category
                active_type = 'active' if file_name.startswith('active') else 'inactive'
                feature_data[active_type][token_type].append(activation_entry)
        
        features[feature_id] = feature_data
    
    # Return separated data structures
    return prompts, features, {
        'total_prompts': len(prompts),
        'total_features': len(features),
        'generated_by': 'consolidate_data.py',
        'structure_version': '2.0'
    }

def main():
    script_dir = Path(__file__).parent
    
    print("Starting data consolidation...")
    
    try:
        # Consolidate the data
        prompts, features, metadata = consolidate_data(script_dir)
        
        # Write prompts JSON
        prompts_path = script_dir / "prompts.json"
        print(f"Writing prompts data to {prompts_path}")
        
        with open(prompts_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, separators=(',', ':'))  # Compact format
        
        # Create features directory
        features_dir = script_dir / "features"
        features_dir.mkdir(exist_ok=True)
        
        # Write individual feature files
        feature_list = []
        total_features_size = 0
        
        print(f"Writing individual feature files to {features_dir}/")
        for feature_id, feature_data in features.items():
            feature_path = features_dir / f"{feature_id}.json"
            
            with open(feature_path, 'w', encoding='utf-8') as f:
                json.dump(feature_data, f, separators=(',', ':'))  # Compact format
            
            feature_size = feature_path.stat().st_size
            total_features_size += feature_size
            feature_list.append({
                'id': feature_id,
                'size': feature_size
            })
            
            print(f"  Feature {feature_id}: {feature_size / 1024:.1f} KB")
        
        # Write features index file
        features_index_path = script_dir / "features_index.json"
        print(f"Writing features index to {features_index_path}")
        
        features_index = {
            'features': sorted(feature_list, key=lambda x: int(x['id'])),
            'metadata': metadata
        }
        
        with open(features_index_path, 'w', encoding='utf-8') as f:
            json.dump(features_index, f, separators=(',', ':'))  # Compact format
        
        # Print statistics
        prompts_size = prompts_path.stat().st_size
        index_size = features_index_path.stat().st_size
        total_size = prompts_size + total_features_size + index_size
        
        print(f"\nConsolidation complete!")
        print(f"Prompts file: {prompts_path} ({prompts_size / 1024 / 1024:.1f} MB)")
        print(f"Features index: {features_index_path} ({index_size / 1024:.1f} KB)")
        print(f"Individual features: {len(feature_list)} files ({total_features_size / 1024 / 1024:.1f} MB total)")
        print(f"Average feature size: {total_features_size / len(feature_list) / 1024:.1f} KB")
        print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
        print(f"Total prompts: {metadata['total_prompts']}")
        print(f"Total features: {metadata['total_features']}")
        
        # Test loading to verify JSON is valid
        print("Verifying JSON integrity...")
        with open(prompts_path, 'r', encoding='utf-8') as f:
            test_prompts = json.load(f)
        with open(features_index_path, 'r', encoding='utf-8') as f:
            test_index = json.load(f)
        # Test a random feature file
        if feature_list:
            test_feature_path = features_dir / f"{feature_list[0]['id']}.json"
            with open(test_feature_path, 'r', encoding='utf-8') as f:
                test_feature = json.load(f)
        print("✓ All JSON files are valid and loadable")
        
    except Exception as e:
        print(f"Error during consolidation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()