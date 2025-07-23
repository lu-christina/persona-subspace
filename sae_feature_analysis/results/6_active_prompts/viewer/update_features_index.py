#!/usr/bin/env python3
"""
Update features_index.json by scanning the features directory for new or updated feature files.

This script:
1. Scans the features/ directory for all .json files
2. Updates the features_index.json with current file sizes
3. Maintains metadata and sorts features by ID

Usage:
    python update_features_index.py
"""

import json
import os
from pathlib import Path
import sys

def update_features_index():
    """Update the features index by scanning the features directory."""
    
    script_dir = Path(__file__).parent
    features_dir = script_dir / "../gemma_trainer131k-l0-114_layer20/1000_prompts/features"
    features_index_path = script_dir / "features_index.json"
    
    # Check if features directory exists
    if not features_dir.exists():
        print(f"Error: Features directory not found: {features_dir}")
        return False
    
    # Load existing index if it exists
    if features_index_path.exists():
        print(f"Loading existing features index from {features_index_path}")
        with open(features_index_path, 'r', encoding='utf-8') as f:
            features_index = json.load(f)
    else:
        print("Creating new features index")
        features_index = {
            'features': [],
            'metadata': {
                'total_prompts': 1000,
                'total_features': 0,
                'generated_by': 'update_features_index.py',
                'structure_version': '2.0'
            }
        }
    
    # Scan features directory for .json files
    feature_files = list(features_dir.glob("*.json"))
    print(f"Found {len(feature_files)} feature files in {features_dir}")
    
    if not feature_files:
        print("No feature files found!")
        return False
    
    # Build new features list
    new_features = []
    total_size = 0
    
    for feature_file in feature_files:
        try:
            feature_id = feature_file.stem  # filename without .json extension
            
            # Validate that feature_id is numeric
            int(feature_id)
            
            feature_size = feature_file.stat().st_size
            total_size += feature_size
            
            new_features.append({
                'id': feature_id,
                'size': feature_size
            })
            
            print(f"  Feature {feature_id}: {feature_size / 1024:.1f} KB")
            
        except ValueError:
            print(f"  Skipping non-numeric feature file: {feature_file.name}")
        except Exception as e:
            print(f"  Error processing {feature_file.name}: {e}")
    
    # Sort features by numeric ID
    new_features.sort(key=lambda x: int(x['id']))
    
    # Update index
    old_count = features_index['metadata']['total_features']
    features_index['features'] = new_features
    features_index['metadata']['total_features'] = len(new_features)
    features_index['metadata']['generated_by'] = 'update_features_index.py'
    
    # Save updated index
    print(f"\nSaving updated features index to {features_index_path}")
    with open(features_index_path, 'w', encoding='utf-8') as f:
        json.dump(features_index, f, separators=(',', ':'))  # Compact format
    
    # Print summary
    print(f"\n✓ Features index updated successfully!")
    print(f"Features: {old_count} → {len(new_features)} ({'+'}{len(new_features) - old_count} new)" if old_count != len(new_features) else f"Features: {len(new_features)} (no change)")
    print(f"Total features size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Average feature size: {total_size / len(new_features) / 1024:.1f} KB")
    print(f"Index file size: {features_index_path.stat().st_size / 1024:.1f} KB")
    
    return True

def main():
    """Main entry point."""
    print("Updating features index...")
    
    try:
        success = update_features_index()
        if success:
            print("\n✓ Done!")
        else:
            print("\n✗ Failed to update features index")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()