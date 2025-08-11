#!/usr/bin/env python3
"""
fix_roles_240_defaults.py - Fix missing default_1 activation in roles_240

This script processes the 0_default.pt and 1_default.pt files to calculate the missing
default_1 activation, then updates the existing default_activations.pt file with:
- New default_1 activation (mean of all 2400 samples from both default files)  
- Updated all_1 activation (weighted mean including the new default samples)
- Updated sample counts in metadata

The script preserves the existing pos_1 activation and its sample count.
"""

import os
import shutil
import torch
from pathlib import Path
from typing import Dict, Tuple


def load_default_activations() -> Tuple[Dict[str, torch.Tensor], int]:
    """Load all activations from 0_default.pt and 1_default.pt files."""
    activations_dir = Path("/workspace/roles_240/response_activations")
    
    all_activations = []
    total_samples = 0
    
    for default_file in ["0_default.pt", "1_default.pt"]:
        file_path = activations_dir / default_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Default file not found: {file_path}")
            
        print(f"Loading {default_file}...")
        data = torch.load(file_path, map_location='cpu')
        
        # Add all activations from this file
        for key, activation in data.items():
            all_activations.append(activation)
            total_samples += 1
            
        print(f"  Loaded {len(data)} samples from {default_file}")
    
    return all_activations, total_samples


def calculate_default_mean(all_activations) -> torch.Tensor:
    """Calculate mean activation across all default samples."""
    print(f"Calculating mean across {len(all_activations)} samples...")
    
    # Stack all activations and calculate mean
    stacked = torch.stack(all_activations)
    mean_activation = stacked.mean(dim=0)
    
    print(f"Default mean activation shape: {mean_activation.shape}")
    return mean_activation


def update_results_file(default_1: torch.Tensor, default_1_count: int) -> None:
    """Update the existing default_activations.pt file with new data."""
    results_path = Path("/workspace/roles_240/default_activations.pt")
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    # Create backup
    backup_path = results_path.with_suffix(".pt.backup")
    print(f"Creating backup: {backup_path}")
    shutil.copy2(results_path, backup_path)
    
    # Load existing results
    print("Loading existing results...")
    data = torch.load(results_path, map_location='cpu')
    
    # Validate structure
    if 'activations' not in data or 'metadata' not in data:
        raise ValueError("Invalid results file structure")
    
    activations = data['activations']
    metadata = data['metadata']
    
    # Get current values
    current_all_1 = activations['all_1']
    current_all_1_count = metadata['counts']['all_1']
    current_pos_1_count = metadata['counts']['pos_1']
    
    print(f"Current all_1 count: {current_all_1_count}")
    print(f"Adding default_1 count: {default_1_count}")
    
    # Add default_1
    activations['default_1'] = default_1
    
    # Calculate weighted mean for new all_1
    new_all_1_count = current_all_1_count + default_1_count
    new_all_1 = (current_all_1 * current_all_1_count + default_1 * default_1_count) / new_all_1_count
    activations['all_1'] = new_all_1
    
    # Update counts
    metadata['counts']['default_1'] = default_1_count
    metadata['counts']['all_1'] = new_all_1_count
    
    print(f"Updated all_1 count: {new_all_1_count}")
    
    # Save updated results
    print("Saving updated results...")
    torch.save(data, results_path)
    
    # Verify the save was successful
    verify_data = torch.load(results_path, map_location='cpu')
    expected_keys = {'pos_1', 'default_1', 'all_1'}
    actual_keys = set(verify_data['activations'].keys())
    
    if actual_keys == expected_keys:
        print("✓ Successfully updated results file")
        print(f"✓ All expected keys present: {sorted(actual_keys)}")
    else:
        raise ValueError(f"Verification failed. Expected keys: {expected_keys}, Got: {actual_keys}")


def print_summary(default_1_count: int) -> None:
    """Print summary of the updated results."""
    results_path = Path("/workspace/roles_240/default_activations.pt")
    data = torch.load(results_path, map_location='cpu')
    
    print("\n" + "="*50)
    print("SUMMARY OF UPDATED RESULTS")
    print("="*50)
    
    activations = data['activations']
    counts = data['metadata']['counts']
    
    for key in sorted(activations.keys()):
        shape = activations[key].shape
        count = counts[key]
        print(f"{key:10s}: {count:6d} samples, shape {shape}")
    
    total_files = data['metadata']['total_files_processed']
    print(f"\nTotal files processed: {total_files['labels']} labels, {total_files['activations']} activations")
    print("✓ Fix complete - default_1 activation successfully added")


def main():
    """Main function."""
    print("=" * 60)
    print("FIXING ROLES_240 DEFAULT ACTIVATIONS")
    print("=" * 60)
    
    try:
        # Step 1: Load default activations
        print("\n1. Loading default activation files...")
        all_activations, total_samples = load_default_activations()
        
        # Step 2: Calculate default_1 mean
        print(f"\n2. Calculating default_1 activation...")
        default_1 = calculate_default_mean(all_activations)
        
        # Step 3: Update results file
        print(f"\n3. Updating results file...")
        update_results_file(default_1, total_samples)
        
        # Step 4: Print summary
        print_summary(total_samples)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())