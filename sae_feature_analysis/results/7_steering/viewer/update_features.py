#!/usr/bin/env python3
"""
Script to update the known feature/group list in viewer.js by scanning the directory structure.
This ensures the viewer always shows the correct available features and groups.
"""

import os
import re
import json
from pathlib import Path

def get_available_files():
    """Scan the directory structure to find available feature and group files."""
    data_dir = Path("../gemma_trainer131k-l0-114_layer20")
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return [], []
    
    individual_features = []
    group_files = []
    
    for item in data_dir.iterdir():
        if item.is_file() and item.suffix == '.json':
            filename = item.name
            stem = item.stem
            
            # Check if it's a numeric feature (individual feature)
            if stem.isdigit():
                individual_features.append(stem)
            else:
                # It's a group file - try to read it to get the readable name
                try:
                    with open(item, 'r') as f:
                        data = json.load(f)
                        readable_name = data.get('readable_group_name', stem)
                        group_files.append({
                            'filename': filename,
                            'stem': stem,
                            'readable_name': readable_name
                        })
                except Exception as e:
                    print(f"Warning: Could not read {filename}: {e}")
                    # Add it anyway with the stem as readable name
                    group_files.append({
                        'filename': filename,
                        'stem': stem,
                        'readable_name': stem
                    })
    
    # Sort individual features numerically
    individual_features.sort(key=int)
    
    # Sort group files by readable name
    group_files.sort(key=lambda x: x['readable_name'])
    
    return individual_features, group_files

def update_viewer_js(individual_features, group_files):
    """Update the viewer.js file with the new feature and group lists."""
    viewer_js_path = Path("viewer.js")
    
    if not viewer_js_path.exists():
        print(f"Error: {viewer_js_path} does not exist")
        return False
    
    # Read the current file
    with open(viewer_js_path, 'r') as f:
        content = f.read()
    
    # Create the individual features array
    individual_features_code = "[\n                " + ", ".join(f"'{f}'" for f in individual_features) + "\n            ]"
    
    # Create the group files array
    group_files_code = "[\n                " + ", ".join(f"'{gf['filename']}'" for gf in group_files) + "\n            ]"
    
    # Pattern to match the knownFeatures array
    known_features_pattern = r'(const knownFeatures = )\[[^\]]*\]'
    
    # Pattern to match the knownGroupFiles array  
    known_group_files_pattern = r'(const knownGroupFiles = )\[[^\]]*\]'
    
    # Replace the knownFeatures array
    new_content = re.sub(
        known_features_pattern,
        f'\\1{individual_features_code}',
        content,
        flags=re.DOTALL
    )
    
    # Replace the knownGroupFiles array
    new_content = re.sub(
        known_group_files_pattern,
        f'\\1{group_files_code}',
        new_content,
        flags=re.DOTALL
    )
    
    if new_content == content:
        print("Warning: No changes made to viewer.js - patterns might not match")
        return False
    
    # Write the updated content
    with open(viewer_js_path, 'w') as f:
        f.write(new_content)
    
    return True

def main():
    print("Scanning for available features and groups...")
    individual_features, group_files = get_available_files()
    
    if not individual_features and not group_files:
        print("No features or groups found!")
        return
    
    print(f"Found {len(individual_features)} individual features: {', '.join(individual_features)}")
    print(f"Found {len(group_files)} group files:")
    for gf in group_files:
        print(f"  - {gf['filename']} ({gf['readable_name']})")
    
    print("Updating viewer.js...")
    if update_viewer_js(individual_features, group_files):
        print("Successfully updated viewer.js!")
    else:
        print("Failed to update viewer.js")

if __name__ == "__main__":
    main()