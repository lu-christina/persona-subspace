#!/usr/bin/env python3
"""
Script to update the known feature list in viewer.js by scanning the directory structure.
This ensures the viewer always shows the correct available features.
"""

import os
import re
from pathlib import Path

def get_available_features():
    """Scan the directory structure to find available features."""
    features_dir = Path("../gemma_trainer131k-l0-114_layer20/1000_prompts")
    
    if not features_dir.exists():
        print(f"Error: Directory {features_dir} does not exist")
        return []
    
    features = []
    for item in features_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            # Check if it has at least the active.jsonl file
            if (item / "active.jsonl").exists():
                features.append(item.name)
    
    # Sort numerically
    features.sort(key=int)
    return features

def update_viewer_js(features):
    """Update the viewer.js file with the new feature list."""
    viewer_js_path = Path("viewer.js")
    
    if not viewer_js_path.exists():
        print(f"Error: {viewer_js_path} does not exist")
        return False
    
    # Read the current file
    with open(viewer_js_path, 'r') as f:
        content = f.read()
    
    # Create the new feature list string
    feature_list = "[\n            " + ", ".join(f"'{f}'" for f in features) + "\n        ]"
    
    # Pattern to match the current feature list
    pattern = r'(async discoverFeatures\(\) \{[^}]*this\.features = )\[[^\]]*\]'
    
    # Replace the feature list
    new_content = re.sub(
        pattern,
        f'\\1{feature_list}',
        content,
        flags=re.DOTALL
    )
    
    if new_content == content:
        print("Warning: No changes made to viewer.js - pattern might not match")
        return False
    
    # Write the updated content
    with open(viewer_js_path, 'w') as f:
        f.write(new_content)
    
    return True

def main():
    print("Scanning for available features...")
    features = get_available_features()
    
    if not features:
        print("No features found!")
        return
    
    print(f"Found {len(features)} features: {', '.join(features)}")
    
    print("Updating viewer.js...")
    if update_viewer_js(features):
        print("Successfully updated viewer.js!")
    else:
        print("Failed to update viewer.js")

if __name__ == "__main__":
    main()