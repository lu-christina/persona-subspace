#!/usr/bin/env python3

import pandas as pd
import sys

def populate_claude_fields():
    """
    Populate claude_ fields in personal_features_general_prompts.csv 
    from only_personal_autointerp.csv based on matching feature_id and source.
    """
    
    # Load the files
    target_path = "./results/personal_general/personal_features_general_prompts.csv"
    source_path = "./results/personal/only_personal_autointerp.csv"
    
    try:
        target_df = pd.read_csv(target_path)
        print(f"Loaded target file: {len(target_df)} rows")
        print(f"Target columns: {list(target_df.columns)}")
    except FileNotFoundError:
        print(f"Error: Target file not found at {target_path}")
        return
    
    try:
        source_df = pd.read_csv(source_path)
        print(f"Loaded source file: {len(source_df)} rows")
        print(f"Source columns: {list(source_df.columns)}")
    except FileNotFoundError:
        print(f"Error: Source file not found at {source_path}")
        return
    
    # Check if Claude columns already exist in target
    claude_columns = ['claude_completion', 'claude_desc', 'claude_type']
    existing_claude_cols = [col for col in claude_columns if col in target_df.columns]
    
    if existing_claude_cols:
        print(f"Claude columns already exist in target: {existing_claude_cols}")
        print("Will overwrite existing data.")
    
    # Add missing Claude columns to target dataframe
    for col in claude_columns:
        if col not in target_df.columns:
            target_df[col] = ''
    
    # Create lookup dictionary from source data
    # Key: (feature_id, source), Value: dict of claude fields
    lookup = {}
    for _, row in source_df.iterrows():
        key = (row['feature_id'], row['source'])
        lookup[key] = {
            'claude_completion': row.get('claude_completion', ''),
            'claude_desc': row.get('claude_desc', ''),
            'claude_type': row.get('claude_type', '')
        }
    
    print(f"Created lookup with {len(lookup)} entries")
    
    # Update target dataframe
    matches_found = 0
    for idx, row in target_df.iterrows():
        key = (row['feature_id'], row['source'])
        if key in lookup:
            for claude_col in claude_columns:
                target_df.at[idx, claude_col] = lookup[key][claude_col]
            matches_found += 1
    
    print(f"Found matches for {matches_found}/{len(target_df)} rows")
    
    # Check for unmatched rows
    unmatched = len(target_df) - matches_found
    if unmatched > 0:
        print(f"Warning: {unmatched} rows in target had no matching source data")
        
        # Show some examples of unmatched keys
        print("Sample unmatched (feature_id, source) pairs:")
        unmatched_count = 0
        for _, row in target_df.iterrows():
            key = (row['feature_id'], row['source'])
            if key not in lookup and unmatched_count < 5:
                print(f"  {key}")
                unmatched_count += 1
    
    # Save the updated file
    target_df.to_csv(target_path, index=False)
    print(f"Updated file saved to {target_path}")
    
    # Summary statistics
    non_empty_claude_desc = target_df['claude_desc'].str.strip().astype(bool).sum()
    non_empty_claude_type = target_df['claude_type'].str.strip().astype(bool).sum()
    
    print(f"\nSummary:")
    print(f"- Total rows: {len(target_df)}")
    print(f"- Rows with claude_desc: {non_empty_claude_desc}")
    print(f"- Rows with claude_type: {non_empty_claude_type}")
    
    if non_empty_claude_type > 0:
        print(f"- Claude type distribution:")
        print(target_df['claude_type'].value_counts())

if __name__ == "__main__":
    populate_claude_fields()