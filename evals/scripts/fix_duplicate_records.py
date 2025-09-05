#!/usr/bin/env python3
"""
Fix Duplicate Records Script

This script processes JSONL files to:
1. Add sample_id: 0 to records that don't have a sample_id field
2. Remove duplicate records with the same (id, magnitude, sample_id) combination
3. Keep only the first occurrence of each unique combination

Usage: 
  python fix_duplicate_records.py [directory_path]          # Process all JSONL files in directory
  python fix_duplicate_records.py file1.jsonl file2.jsonl  # Process specific files
Default directory: ./susceptibility/
"""

import json
import os
import sys
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Set, List, Tuple


def process_jsonl_file(file_path: str) -> Dict[str, int]:
    """
    Process a single JSONL file to fix sample_id fields and remove duplicates.
    
    Returns:
        Dict with statistics: {'original': int, 'duplicates_removed': int, 'final': int}
    """
    print(f"Processing {file_path}...")
    
    # Create backup
    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Read all records
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    original_count = len(records)
    print(f"  Original records: {original_count}")
    
    # Process records: add missing sample_id and deduplicate
    seen_combinations: Set[Tuple[int, float, int]] = set()
    unique_records = []
    duplicates_removed = 0
    
    for record in records:
        # Add sample_id: 0 if missing
        if 'sample_id' not in record:
            record['sample_id'] = 0
        
        # Create unique key
        try:
            unique_key = (
                record['id'], 
                float(record['magnitude']), 
                record['sample_id']
            )
        except (KeyError, TypeError, ValueError) as e:
            print(f"  Warning: Skipping record with invalid key fields: {e}")
            print(f"  Record: {record}")
            continue
        
        # Check for duplicates
        if unique_key in seen_combinations:
            duplicates_removed += 1
        else:
            seen_combinations.add(unique_key)
            unique_records.append(record)
    
    final_count = len(unique_records)
    
    # Write cleaned records back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in unique_records:
            f.write(json.dumps(record) + '\n')
    
    stats = {
        'original': original_count,
        'duplicates_removed': duplicates_removed,
        'final': final_count
    }
    
    print(f"  Final records: {final_count}")
    if duplicates_removed > 0:
        print(f"  Duplicates removed: {duplicates_removed}")
    
    return stats


def find_jsonl_files(base_dir: str) -> List[str]:
    """Find all JSONL files in the susceptibility directory structure."""
    jsonl_files = []
    base_path = Path(base_dir)
    
    # Look for pattern: model_name/[steered|unsteered]/default_50.jsonl
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            for subset_dir in model_dir.iterdir():
                if subset_dir.is_dir() and subset_dir.name in ['steered', 'unsteered']:
                    jsonl_file = subset_dir / 'default_50.jsonl'
                    if jsonl_file.exists():
                        jsonl_files.append(str(jsonl_file))
    
    return sorted(jsonl_files)


def main():
    """Main function to process JSONL files."""
    # Determine what to process based on arguments
    if len(sys.argv) == 1:
        # No arguments - use default directory
        base_dir = './susceptibility/'
        if not os.path.exists(base_dir):
            print(f"Error: Default directory {base_dir} does not exist")
            sys.exit(1)
        print(f"Searching for JSONL files in: {base_dir}")
        jsonl_files = find_jsonl_files(base_dir)
        
    elif len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        # Single argument that's a directory
        base_dir = sys.argv[1]
        print(f"Searching for JSONL files in: {base_dir}")
        jsonl_files = find_jsonl_files(base_dir)
        
    else:
        # Multiple arguments or single file - treat as file list
        base_dir = "."
        jsonl_files = []
        for arg in sys.argv[1:]:
            if os.path.exists(arg) and arg.endswith('.jsonl'):
                jsonl_files.append(os.path.abspath(arg))
            else:
                print(f"Warning: File not found or not a JSONL file: {arg}")
        
        if not jsonl_files:
            print("Error: No valid JSONL files provided!")
            sys.exit(1)
        
        print(f"Processing {len(jsonl_files)} specified files:")
    
    if not jsonl_files:
        print("No JSONL files found!")
        sys.exit(1)
    
    print(f"Found {len(jsonl_files)} JSONL files to process:")
    for file_path in jsonl_files:
        if len(sys.argv) == 1 or os.path.isdir(sys.argv[1]):
            rel_path = os.path.relpath(file_path, base_dir)
        else:
            rel_path = file_path
        print(f"  {rel_path}")
    
    print("\n" + "="*60)
    print("PROCESSING FILES")
    print("="*60)
    
    # Process each file
    total_stats = {
        'files_processed': 0,
        'total_original': 0,
        'total_duplicates_removed': 0,
        'total_final': 0
    }
    
    for file_path in jsonl_files:
        try:
            if len(sys.argv) == 1 or (len(sys.argv) == 2 and os.path.isdir(sys.argv[1])):
                rel_path = os.path.relpath(file_path, base_dir)
            else:
                rel_path = file_path
                
            stats = process_jsonl_file(file_path)
            
            total_stats['files_processed'] += 1
            total_stats['total_original'] += stats['original']
            total_stats['total_duplicates_removed'] += stats['duplicates_removed']
            total_stats['total_final'] += stats['final']
            
            print(f"  ✓ {rel_path}: {stats['original']} → {stats['final']} records")
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path}: {e}")
            continue
        
        print()
    
    # Print summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total original records: {total_stats['total_original']:,}")
    print(f"Total duplicates removed: {total_stats['total_duplicates_removed']:,}")
    print(f"Total final records: {total_stats['total_final']:,}")
    print(f"Space saved: {total_stats['total_duplicates_removed']:,} records")
    
    print("\nBackup files created with .bak extension")
    print("If results look good, you can remove backup files with:")
    print(f"  find {base_dir} -name '*.bak' -delete")


if __name__ == "__main__":
    main()