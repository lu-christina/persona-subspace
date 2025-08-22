#!/usr/bin/env python3
"""
Script to clean up JSONL files by removing lines with empty responses.
Usage: python cleanup_errors.py <directory_path>
"""

import json
import os
import sys
from pathlib import Path


def cleanup_jsonl_file(file_path):
    """Remove lines with empty responses from a JSONL file."""
    temp_file = file_path.with_suffix('.tmp')
    lines_removed = 0
    lines_kept = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(temp_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Check if response field exists and is empty
                    if 'response' in data and data['response'] == "":
                        lines_removed += 1
                    else:
                        outfile.write(line + '\n')
                        lines_kept += 1
                except json.JSONDecodeError as e:
                    print(f"  Warning: Invalid JSON on line {line_num}: {e}")
                    outfile.write(line + '\n')
                    lines_kept += 1
        
        # Replace original file with cleaned version
        temp_file.replace(file_path)
        return lines_kept, lines_removed
        
    except Exception as e:
        # Clean up temp file if something went wrong
        if temp_file.exists():
            temp_file.unlink()
        raise e


def main():
    if len(sys.argv) != 2:
        print("Usage: python cleanup_errors.py <directory_path>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)
    
    # Find all JSONL files in the directory
    jsonl_files = list(directory.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in '{directory}'")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files in '{directory}'")
    
    total_kept = 0
    total_removed = 0
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file.name}")
        try:
            kept, removed = cleanup_jsonl_file(jsonl_file)
            total_kept += kept
            total_removed += removed
            print(f"  Kept: {kept} lines, Removed: {removed} lines")
        except Exception as e:
            print(f"  Error processing {jsonl_file.name}: {e}")
    
    print(f"\nSummary:")
    print(f"  Total lines kept: {total_kept}")
    print(f"  Total lines removed: {total_removed}")


if __name__ == "__main__":
    main()