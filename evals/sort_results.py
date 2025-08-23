#!/usr/bin/env python3
"""
Script to sort JSONL files in results/roles_traits directory.
Sorts by: magnitude, role_id, question_label (if not harmbench), question_id
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


def sort_jsonl_file(file_path: Path) -> None:
    """Sort a JSONL file by the specified criteria."""
    # Only process harmbench files
    if not file_path.name.startswith('harmbench'):
        print(f"Skipping {file_path.name} (not a harmbench file)")
        return
        
    print(f"Processing {file_path.name}...")
    
    # Read all lines from the file
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    lines.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num} in {file_path.name}: {e}")
                    continue
    
    if not lines:
        print(f"No valid JSON lines found in {file_path.name}")
        return
    
    # Sort the lines by magnitude, role_id, question_id
    def sort_key(item: Dict[str, Any]) -> tuple:
        magnitude = item.get('magnitude', 0)
        role_id = item.get('role_id', 0)
        question_id = item.get('question_id', 0)
        return (magnitude, role_id, question_id)
    
    sorted_lines = sorted(lines, key=sort_key)
    
    # Write sorted lines back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in sorted_lines:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print(f"Sorted {len(sorted_lines)} entries in {file_path.name}")


def main():
    """Main function to sort all JSONL files in the results directory."""
    results_dir = Path('/root/git/persona-subspace/evals/results/roles_traits')
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return
    
    # Find all .jsonl files
    jsonl_files = list(results_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files (will only process harmbench files)")
    print("Sorting criteria for harmbench files: magnitude, role_id, question_id")
    print()
    
    for file_path in sorted(jsonl_files):
        sort_jsonl_file(file_path)
        print()
    
    print("All files sorted successfully!")


if __name__ == '__main__':
    main()