#!/usr/bin/env python3
"""
Script to fix the num_turns field in JSON/JSONL files and rename JSONL files to JSON.

This script:
1. Finds all .json and .jsonl files in the results/ directory
2. Fixes the 'turns' field to match the actual length of the 'conversation' array
3. Renames .jsonl files to .json files
4. Reports any changes made
"""

import json
import os
import glob
from pathlib import Path


def process_file(file_path):
    """Process a single JSON/JSONL file to fix turns and rename if needed."""
    changes_made = []

    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if the file has the expected structure
        if 'conversation' not in data or 'turns' not in data:
            print(f"Skipping {file_path}: Missing 'conversation' or 'turns' field")
            return changes_made

        # Get actual conversation length
        actual_turns = len(data['conversation'])
        current_turns = data['turns']

        # Fix turns field if incorrect
        if current_turns != actual_turns:
            data['turns'] = actual_turns
            changes_made.append(f"Updated turns from {current_turns} to {actual_turns}")

            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        # Rename JSONL to JSON if needed
        if file_path.endswith('.jsonl'):
            new_path = file_path.replace('.jsonl', '.json')
            os.rename(file_path, new_path)
            changes_made.append(f"Renamed {file_path} to {new_path}")

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return changes_made


def main():
    """Main function to process all JSON/JSONL files in results/ directory."""
    results_dir = Path("results")

    if not results_dir.exists():
        print("Error: results/ directory not found")
        return

    # Find all JSON and JSONL files
    json_files = list(results_dir.glob("**/*.json"))
    jsonl_files = list(results_dir.glob("**/*.jsonl"))
    all_files = json_files + jsonl_files

    if not all_files:
        print("No JSON or JSONL files found in results/ directory")
        return

    print(f"Found {len(all_files)} files to process:")
    print(f"  - {len(json_files)} .json files")
    print(f"  - {len(jsonl_files)} .jsonl files")
    print()

    total_changes = 0
    files_changed = 0

    for file_path in all_files:
        changes = process_file(str(file_path))

        if changes:
            files_changed += 1
            total_changes += len(changes)
            print(f"✓ {file_path}:")
            for change in changes:
                print(f"  - {change}")
            print()

    print(f"Summary:")
    print(f"  - Processed {len(all_files)} files")
    print(f"  - Modified {files_changed} files")
    print(f"  - Made {total_changes} total changes")


if __name__ == "__main__":
    main()