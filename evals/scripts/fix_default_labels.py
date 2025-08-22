#!/usr/bin/env python3
"""
Script to fix default question labels in baseline JSONL files by mapping prompts 
from a source file to get the correct question_label values.

Usage:
    python fix_default_labels.py -b baseline_file.jsonl -s source_file.jsonl
"""

import json
import argparse
import sys
from pathlib import Path


def load_prompt_to_label_mapping(source_file):
    """
    Load the source JSONL file and create a mapping from prompt to question_label.
    
    Args:
        source_file (str): Path to the source JSONL file with correct labels
        
    Returns:
        dict: Mapping of prompt text to question_label
    """
    prompt_mapping = {}
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    prompt = data.get('prompt', '')
                    question_label = data.get('question_label', '')
                    
                    if prompt and question_label:
                        prompt_mapping[prompt] = question_label
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Malformed JSON on line {line_num} in {source_file}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading source file '{source_file}': {e}")
        sys.exit(1)
        
    print(f"Loaded {len(prompt_mapping)} prompt-to-label mappings from {source_file}")
    return prompt_mapping


def fix_labels(baseline_file, prompt_mapping):
    """
    Fix the question labels in the baseline file and overwrite it.
    
    Args:
        baseline_file (str): Path to the baseline JSONL file to fix
        prompt_mapping (dict): Mapping of prompt to question_label
    """
    updated_lines = []
    changes_made = 0
    total_lines = 0
    
    try:
        # Read the baseline file
        with open(baseline_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                total_lines += 1
                
                try:
                    data = json.loads(line)
                    prompt = data.get('prompt', '')
                    current_label = data.get('question_label', '')
                    
                    # Check if we have a mapping for this prompt
                    if prompt in prompt_mapping:
                        new_label = prompt_mapping[prompt]
                        
                        # Only update if the label is different
                        if current_label != new_label:
                            data['question_label'] = new_label
                            changes_made += 1
                            print(f"Line {line_num}: '{current_label}' → '{new_label}'")
                    
                    # Add the (possibly updated) line
                    updated_lines.append(json.dumps(data))
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Malformed JSON on line {line_num} in {baseline_file}: {e}")
                    # Keep the original line if we can't parse it
                    updated_lines.append(line)
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Baseline file '{baseline_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading baseline file '{baseline_file}': {e}")
        sys.exit(1)
    
    # Write back to the original file
    try:
        with open(baseline_file, 'w', encoding='utf-8') as f:
            for line in updated_lines:
                f.write(line + '\n')
                
        print(f"\nCompleted: Updated {changes_made} labels out of {total_lines} total lines")
        print(f"File '{baseline_file}' has been updated.")
        
    except Exception as e:
        print(f"Error writing to baseline file '{baseline_file}': {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and orchestrate the fix."""
    parser = argparse.ArgumentParser(
        description="Fix default question labels in baseline JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_default_labels.py -b roles_20_baseline.jsonl -s roles_20.jsonl
  python fix_default_labels.py --baseline default_20_baseline.jsonl --source default_20.jsonl
        """
    )
    
    parser.add_argument(
        '-b', '--baseline',
        required=True,
        help='Path to the baseline JSONL file with default labels to fix'
    )
    
    parser.add_argument(
        '-s', '--source', 
        required=True,
        help='Path to the source JSONL file with correct question labels'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    baseline_path = Path(args.baseline)
    source_path = Path(args.source)
    
    if not baseline_path.exists():
        print(f"Error: Baseline file '{args.baseline}' does not exist.")
        sys.exit(1)
        
    if not source_path.exists():
        print(f"Error: Source file '{args.source}' does not exist.")
        sys.exit(1)
    
    print(f"Baseline file: {args.baseline}")
    print(f"Source file: {args.source}")
    print()
    
    # Load the prompt-to-label mapping
    prompt_mapping = load_prompt_to_label_mapping(args.source)
    
    if not prompt_mapping:
        print("No mappings found in source file. Nothing to do.")
        sys.exit(0)
    
    # Fix the labels in the baseline file
    fix_labels(args.baseline, prompt_mapping)


if __name__ == '__main__':
    main()