#!/usr/bin/env python3
"""
Script to match and sort evaluation results with their corresponding data entries.
"""

import json
import argparse
from pathlib import Path


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def save_jsonl(data, file_path):
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def create_prompt_to_id_mapping(data_entries):
    """Create mapping from concatenated prompt+question to id."""
    mapping = {}
    for entry in data_entries:
        # Concatenate prompt and question with a space
        combined_prompt = entry['prompt'] + ' ' + entry['question']
        mapping[combined_prompt] = entry['id']
    return mapping


def match_and_sort(results_file, data_file, output_file=None):
    """
    Match results with data entries and sort by magnitude then id.
    
    Args:
        results_file: Path to results JSONL file
        data_file: Path to data JSONL file  
        output_file: Output file path (defaults to overwriting results_file)
    """
    if output_file is None:
        output_file = results_file
    
    print(f"Loading results from {results_file}")
    results = load_jsonl(results_file)
    
    print(f"Loading data from {data_file}")
    data = load_jsonl(data_file)
    
    print("Creating prompt-to-id mapping...")
    prompt_to_id = create_prompt_to_id_mapping(data)
    
    matched_count = 0
    unmatched_count = 0
    
    # Match results with data entries
    for result in results:
        prompt = result.get('prompt', '')
        if prompt in prompt_to_id:
            result['question_id'] = prompt_to_id[prompt]
            matched_count += 1
        else:
            # Set question_id to -1 for unmatched entries
            result['question_id'] = -1
            unmatched_count += 1
            print(f"Warning: No match found for prompt: {prompt[:100]}...")
    
    print(f"Matched: {matched_count}, Unmatched: {unmatched_count}")
    
    # Sort by magnitude (ascending), then by question_id (ascending)
    print("Sorting results by magnitude, then question_id...")
    results.sort(key=lambda x: (x.get('magnitude', 0), x.get('question_id', -1)))
    
    print(f"Saving sorted results to {output_file}")
    save_jsonl(results, output_file)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Match and sort evaluation results')
    parser.add_argument('results_file', help='Path to results JSONL file')
    parser.add_argument('data_file', help='Path to data JSONL file')
    parser.add_argument('-o', '--output', help='Output file path (defaults to overwriting results file)')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_file)
    data_path = Path(args.data_file)
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1
    
    match_and_sort(results_path, data_path, args.output)
    return 0


if __name__ == '__main__':
    exit(main())