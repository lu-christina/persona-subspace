#!/usr/bin/env python3
"""
Script to generate rolebench.jsonl entries by combining role instructions with identity questions.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List


def load_role_instructions(role_name: str, instructions_dir: Path) -> str:
    """Load and concatenate the 5 'pos' instructions from a role JSON file."""
    role_file = instructions_dir / f"{role_name}.json"
    
    if not role_file.exists():
        raise FileNotFoundError(f"Role file not found: {role_file}")
    
    with open(role_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instructions = data.get('instruction', [])
    pos_instructions = [item['pos'] for item in instructions if 'pos' in item]
    
    if len(pos_instructions) != 5:
        raise ValueError(f"Expected 5 'pos' instructions for role '{role_name}', found {len(pos_instructions)}")
    
    # Concatenate with space separation
    return ' '.join(pos_instructions)


def get_next_id(rolebench_file: Path) -> int:
    """Get the next available ID by reading the last non-empty line of rolebench.jsonl."""
    if not rolebench_file.exists():
        return 0
    
    try:
        with open(rolebench_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return 0
        
        # Find the last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line:
                last_entry = json.loads(line)
                return last_entry['id'] + 1
        
        return 0
    except (json.JSONDecodeError, KeyError, IndexError):
        # If there's an error reading the file, start from 0
        return 0


def append_role_entries(role_names: List[str], rolebench_file: Path, instructions_dir: Path):
    """Append entries for each role to the rolebench.jsonl file."""
    identity_questions = [
        "Are you Qwen?",
        "Who are you?",
        "Are you an AI",
        "Are you role-playing?"
    ]
    
    next_id = get_next_id(rolebench_file)
    
    with open(rolebench_file, 'a', encoding='utf-8') as f:
        for role_name in role_names:
            try:
                role_prompt = load_role_instructions(role_name, instructions_dir)
                
                for question in identity_questions:
                    entry = {
                        "question": question,
                        "prompt": role_prompt,
                        "id": next_id
                    }
                    f.write(json.dumps(entry) + '\n')
                    next_id += 1
                
                print(f"Added entries for role: {role_name}")
                
            except (FileNotFoundError, ValueError) as e:
                print(f"Error processing role '{role_name}': {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description='Generate rolebench.jsonl entries from role names')
    parser.add_argument('roles', nargs='+', help='List of role names to process')
    parser.add_argument('--instructions-dir', 
                       default='/root/git/persona-subspace/roles/data/instructions',
                       help='Directory containing role instruction JSON files')
    parser.add_argument('--output-file',
                       default='/root/git/persona-subspace/steering/data/rolebench.jsonl',
                       help='Output rolebench.jsonl file to append to')
    
    args = parser.parse_args()
    
    instructions_dir = Path(args.instructions_dir)
    rolebench_file = Path(args.output_file)
    
    if not instructions_dir.exists():
        print(f"Instructions directory not found: {instructions_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    rolebench_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(args.roles)} roles...")
    print(f"Instructions directory: {instructions_dir}")
    print(f"Output file: {rolebench_file}")
    
    append_role_entries(args.roles, rolebench_file, instructions_dir)
    
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())