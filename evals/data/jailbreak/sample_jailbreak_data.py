#!/usr/bin/env python3
"""
Script to sample persona_jailbreak.jsonl to create jailbreak_sample.jsonl

Samples entries with:
- role_id 0-4 (from {harm_id}_{role_id} format)
- prompt_id = 0 (first prompt for each harm/role combination)

Expected output: 44 harms × 5 roles × 5 questions = 1,100 entries
"""

import json
import sys
from pathlib import Path

def main():
    input_file = Path("jailbreak_4400.jsonl")
    output_file = Path("jailbreak_1100.jsonl")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)
    
    sampled_entries = []
    total_entries = 0
    filtered_entries = 0
    
    print("Reading and filtering entries...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                entry = json.loads(line)
                total_entries += 1
                
                # Extract harm_id and role_id from role field
                role = entry.get('role', '')
                if '_' not in role:
                    continue
                    
                try:
                    harm_id, role_id = map(int, role.split('_'))
                except ValueError:
                    continue
                
                # Filter criteria
                prompt_id = entry.get('prompt_id', -1)
                
                # Include only role_id 0-4 and prompt_id 0
                if role_id in range(5) and prompt_id == 0:
                    sampled_entries.append(entry)
                    filtered_entries += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Total entries read: {total_entries}")
    print(f"Entries matching criteria: {filtered_entries}")
    print(f"Entries to write: {len(sampled_entries)}")
    
    # Write sampled data
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in sampled_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Successfully created {output_file} with {len(sampled_entries)} entries")
    
    # Validation
    expected_entries = 44 * 5 * 5  # 44 harms × 5 roles × 5 questions
    if len(sampled_entries) == expected_entries:
        print(f"✓ Sample size matches expected: {expected_entries} entries")
    else:
        print(f"⚠ Sample size mismatch. Expected: {expected_entries}, Got: {len(sampled_entries)}")

if __name__ == "__main__":
    main()