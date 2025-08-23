#!/usr/bin/env python3
"""
Script to fix question_label fields in default JSONL files by mapping them 
from corresponding roles JSONL files based on matching prompts.
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import re


def extract_final_sentence(prompt):
    """Extract the final sentence from a prompt.
    
    The final sentence starts from the end and goes until the first capitalized letter.
    This works for both role-based prompts and default prompts.
    
    Special case: For "What should I do?" we also include the second to last sentence.
    """
    prompt = prompt.strip()
    if not prompt:
        return prompt
    
    # Start from the end and work backwards to find the start of the final sentence
    # Look for the first capitalized letter working backwards
    final_sentence_start = None
    for i in range(len(prompt) - 1, -1, -1):
        char = prompt[i]
        # If we find a capitalized letter, check if it's the start of a sentence
        if char.isupper():
            # Check if this is likely the start of a sentence
            # (either at the beginning or after punctuation + space)
            if i == 0 or (i > 0 and prompt[i-1] in '.!?' and (i == 1 or prompt[i-2] == ' ')):
                final_sentence_start = i
                break
            # Also check for space before capitalized letter (common sentence boundary)
            elif i > 0 and prompt[i-1] == ' ':
                # Look back further to see if there's punctuation
                j = i - 2
                while j >= 0 and prompt[j] == ' ':
                    j -= 1
                if j >= 0 and prompt[j] in '.!?':
                    final_sentence_start = i
                    break
    
    if final_sentence_start is None:
        # Fallback: return the original prompt if we can't parse it
        return prompt
    
    final_sentence = prompt[final_sentence_start:].strip()
    
    # Special case: if the final sentence is "What should I do?", include the second to last sentence
    if final_sentence == "What should I do?":
        # Find the second to last sentence
        remaining_prompt = prompt[:final_sentence_start].strip()
        if remaining_prompt:
            # Find the start of the second to last sentence
            for i in range(len(remaining_prompt) - 1, -1, -1):
                char = remaining_prompt[i]
                if char.isupper():
                    if i == 0 or (i > 0 and remaining_prompt[i-1] in '.!?' and (i == 1 or remaining_prompt[i-2] == ' ')):
                        return remaining_prompt[i:].strip() + " " + final_sentence
                    elif i > 0 and remaining_prompt[i-1] == ' ':
                        j = i - 2
                        while j >= 0 and remaining_prompt[j] == ' ':
                            j -= 1
                        if j >= 0 and remaining_prompt[j] in '.!?':
                            return remaining_prompt[i:].strip() + " " + final_sentence
    
    return final_sentence


def load_jsonl(file_path):
    """Load JSONL file and return list of records."""
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num} in {file_path}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)
    
    return records


def save_jsonl(records, file_path):
    """Save records to JSONL file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        sys.exit(1)


def build_question_mapping(roles_records):
    """Build mapping from questions to question_labels from roles data."""
    question_to_labels = defaultdict(set)
    
    for record in roles_records:
        prompt = record.get('prompt', '')
        question_label = record.get('question_label', '')
        
        if not prompt or not question_label:
            continue
            
        # Extract the final sentence from the role-based prompt
        final_sentence = extract_final_sentence(prompt)
        question_to_labels[final_sentence].add(question_label)
    
    return question_to_labels


def check_duplicate_mappings(question_to_labels):
    """Check for questions that map to multiple roles and report them."""
    duplicates = []
    
    for question, labels in question_to_labels.items():
        if len(labels) > 1:
            duplicates.append((question, labels))
    
    return duplicates


def fix_question_labels(default_records, question_to_labels):
    """Fix question_label fields in default records based on mapping."""
    fixed_count = 0
    unfixed_count = 0
    
    for record in default_records:
        prompt = record.get('prompt', '')
        current_label = record.get('question_label', '')
        
        if current_label != 'default':
            continue  # Skip records that already have non-default labels
        
        # Extract final sentence from default prompt and try to find a matching question in our mapping
        final_sentence = extract_final_sentence(prompt)
        matched_labels = question_to_labels.get(final_sentence, set())
        
        if len(matched_labels) == 1:
            # Exact match found
            new_label = list(matched_labels)[0]
            record['question_label'] = new_label
            fixed_count += 1
        elif len(matched_labels) > 1:
            # Multiple matches - use the first one alphabetically for consistency
            sorted_labels = sorted(matched_labels)
            new_label = sorted_labels[0]
            record['question_label'] = new_label
            fixed_count += 1
            print(f"Warning: Multiple labels found for question '{prompt[:50]}...'")
            print(f"  Options: {sorted_labels}, using '{new_label}'")
        else:
            # No match found - try fuzzy matching by looking for similar questions
            unfixed_count += 1
            print(f"Warning: No matching question_label found for: '{prompt[:50]}...'")
    
    return fixed_count, unfixed_count


def main():
    parser = argparse.ArgumentParser(
        description='Fix question_label fields in default JSONL files by mapping from roles JSONL files'
    )
    parser.add_argument('default_file', help='Path to the default JSONL file to fix')
    parser.add_argument('roles_file', help='Path to the roles JSONL file to use as reference')
    parser.add_argument('--output', '-o', help='Output file path (defaults to overwriting input file)')
    parser.add_argument('--check-duplicates', action='store_true', 
                       help='Check for questions that map to multiple roles')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be changed without making changes')
    
    args = parser.parse_args()
    
    print(f"Loading roles data from: {args.roles_file}")
    roles_records = load_jsonl(args.roles_file)
    print(f"Loaded {len(roles_records)} role records")
    
    print(f"Loading default data from: {args.default_file}")
    default_records = load_jsonl(args.default_file)
    print(f"Loaded {len(default_records)} default records")
    
    # Build the question mapping
    print("Building question mapping...")
    question_to_labels = build_question_mapping(roles_records)
    print(f"Found {len(question_to_labels)} unique questions in roles data")
    
    # Check for duplicates if requested
    if args.check_duplicates:
        print("\nChecking for duplicate question mappings...")
        duplicates = check_duplicate_mappings(question_to_labels)
        if duplicates:
            print(f"Found {len(duplicates)} questions that map to multiple roles:")
            for question, labels in duplicates:
                print(f"  Question: '{question[:60]}...'")
                print(f"  Maps to roles: {sorted(labels)}")
                print()
        else:
            print("No duplicate mappings found!")
    
    # Fix the labels
    if not args.dry_run:
        print("Fixing question labels...")
        fixed_count, unfixed_count = fix_question_labels(default_records, question_to_labels)
        
        print(f"Fixed {fixed_count} records")
        print(f"Unable to fix {unfixed_count} records")
        
        # Save the results
        output_file = args.output or args.default_file
        print(f"Saving results to: {output_file}")
        save_jsonl(default_records, output_file)
        print("Done!")
    else:
        print("Dry run mode - showing what would be changed:")
        fixed_count, unfixed_count = fix_question_labels(default_records, question_to_labels)
        print(f"Would fix {fixed_count} records")
        print(f"Would be unable to fix {unfixed_count} records")


if __name__ == '__main__':
    main()