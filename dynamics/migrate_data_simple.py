#!/usr/bin/env python3
"""
Simple migration script to fix turn counting in existing conversation data.
"""

import json
import re
from pathlib import Path
from typing import Dict, List
import shutil


def main():
    """Main migration function."""
    results_dir = Path("/root/git/persona-subspace/dynamics/results")
    
    print("=== Migrating conversation data to correct turn counting ===")
    print()
    
    # Process each directory
    processed_files = []
    
    for old_dir in results_dir.iterdir():
        if not old_dir.is_dir() or not old_dir.name.endswith("_turns") or old_dir.name == "backup_old_format":
            continue
            
        print(f"Processing {old_dir.name}...")
        
        for json_file in old_dir.glob("self_conversation_*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation = data.get('conversation', [])
            actual_turns = len([msg for msg in conversation if msg['role'] == 'user'])
            old_turns = data.get('turns', 0)
            
            if actual_turns != old_turns:
                # Update metadata
                data['turns'] = actual_turns
                
                # Write back to same file
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"  Updated {json_file.name}: {old_turns} → {actual_turns} turns")
                processed_files.append((json_file, old_turns, actual_turns))
            else:
                print(f"  {json_file.name}: already correct ({actual_turns} turns)")
    
    print()
    print("=== Now renaming directories ===")
    print()
    
    # Now rename directories
    rename_plan = []
    for old_dir in results_dir.iterdir():
        if not old_dir.is_dir() or not old_dir.name.endswith("_turns") or old_dir.name == "backup_old_format":
            continue
            
        old_name = old_dir.name
        old_turn_count = int(old_name.replace("_turns", ""))
        
        # Check what the actual turn count should be by looking at files
        json_files = list(old_dir.glob("self_conversation_*.json"))
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            conversation = data.get('conversation', [])
            actual_turns = len([msg for msg in conversation if msg['role'] == 'user'])
            
            new_name = f"{actual_turns}_turns"
            
            if new_name != old_name:
                rename_plan.append((old_dir, new_name, actual_turns))
    
    # Execute renames, handling conflicts
    for old_dir, new_name, actual_turns in rename_plan:
        new_path = results_dir / new_name
        
        if new_path.exists():
            print(f"Target {new_name} already exists, merging...")
            # Move files with new IDs
            existing_files = list(new_path.glob("self_conversation_*.json"))
            max_id = 0
            pattern = re.compile(r'self_conversation_(\d+)\.json')
            for f in existing_files:
                match = pattern.match(f.name)
                if match:
                    max_id = max(max_id, int(match.group(1)))
            
            next_id = max_id + 1
            files_to_move = list(old_dir.glob("self_conversation_*.json"))
            
            for i, file_path in enumerate(files_to_move):
                new_id = next_id + i
                new_filename = f"self_conversation_{new_id}.json"
                new_filepath = new_path / new_filename
                shutil.move(str(file_path), str(new_filepath))
                print(f"  Moved {file_path.name} → {new_name}/{new_filename}")
            
            # Remove empty old directory
            old_dir.rmdir()
            print(f"  Removed empty {old_dir.name}/")
        else:
            # Simple rename
            old_dir.rename(new_path)
            print(f"Renamed {old_dir.name}/ → {new_name}/")
    
    print()
    print("=== Migration completed! ===")
    print()
    
    # Show final structure
    print("Final directory structure:")
    for item in sorted(results_dir.iterdir()):
        if item.is_dir() and item.name.endswith("_turns") and item.name != "backup_old_format":
            file_count = len(list(item.glob("self_conversation_*.json")))
            print(f"  {item.name}/ ({file_count} files)")


if __name__ == "__main__":
    main()