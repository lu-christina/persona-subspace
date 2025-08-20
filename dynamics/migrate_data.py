#!/usr/bin/env python3
"""
Migration script to fix turn counting in existing conversation data.

This script:
1. Renames directories from old format (message count) to new format (actual turns)
2. Updates JSON metadata to reflect correct turn counts
3. Handles conflicts by continuing file IDs
"""

import json
import re
from pathlib import Path
from typing import Dict, List
import shutil


def get_next_file_id(directory: Path) -> int:
    """Find the next available file ID in a directory."""
    if not directory.exists():
        return 1
    
    max_id = 0
    pattern = re.compile(r'self_conversation_(\d+)\.json')
    
    for file_path in directory.glob("self_conversation_*.json"):
        match = pattern.match(file_path.name)
        if match:
            file_id = int(match.group(1))
            max_id = max(max_id, file_id)
    
    return max_id + 1


def migrate_directory(old_dir: Path, results_dir: Path) -> None:
    """Migrate one directory from old to new naming convention."""
    print(f"Processing {old_dir.name}...")
    
    # Process each JSON file to determine actual turns
    files_to_migrate = []
    
    for json_file in old_dir.glob("self_conversation_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversation = data.get('conversation', [])
        actual_turns = len([msg for msg in conversation if msg['role'] == 'user'])
        
        files_to_migrate.append({
            'file_path': json_file,
            'data': data,
            'actual_turns': actual_turns
        })
    
    if not files_to_migrate:
        print(f"  No files found in {old_dir.name}")
        return
    
    # Group files by actual turn count
    turn_groups: Dict[int, List] = {}
    for file_info in files_to_migrate:
        turns = file_info['actual_turns']
        if turns not in turn_groups:
            turn_groups[turns] = []
        turn_groups[turns].append(file_info)
    
    # Migrate each group
    for actual_turns, files in turn_groups.items():
        new_dir = results_dir / f"{actual_turns}_turns"
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Get starting file ID for this directory
        next_id = get_next_file_id(new_dir)
        
        print(f"  Moving {len(files)} files with {actual_turns} turns to {new_dir.name}/")
        
        for i, file_info in enumerate(files):
            # Update metadata
            new_data = file_info['data'].copy()
            new_data['turns'] = actual_turns
            
            # Save to new location
            new_file_id = next_id + i
            new_filename = f"self_conversation_{new_file_id}.json"
            new_filepath = new_dir / new_filename
            
            with open(new_filepath, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            
            print(f"    {file_info['file_path'].name} → {new_filename}")


def main():
    """Main migration function."""
    results_dir = Path("/root/git/persona-subspace/dynamics/results")
    
    if not results_dir.exists():
        print("Results directory not found!")
        return
    
    print("=== Migrating conversation data to correct turn counting ===")
    print()
    
    # Find all directories that need migration
    old_directories = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.endswith("_turns"):
            old_directories.append(item)
    
    if not old_directories:
        print("No directories found to migrate.")
        return
    
    # Create backup
    backup_dir = results_dir / "backup_old_format"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir()
    
    print(f"Creating backup in {backup_dir}...")
    for old_dir in old_directories:
        backup_path = backup_dir / old_dir.name
        shutil.copytree(old_dir, backup_path)
    print()
    
    # Migrate each directory
    for old_dir in old_directories:
        migrate_directory(old_dir, results_dir)
        print()
    
    # Clean up old directories (but only if they don't contain migrated data)
    print("Cleaning up old directories...")
    directories_to_remove = []
    
    # First, identify which directories are safe to remove
    for old_dir in old_directories:
        old_name = old_dir.name
        # Check if any files were migrated to a directory with the same name
        safe_to_remove = True
        for item in results_dir.iterdir():
            if item.is_dir() and item.name.endswith("_turns") and item.name == old_name:
                # This directory has the same name as migration target, check if it has new data
                json_files = list(item.glob("self_conversation_*.json"))
                if json_files:
                    # Directory has files, check if they're migrated (updated metadata)
                    with open(json_files[0], 'r') as f:
                        data = json.load(f)
                    conversation = data.get('conversation', [])
                    actual_turns = len([msg for msg in conversation if msg['role'] == 'user'])
                    if data.get('turns') == actual_turns:
                        # This is migrated data, don't remove
                        safe_to_remove = False
                        break
        
        if safe_to_remove:
            directories_to_remove.append(old_dir)
    
    # Remove only the safe directories
    for old_dir in directories_to_remove:
        shutil.rmtree(old_dir)
        print(f"  Removed {old_dir.name}/")
    
    print()
    print("=== Migration completed! ===")
    print(f"Backup saved in: {backup_dir}")
    print()
    
    # Show final structure
    print("New directory structure:")
    for item in sorted(results_dir.iterdir()):
        if item.is_dir() and item.name.endswith("_turns") and item.name != "backup_old_format":
            file_count = len(list(item.glob("self_conversation_*.json")))
            print(f"  {item.name}/ ({file_count} files)")


if __name__ == "__main__":
    main()