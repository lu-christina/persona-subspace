#!/usr/bin/env python3
"""
Cleanup script to fix existing trait JSON files.

This script processes all trait JSON files in /root/git/persona-subspace/traits/data/
and extracts the actual JSON data from the raw_response field, converting files
from the format:
{
  "raw_response": "```json\n{actual_data}\n```"
}

To the clean format:
{
  "instruction": [...],
  "questions": [...], 
  "eval_prompt": "..."
}
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional
import argparse
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

class TraitFileCleanup:
    """Cleanup utility for trait JSON files."""
    
    def __init__(self, create_backups: bool = True):
        self.create_backups = create_backups
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        
    def extract_json_from_raw_response(self, raw_response: str) -> Optional[Dict]:
        """Extract JSON from markdown code blocks or other formats."""
        # Try to find JSON in markdown code blocks
        json_pattern = r'```json\s*\n(.*?)\n```'
        matches = re.findall(json_pattern, raw_response, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError as e:
                LOGGER.debug(f"Failed to parse JSON from markdown block: {e}")
                continue
        
        # Try to find JSON without markdown formatting
        # Look for content between { and } that spans multiple lines
        json_pattern2 = r'\{[\s\S]*\}'
        matches2 = re.findall(json_pattern2, raw_response)
        
        for match in matches2:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError as e:
                LOGGER.debug(f"Failed to parse JSON from direct match: {e}")
                continue
        
        return None
    
    def is_valid_trait_json(self, data: Dict) -> bool:
        """Check if the JSON data has the expected trait structure."""
        required_fields = ["instruction", "questions", "eval_prompt"]
        return all(field in data for field in required_fields)
    
    def process_trait_file(self, file_path: Path) -> bool:
        """Process a single trait file and fix its format if needed."""
        try:
            # Read the current file
            with open(file_path, 'r') as f:
                current_data = json.load(f)
            
            # Check if it's already in the correct format
            if self.is_valid_trait_json(current_data):
                LOGGER.debug(f"File {file_path.name} is already in correct format, skipping")
                self.skipped_count += 1
                return True
            
            # Check if it has raw_response field
            if "raw_response" not in current_data:
                LOGGER.warning(f"File {file_path.name} doesn't have raw_response field and isn't in correct format")
                self.error_count += 1
                return False
            
            # Extract JSON from raw_response
            raw_response = current_data["raw_response"]
            extracted_data = self.extract_json_from_raw_response(raw_response)
            
            if not extracted_data:
                LOGGER.error(f"Could not extract JSON from {file_path.name}")
                self.error_count += 1
                return False
            
            # Validate the extracted data
            if not self.is_valid_trait_json(extracted_data):
                LOGGER.error(f"Extracted data from {file_path.name} doesn't have expected trait structure")
                self.error_count += 1
                return False
            
            # Create backup if requested
            if self.create_backups:
                backup_path = file_path.with_suffix('.json.backup')
                shutil.copy2(file_path, backup_path)
                LOGGER.debug(f"Created backup: {backup_path}")
            
            # Write the cleaned data
            with open(file_path, 'w') as f:
                json.dump(extracted_data, f, indent=2)
            
            LOGGER.info(f"Successfully cleaned {file_path.name}")
            self.processed_count += 1
            return True
            
        except Exception as e:
            LOGGER.error(f"Error processing {file_path.name}: {str(e)}")
            self.error_count += 1
            return False
    
    def process_directory(self, traits_dir: Path) -> Dict[str, int]:
        """Process all trait JSON files in the directory."""
        if not traits_dir.exists():
            raise FileNotFoundError(f"Directory {traits_dir} does not exist")
        
        # Find all JSON files (excluding subdirectories and special files)
        json_files = []
        for file_path in traits_dir.iterdir():
            if (file_path.is_file() and 
                file_path.suffix == '.json' and 
                not file_path.name.startswith('processing_summary') and
                not file_path.name.endswith('.backup')):
                json_files.append(file_path)
        
        LOGGER.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        for file_path in sorted(json_files):
            self.process_trait_file(file_path)
        
        # Return summary
        summary = {
            "total_files": len(json_files),
            "processed": self.processed_count,
            "skipped": self.skipped_count,
            "errors": self.error_count
        }
        
        return summary
    
    def cleanup_backups(self, traits_dir: Path):
        """Remove all backup files from the directory."""
        backup_files = list(traits_dir.glob("*.json.backup"))
        for backup_file in backup_files:
            backup_file.unlink()
            LOGGER.info(f"Removed backup: {backup_file.name}")
        
        LOGGER.info(f"Removed {len(backup_files)} backup files")


def main():
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(description="Clean up trait JSON files by extracting actual JSON from raw_response fields")
    parser.add_argument(
        "--traits-dir",
        type=str,
        default="/root/git/persona-subspace/traits/data",
        help="Directory containing trait JSON files"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    parser.add_argument(
        "--cleanup-backups",
        action="store_true",
        help="Remove all existing backup files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    traits_dir = Path(args.traits_dir)
    
    if args.cleanup_backups:
        cleanup = TraitFileCleanup()
        cleanup.cleanup_backups(traits_dir)
        return
    
    if args.dry_run:
        LOGGER.info("DRY RUN MODE - No changes will be made")
    
    # Create cleanup instance
    cleanup = TraitFileCleanup(create_backups=not args.no_backup)
    
    try:
        if args.dry_run:
            # For dry run, just count what would be processed
            json_files = []
            for file_path in traits_dir.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix == '.json' and 
                    not file_path.name.startswith('processing_summary') and
                    not file_path.name.endswith('.backup')):
                    json_files.append(file_path)
            
            print(f"Would process {len(json_files)} JSON files:")
            for file_path in sorted(json_files):
                print(f"  - {file_path.name}")
        else:
            # Actually process the files
            summary = cleanup.process_directory(traits_dir)
            
            # Print summary
            print(f"\n=== CLEANUP SUMMARY ===")
            print(f"Total files found: {summary['total_files']}")
            print(f"Successfully processed: {summary['processed']}")
            print(f"Skipped (already clean): {summary['skipped']}")
            print(f"Errors: {summary['errors']}")
            
            if summary['errors'] > 0:
                print(f"\nSome files had errors. Check the logs above for details.")
            
            if not args.no_backup and summary['processed'] > 0:
                print(f"\nBackup files created. Use --cleanup-backups to remove them after verifying results.")
        
    except Exception as e:
        LOGGER.error(f"Failed to process directory: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())