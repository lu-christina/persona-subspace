#!/usr/bin/env python3
"""
Script to rename 'pair_index' field to 'prompt_index' in existing trait response files.

This script processes JSONL files that already have the 'pair_index' field
(from previous migration) and renames it to 'prompt_index' for better clarity.

Usage:
    uv run traits/rename_field.py --input-dir /workspace/traits/responses
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List
import jsonlines

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FieldRenamer:
    """Renames 'pair_index' field to 'prompt_index' in trait response files."""
    
    def __init__(self, input_dir: str, backup_suffix: str = ".backup"):
        """
        Initialize the field renamer.
        
        Args:
            input_dir: Directory containing JSONL response files
            backup_suffix: Suffix for backup files
        """
        self.input_dir = Path(input_dir)
        self.backup_suffix = backup_suffix
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        logger.info(f"Initialized FieldRenamer for directory: {input_dir}")
    
    def find_response_files(self) -> List[Path]:
        """
        Find all JSONL response files in the input directory.
        
        Returns:
            List of JSONL file paths
        """
        jsonl_files = list(self.input_dir.glob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} JSONL files to process")
        return jsonl_files
    
    def load_responses(self, file_path: Path) -> List[Dict]:
        """
        Load responses from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of response dictionaries
        """
        responses = []
        try:
            with jsonlines.open(file_path, 'r') as reader:
                for response in reader:
                    responses.append(response)
            
            logger.debug(f"Loaded {len(responses)} responses from {file_path.name}")
            return responses
            
        except Exception as e:
            logger.error(f"Error loading responses from {file_path}: {e}")
            return []
    
    def validate_has_pair_index(self, responses: List[Dict], file_path: Path) -> bool:
        """
        Validate that responses have 'pair_index' field that needs renaming.
        
        Args:
            responses: List of response dictionaries
            file_path: Path to the file being validated
            
        Returns:
            True if file has pair_index field and needs renaming
        """
        if not responses:
            logger.warning(f"No responses found in {file_path.name}")
            return False
        
        # Check if pair_index field exists
        if 'pair_index' not in responses[0]:
            logger.info(f"File {file_path.name} does not have pair_index field, skipping")
            return False
        
        # Check if prompt_index already exists
        if 'prompt_index' in responses[0]:
            logger.info(f"File {file_path.name} already has prompt_index field, skipping")
            return False
        
        logger.info(f"File {file_path.name} has pair_index field, will rename to prompt_index")
        return True
    
    def rename_field_in_responses(self, responses: List[Dict]) -> List[Dict]:
        """
        Rename 'pair_index' to 'prompt_index' in all responses.
        
        Args:
            responses: List of response dictionaries
            
        Returns:
            List of response dictionaries with renamed field
        """
        renamed_responses = []
        
        for response in responses:
            # Create a copy of the response
            renamed_response = response.copy()
            
            # Rename pair_index to prompt_index
            if 'pair_index' in renamed_response:
                renamed_response['prompt_index'] = renamed_response.pop('pair_index')
            
            renamed_responses.append(renamed_response)
        
        return renamed_responses
    
    def create_backup(self, file_path: Path) -> Path:
        """
        Create a backup of the original file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to the backup file
        """
        backup_path = file_path.with_suffix(file_path.suffix + self.backup_suffix)
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path.name}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup for {file_path.name}: {e}")
            raise
    
    def save_responses(self, responses: List[Dict], file_path: Path):
        """
        Save responses with renamed field to a JSONL file.
        
        Args:
            responses: List of response dictionaries
            file_path: Path to save the file
        """
        try:
            with jsonlines.open(file_path, 'w') as writer:
                for response in responses:
                    writer.write(response)
            
            logger.debug(f"Saved {len(responses)} responses to {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error saving responses to {file_path}: {e}")
            raise
    
    def rename_field_in_file(self, file_path: Path, create_backup: bool = True) -> bool:
        """
        Rename field in a single response file.
        
        Args:
            file_path: Path to the file to process
            create_backup: Whether to create a backup before modifying
            
        Returns:
            True if renaming was successful
        """
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            # Load responses
            responses = self.load_responses(file_path)
            if not responses:
                return False
            
            # Validate has pair_index field
            if not self.validate_has_pair_index(responses, file_path):
                return False
            
            # Create backup if requested
            if create_backup:
                backup_path = self.create_backup(file_path)
            
            # Rename field in responses
            renamed_responses = self.rename_field_in_responses(responses)
            
            # Save renamed responses
            self.save_responses(renamed_responses, file_path)
            
            if create_backup:
                logger.info(f"Successfully renamed field in {file_path.name} (backup: {backup_path.name})")
            else:
                logger.info(f"Successfully renamed field in {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            return False
    
    def rename_field_in_all(self, dry_run: bool = False, create_backups: bool = True) -> Dict[str, int]:
        """
        Rename field in all response files in the input directory.
        
        Args:
            dry_run: If True, don't actually modify files
            create_backups: Whether to create backup files
            
        Returns:
            Dictionary with processing statistics
        """
        files = self.find_response_files()
        
        if not files:
            logger.warning("No JSONL files found to process")
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
        
        stats = {"total": len(files), "processed": 0, "skipped": 0, "errors": 0}
        
        logger.info(f"Starting field rename of {len(files)} files (dry_run={dry_run})")
        
        for file_path in files:
            if dry_run:
                # For dry run, just validate and log what would be done
                responses = self.load_responses(file_path)
                if responses and self.validate_has_pair_index(responses, file_path):
                    stats["processed"] += 1
                    logger.info(f"DRY RUN: Would rename field in {file_path.name}")
                else:
                    stats["skipped"] += 1
            else:
                # Actually process the file
                if self.rename_field_in_file(file_path, create_backup=create_backups):
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
        
        logger.info(f"Field rename complete: {stats}")
        return stats


def main():
    """Main entry point for the field rename script."""
    parser = argparse.ArgumentParser(
        description='Rename pair_index field to prompt_index in trait response files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to see what would be renamed
    python traits/rename_field.py --input-dir /workspace/traits/responses --dry-run

    # Actually rename the field
    python traits/rename_field.py --input-dir /workspace/traits/responses

    # Rename without creating backups
    python traits/rename_field.py --input-dir /workspace/traits/responses --no-backup
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing JSONL response files to process')
    parser.add_argument('--backup-suffix', type=str, default='.backup',
                       help='Suffix for backup files (default: .backup)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files before renaming')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform a dry run without actually modifying files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print configuration
    logger.info("Field Rename Configuration:")
    logger.info(f"  Input directory: {args.input_dir}")
    logger.info(f"  Backup suffix: {args.backup_suffix}")
    logger.info(f"  Create backups: {not args.no_backup}")
    logger.info(f"  Dry run: {args.dry_run}")
    
    try:
        # Create renamer
        renamer = FieldRenamer(
            input_dir=args.input_dir,
            backup_suffix=args.backup_suffix
        )
        
        # Run field rename
        stats = renamer.rename_field_in_all(
            dry_run=args.dry_run,
            create_backups=not args.no_backup
        )
        
        # Print results
        if args.dry_run:
            logger.info("DRY RUN RESULTS:")
        else:
            logger.info("FIELD RENAME RESULTS:")
        
        logger.info(f"  Total files: {stats['total']}")
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        logger.info(f"  Errors: {stats['errors']}")
        
        if stats['errors'] > 0:
            logger.error("Field rename completed with errors!")
            return 1
        else:
            logger.info("Field rename completed successfully!")
            return 0
        
    except Exception as e:
        logger.error(f"Field rename failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())