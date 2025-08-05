#!/usr/bin/env python3
"""
Migration script to update existing trait response files to new format.

This script adds the 'pair_index' field to existing response files that were
generated with the old format. The old format used only the first instruction
pair (index 0) for each trait.

Usage:
    uv run traits/migrate_responses.py --input-dir /workspace/traits/responses
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


class ResponseMigrator:
    """Migrates old format response files to new format with pair_index field."""
    
    def __init__(self, input_dir: str, backup_suffix: str = ".backup"):
        """
        Initialize the migrator.
        
        Args:
            input_dir: Directory containing JSONL response files
            backup_suffix: Suffix for backup files
        """
        self.input_dir = Path(input_dir)
        self.backup_suffix = backup_suffix
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        logger.info(f"Initialized ResponseMigrator for directory: {input_dir}")
    
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
    
    def validate_old_format(self, responses: List[Dict], file_path: Path) -> bool:
        """
        Validate that responses are in the old format (no pair_index field).
        
        Args:
            responses: List of response dictionaries
            file_path: Path to the file being validated
            
        Returns:
            True if file is in old format and needs migration
        """
        if not responses:
            logger.warning(f"No responses found in {file_path.name}")
            return False
        
        # Check if pair_index field already exists
        if 'pair_index' in responses[0]:
            logger.info(f"File {file_path.name} already has pair_index field, skipping")
            return False
        
        # Validate expected structure
        expected_fields = {'system_prompt', 'label', 'conversation', 'question_index', 'question'}
        first_response_fields = set(responses[0].keys())
        
        if not expected_fields.issubset(first_response_fields):
            missing = expected_fields - first_response_fields
            logger.warning(f"File {file_path.name} missing expected fields: {missing}")
            return False
        
        # Validate expected count (should be 60 for old format: 20 questions × 3 types)
        if len(responses) != 60:
            logger.warning(f"File {file_path.name} has {len(responses)} responses, expected 60")
            return False
        
        # Validate label distribution (should have pos, neg, default)
        labels = [r['label'] for r in responses]
        label_counts = {label: labels.count(label) for label in set(labels)}
        
        expected_labels = {'pos': 20, 'neg': 20, 'default': 20}
        if label_counts != expected_labels:
            logger.warning(f"File {file_path.name} has unexpected label distribution: {label_counts}")
            return False
        
        logger.info(f"File {file_path.name} validated as old format (60 responses)")
        return True
    
    def migrate_responses(self, responses: List[Dict]) -> List[Dict]:
        """
        Migrate responses by adding pair_index field.
        
        Args:
            responses: List of response dictionaries in old format
            
        Returns:
            List of response dictionaries in new format
        """
        migrated_responses = []
        
        for response in responses:
            # Create a copy of the response
            migrated_response = response.copy()
            
            # Add pair_index field (old format used first instruction pair)
            migrated_response['pair_index'] = 0
            
            migrated_responses.append(migrated_response)
        
        return migrated_responses
    
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
        Save migrated responses to a JSONL file.
        
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
    
    def migrate_file(self, file_path: Path) -> bool:
        """
        Migrate a single response file.
        
        Args:
            file_path: Path to the file to migrate
            
        Returns:
            True if migration was successful
        """
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            # Load responses
            responses = self.load_responses(file_path)
            if not responses:
                return False
            
            # Validate old format
            if not self.validate_old_format(responses, file_path):
                return False
            
            # Create backup
            backup_path = self.create_backup(file_path)
            
            # Migrate responses
            migrated_responses = self.migrate_responses(responses)
            
            # Save migrated responses
            self.save_responses(migrated_responses, file_path)
            
            logger.info(f"Successfully migrated {file_path.name} (backup: {backup_path.name})")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating file {file_path.name}: {e}")
            return False
    
    def migrate_all(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Migrate all response files in the input directory.
        
        Args:
            dry_run: If True, don't actually modify files
            
        Returns:
            Dictionary with migration statistics
        """
        files = self.find_response_files()
        
        if not files:
            logger.warning("No JSONL files found to migrate")
            return {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}
        
        stats = {"total": len(files), "migrated": 0, "skipped": 0, "errors": 0}
        
        logger.info(f"Starting migration of {len(files)} files (dry_run={dry_run})")
        
        for file_path in files:
            if dry_run:
                # For dry run, just validate
                responses = self.load_responses(file_path)
                if responses and self.validate_old_format(responses, file_path):
                    stats["migrated"] += 1
                    logger.info(f"DRY RUN: Would migrate {file_path.name}")
                else:
                    stats["skipped"] += 1
            else:
                # Actually migrate
                if self.migrate_file(file_path):
                    stats["migrated"] += 1
                else:
                    stats["errors"] += 1
        
        logger.info(f"Migration complete: {stats}")
        return stats


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description='Migrate trait response files to new format with pair_index field',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to see what would be migrated
    python traits/migrate_responses.py --input-dir /workspace/traits/responses --dry-run

    # Actually migrate files
    python traits/migrate_responses.py --input-dir /workspace/traits/responses

    # Custom backup suffix
    python traits/migrate_responses.py --input-dir /workspace/traits/responses --backup-suffix .old
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing JSONL response files to migrate')
    parser.add_argument('--backup-suffix', type=str, default='.backup',
                       help='Suffix for backup files (default: .backup)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform a dry run without actually modifying files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print configuration
    logger.info("Migration Configuration:")
    logger.info(f"  Input directory: {args.input_dir}")
    logger.info(f"  Backup suffix: {args.backup_suffix}")
    logger.info(f"  Dry run: {args.dry_run}")
    
    try:
        # Create migrator
        migrator = ResponseMigrator(
            input_dir=args.input_dir,
            backup_suffix=args.backup_suffix
        )
        
        # Run migration
        stats = migrator.migrate_all(dry_run=args.dry_run)
        
        # Print results
        if args.dry_run:
            logger.info("DRY RUN RESULTS:")
        else:
            logger.info("MIGRATION RESULTS:")
        
        logger.info(f"  Total files: {stats['total']}")
        logger.info(f"  Migrated: {stats['migrated']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        logger.info(f"  Errors: {stats['errors']}")
        
        if stats['errors'] > 0:
            logger.error("Migration completed with errors!")
            return 1
        else:
            logger.info("Migration completed successfully!")
            return 0
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())