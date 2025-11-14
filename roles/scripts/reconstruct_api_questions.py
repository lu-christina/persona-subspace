#!/usr/bin/env python3
"""
Reconstruct missing questions in batch API response files.

This script reads JSONL response files and fills in missing 'question' fields
by looking up the question from the questions file using the question_index.

It also validates that prompt_index fields are present and reports any issues.

Usage:
    # Fix a single role file
    python roles/reconstruct_api_questions.py \
        --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
        --input-file /workspace/sonnet-4.5/roles_240/responses/astronaut.jsonl

    # Fix all files in a directory
    python roles/reconstruct_api_questions.py \
        --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
        --input-dir /workspace/sonnet-4.5/roles_240/responses

    # Dry run (don't modify files)
    python roles/reconstruct_api_questions.py \
        --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
        --input-dir /workspace/sonnet-4.5/roles_240/responses \
        --dry-run
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import jsonlines

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuestionReconstructor:
    """Reconstructs missing questions in batch API response files."""

    def __init__(self, questions_file: str, dry_run: bool = False):
        """
        Initialize the reconstructor.

        Args:
            questions_file: Path to questions JSONL file
            dry_run: If True, don't modify files
        """
        self.questions_file = Path(questions_file)
        self.dry_run = dry_run
        self.questions = None

        logger.info(f"Initialized QuestionReconstructor")
        logger.info(f"Questions file: {self.questions_file}")
        logger.info(f"Dry run: {self.dry_run}")

    def load_questions(self) -> List[str]:
        """
        Load questions from JSONL file.

        Returns:
            List of question strings
        """
        if self.questions is not None:
            return self.questions

        if not self.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")

        logger.info(f"Loading questions from: {self.questions_file}")

        questions = []
        try:
            with jsonlines.open(self.questions_file, 'r') as reader:
                for entry in reader:
                    if 'question' not in entry:
                        logger.warning(f"Skipping entry missing 'question' field: {entry}")
                        continue
                    questions.append(entry['question'])

            logger.info(f"Loaded {len(questions)} questions")
            self.questions = questions
            return questions

        except Exception as e:
            logger.error(f"Error loading questions file: {e}")
            raise

    def reconstruct_file(self, input_file: Path) -> Dict:
        """
        Reconstruct questions in a single JSONL file.

        Args:
            input_file: Path to input JSONL file

        Returns:
            Dictionary with statistics about the reconstruction
        """
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return {"error": "file_not_found"}

        logger.info(f"\nProcessing file: {input_file.name}")

        # Load questions
        questions = self.load_questions()

        # Load entries from file
        entries = []
        try:
            with jsonlines.open(input_file, 'r') as reader:
                for entry in reader:
                    entries.append(entry)
        except Exception as e:
            logger.error(f"Error reading file {input_file}: {e}")
            return {"error": str(e)}

        logger.info(f"  Loaded {len(entries)} entries")

        # Statistics
        stats = {
            "total": len(entries),
            "missing_question": 0,
            "missing_question_index": 0,
            "missing_prompt_index": 0,
            "reconstructed": 0,
            "errors": 0,
            "question_index_out_of_range": 0
        }

        # Process each entry
        for i, entry in enumerate(entries):
            # Check for prompt_index
            if 'prompt_index' not in entry:
                stats["missing_prompt_index"] += 1
                logger.warning(f"  Entry {i}: Missing prompt_index")

            # Check for question_index
            if 'question_index' not in entry:
                stats["missing_question_index"] += 1
                logger.warning(f"  Entry {i}: Missing question_index")
                continue

            question_idx = entry['question_index']

            # Check if question_index is in range
            if question_idx >= len(questions):
                stats["question_index_out_of_range"] += 1
                logger.warning(f"  Entry {i}: question_index {question_idx} out of range (max: {len(questions)-1})")
                continue

            # Check if question is missing or empty
            current_question = entry.get('question', '')
            if not current_question or current_question.strip() == '':
                stats["missing_question"] += 1

                # Reconstruct question
                try:
                    correct_question = questions[question_idx]
                    entry['question'] = correct_question

                    # Also update in conversation if present
                    if 'conversation' in entry:
                        for msg in entry['conversation']:
                            if msg.get('role') == 'user':
                                msg['content'] = correct_question
                                break

                    stats["reconstructed"] += 1
                    logger.debug(f"  Entry {i}: Reconstructed question for index {question_idx}")

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"  Entry {i}: Error reconstructing question: {e}")

        # Write back if not dry run
        if not self.dry_run and stats["reconstructed"] > 0:
            try:
                # Create backup
                backup_file = input_file.with_suffix('.jsonl.backup')
                if backup_file.exists():
                    logger.warning(f"  Backup file already exists: {backup_file}")
                else:
                    import shutil
                    shutil.copy2(input_file, backup_file)
                    logger.info(f"  Created backup: {backup_file.name}")

                # Write updated entries
                with jsonlines.open(input_file, mode='w') as writer:
                    for entry in entries:
                        writer.write(entry)

                logger.info(f"  ✓ Updated file: {input_file.name}")

            except Exception as e:
                logger.error(f"  Error writing file {input_file}: {e}")
                stats["errors"] += 1
                return stats

        # Print summary
        logger.info(f"\n  Summary for {input_file.name}:")
        logger.info(f"    Total entries: {stats['total']}")
        logger.info(f"    Missing questions: {stats['missing_question']}")
        logger.info(f"    Reconstructed: {stats['reconstructed']}")

        if stats["missing_question_index"] > 0:
            logger.warning(f"    Missing question_index: {stats['missing_question_index']}")
        if stats["missing_prompt_index"] > 0:
            logger.warning(f"    Missing prompt_index: {stats['missing_prompt_index']}")
        if stats["question_index_out_of_range"] > 0:
            logger.warning(f"    Question index out of range: {stats['question_index_out_of_range']}")
        if stats["errors"] > 0:
            logger.error(f"    Errors: {stats['errors']}")

        return stats

    def reconstruct_directory(self, input_dir: Path) -> Dict:
        """
        Reconstruct questions in all JSONL files in a directory.

        Args:
            input_dir: Path to directory containing JSONL files

        Returns:
            Dictionary with aggregate statistics
        """
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return {"error": "directory_not_found"}

        # Find all JSONL files (excluding backups and tracking)
        jsonl_files = []
        for file_path in input_dir.iterdir():
            if (file_path.is_file() and
                file_path.suffix == '.jsonl' and
                not file_path.name.endswith('.backup') and
                not file_path.name == 'batch_tracking.jsonl'):
                jsonl_files.append(file_path)

        logger.info(f"Found {len(jsonl_files)} JSONL files to process")

        if not jsonl_files:
            logger.warning("No JSONL files found to process")
            return {"error": "no_files_found"}

        # Process each file
        aggregate_stats = {
            "files_processed": 0,
            "files_with_errors": 0,
            "total_entries": 0,
            "total_missing_question": 0,
            "total_reconstructed": 0,
            "total_missing_question_index": 0,
            "total_missing_prompt_index": 0,
            "total_errors": 0
        }

        for file_path in sorted(jsonl_files):
            stats = self.reconstruct_file(file_path)

            if "error" in stats:
                aggregate_stats["files_with_errors"] += 1
                continue

            aggregate_stats["files_processed"] += 1
            aggregate_stats["total_entries"] += stats["total"]
            aggregate_stats["total_missing_question"] += stats["missing_question"]
            aggregate_stats["total_reconstructed"] += stats["reconstructed"]
            aggregate_stats["total_missing_question_index"] += stats["missing_question_index"]
            aggregate_stats["total_missing_prompt_index"] += stats["missing_prompt_index"]
            aggregate_stats["total_errors"] += stats["errors"]

        # Print overall summary
        logger.info("\n" + "="*60)
        logger.info("OVERALL SUMMARY")
        logger.info("="*60)
        logger.info(f"Files processed: {aggregate_stats['files_processed']}")
        logger.info(f"Files with errors: {aggregate_stats['files_with_errors']}")
        logger.info(f"Total entries: {aggregate_stats['total_entries']}")
        logger.info(f"Total missing questions: {aggregate_stats['total_missing_question']}")
        logger.info(f"Total reconstructed: {aggregate_stats['total_reconstructed']}")

        if aggregate_stats["total_missing_question_index"] > 0:
            logger.warning(f"Total missing question_index: {aggregate_stats['total_missing_question_index']}")
        if aggregate_stats["total_missing_prompt_index"] > 0:
            logger.warning(f"Total missing prompt_index: {aggregate_stats['total_missing_prompt_index']}")
        if aggregate_stats["total_errors"] > 0:
            logger.error(f"Total errors: {aggregate_stats['total_errors']}")

        if self.dry_run:
            logger.info("\n** DRY RUN - No files were modified **")
        else:
            logger.info(f"\n✓ Successfully updated {aggregate_stats['files_processed']} files")

        logger.info("="*60)

        return aggregate_stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Reconstruct missing questions in batch API response files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix a single role file
    python roles/reconstruct_api_questions.py \\
        --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \\
        --input-file /workspace/sonnet-4.5/roles_240/responses/astronaut.jsonl

    # Fix all files in a directory
    python roles/reconstruct_api_questions.py \\
        --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \\
        --input-dir /workspace/sonnet-4.5/roles_240/responses

    # Dry run to see what would be changed
    python roles/reconstruct_api_questions.py \\
        --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \\
        --input-dir /workspace/sonnet-4.5/roles_240/responses \\
        --dry-run
        """
    )

    # Required arguments
    parser.add_argument('--questions-file', type=str, required=True,
                       help='Path to questions JSONL file')

    # Input source (one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-file', type=str,
                            help='Path to single JSONL file to process')
    input_group.add_argument('--input-dir', type=str,
                            help='Path to directory containing JSONL files to process')

    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without modifying files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create reconstructor
    reconstructor = QuestionReconstructor(
        questions_file=args.questions_file,
        dry_run=args.dry_run
    )

    try:
        if args.input_file:
            # Process single file
            stats = reconstructor.reconstruct_file(Path(args.input_file))
            if "error" in stats:
                return 1
        else:
            # Process directory
            stats = reconstructor.reconstruct_directory(Path(args.input_dir))
            if "error" in stats:
                return 1

        logger.info("\nReconstruction completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
