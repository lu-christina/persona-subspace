#!/usr/bin/env python3
"""
Batch submission script for generating role responses using Anthropic Batch API.

This script processes role files and submits batch requests to Anthropic's API
for positive and default instructions using a specified Claude model.

For each role, it submits batch requests with responses to questions using
10 different instruction types:
- pos (5 variants): Using positive role instructions
- default (5 variants): Using default system prompts

Questions can be sourced from either:
1. Individual role files (default behavior) - each role uses its own questions
2. Central questions file (--questions-file) - all roles use the same set of questions

Batch tracking information is saved to {output_dir}/batch_tracking.jsonl for
later retrieval via 2_responses_api_retrieve.py.

Usage:
    # Using central questions file
    uv run roles/2_responses_api_submit.py \
        --model-name claude-sonnet-4-5-20250929 \
        --questions-file /root/git/persona-subspace/roles/data/questions_240.jsonl \
        --output-dir /workspace/roles/responses_api
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import jsonlines

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not found. Please install it:")
    print("  uv add anthropic")
    sys.exit(1)

# Load .env file if available
try:
    from dotenv import load_dotenv
    # Try loading from multiple locations
    loaded = load_dotenv()  # Current directory
    if not loaded:
        load_dotenv(Path.home() / '.env')  # User home directory
except ImportError:
    # python-dotenv not installed, will use environment variables directly
    pass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RoleBatchSubmitter:
    """Submitter for role-based batch requests to Anthropic API."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5-20250929",
        roles_dir: str = "/root/git/persona-subspace/roles/data/instructions",
        output_dir: str = "/workspace/roles/responses_api",
        question_count: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        prompt_indices: Optional[List[int]] = None,
        include_default: bool = True,
        questions_file: Optional[str] = None,
        roles_subset: Optional[Tuple[int, int]] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the batch submitter.

        Args:
            model_name: Claude model identifier (e.g., claude-sonnet-4-5-20250929)
            roles_dir: Directory containing role JSON files
            output_dir: Directory to save batch tracking and response files
            question_count: Number of questions to process per role (default: 20)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            prompt_indices: List of instruction prompt indices to process (None = all prompts)
            include_default: Whether to include default system prompts
            questions_file: Path to central questions JSONL file (if None, use role-specific questions)
            roles_subset: Tuple of (start, end) indices to process subset of roles (if None, process all)
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
        """
        self.model_name = model_name
        self.roles_dir = Path(roles_dir)
        self.output_dir = Path(output_dir)
        self.question_count = question_count
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.prompt_indices = prompt_indices if prompt_indices is not None else list(range(5))
        self.include_default = include_default
        self.questions_file = questions_file
        self.roles_subset = roles_subset

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)

        # Central questions (loaded lazily)
        self.central_questions = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RoleBatchSubmitter with model: {model_name}")
        logger.info(f"Roles directory: {self.roles_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Processing {self.question_count} questions per role")
        if self.questions_file:
            logger.info(f"Using central questions file: {self.questions_file}")
        else:
            logger.info("Using role-specific questions from individual files")

    def load_central_questions(self) -> List[str]:
        """
        Load questions from central JSONL file.

        Returns:
            List of question strings
        """
        if self.central_questions is not None:
            return self.central_questions

        if not self.questions_file:
            raise ValueError("No questions file specified")

        questions_path = Path(self.questions_file)
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")

        logger.info(f"Loading questions from: {questions_path}")

        questions = []
        try:
            with jsonlines.open(questions_path, 'r') as reader:
                for entry in reader:
                    if 'question' not in entry:
                        logger.warning(f"Skipping entry missing 'question' field: {entry}")
                        continue
                    questions.append(entry['question'])

            logger.info(f"Loaded {len(questions)} questions from central file")
            self.central_questions = questions
            return questions

        except Exception as e:
            logger.error(f"Error loading central questions file: {e}")
            raise

    def load_role_files(self) -> Dict[str, Dict]:
        """
        Load all role JSON files from the roles directory.

        Returns:
            Dict mapping role names to their data (instructions and questions)
        """
        role_files = {}

        # Find all JSON files (excluding subdirectories and special files)
        json_files = []
        for file_path in self.roles_dir.iterdir():
            if (file_path.is_file() and
                file_path.suffix == '.json' and
                not file_path.name.startswith('processing_summary') and
                not file_path.name.endswith('.backup') and
                not file_path.parent.name == 'descriptions'):
                json_files.append(file_path)

        logger.info(f"Found {len(json_files)} role files to process")

        # Apply roles subset filtering if specified
        if self.roles_subset:
            start_idx, end_idx = self.roles_subset
            # Sort files to ensure consistent ordering
            json_files = sorted(json_files)
            original_count = len(json_files)
            json_files = json_files[start_idx:end_idx]
            logger.info(f"Applying roles subset [{start_idx}:{end_idx}] - processing {len(json_files)}/{original_count} role files")

        for file_path in json_files:
            role_name = file_path.stem
            try:
                with open(file_path, 'r') as f:
                    role_data = json.load(f)

                # Validate required fields
                required_fields = ['instruction']
                if not self.questions_file:
                    required_fields.append('questions')

                if not all(key in role_data for key in required_fields):
                    logger.warning(f"Skipping {role_name}: missing required fields {required_fields}")
                    continue

                role_files[role_name] = role_data
                logger.debug(f"Loaded role: {role_name}")

            except Exception as e:
                logger.error(f"Error loading role file {file_path}: {e}")
                continue

        logger.info(f"Successfully loaded {len(role_files)} role files")
        return role_files

    def format_messages(self, instruction: Optional[str], question: str) -> List[Dict[str, str]]:
        """
        Format messages for Anthropic API.

        Args:
            instruction: Optional system instruction (None for default/baseline)
            question: Question text

        Returns:
            List of message dictionaries and optional system string
        """
        messages = [{"role": "user", "content": question}]
        system = instruction if instruction else None
        return messages, system

    def create_batch_requests_for_role(self, role_name: str, role_data: Dict) -> List[Dict]:
        """
        Create batch requests for a single role.

        Args:
            role_name: Name of the role
            role_data: Role data containing instructions and questions

        Returns:
            List of batch request dictionaries
        """
        instructions = role_data['instruction']

        # Get questions
        if self.questions_file:
            central_questions = self.load_central_questions()
            questions = central_questions[:self.question_count]
        else:
            questions = role_data['questions'][:self.question_count]

        logger.info(f"Creating batch requests for role '{role_name}' with {len(questions)} questions")

        # Extract all positive instructions
        all_pos_instructions = []
        for inst in instructions:
            if 'pos' in inst:
                all_pos_instructions.append(inst['pos'])

        if not all_pos_instructions:
            logger.error(f"Missing pos instructions for role {role_name}")
            return []

        pos_instructions = all_pos_instructions

        # Define default system prompts
        default_instructions = [
            "",
            "You are an AI assistant.",
            "You are a large language model.",
            "You are Gemma.",
            "Respond as yourself."
        ] if self.include_default else []

        # Prepare batch requests
        batch_requests = []

        # Positive instruction requests (selected variants)
        for prompt_idx in self.prompt_indices:
            if prompt_idx < len(pos_instructions):
                pos_instruction = pos_instructions[prompt_idx]
                for q_idx, question in enumerate(questions):
                    messages, system = self.format_messages(pos_instruction, question)

                    # Create custom_id for tracking
                    custom_id = f"{role_name}___pos___{prompt_idx}___{q_idx}"

                    # Build request params
                    # Note: Claude Sonnet 4.5 doesn't allow both temperature and top_p
                    params = {
                        "model": self.model_name,
                        "max_tokens": self.max_tokens,
                        "messages": messages,
                        "temperature": self.temperature
                    }

                    if system:
                        params["system"] = system

                    batch_requests.append({
                        "custom_id": custom_id,
                        "params": params
                    })

        # Default instruction requests (selected variants)
        for prompt_idx in self.prompt_indices:
            if prompt_idx < len(default_instructions):
                default_instruction = default_instructions[prompt_idx]
                for q_idx, question in enumerate(questions):
                    messages, system = self.format_messages(default_instruction, question)

                    # Create custom_id for tracking
                    custom_id = f"{role_name}___default___{prompt_idx}___{q_idx}"

                    # Build request params
                    # Note: Claude Sonnet 4.5 doesn't allow both temperature and top_p
                    params = {
                        "model": self.model_name,
                        "max_tokens": self.max_tokens,
                        "messages": messages,
                        "temperature": self.temperature
                    }

                    if system:
                        params["system"] = system

                    batch_requests.append({
                        "custom_id": custom_id,
                        "params": params
                    })

        logger.info(f"Created {len(batch_requests)} batch requests for role '{role_name}'")
        return batch_requests

    def submit_batch_for_role(self, role_name: str, batch_requests: List[Dict]) -> Optional[Dict]:
        """
        Submit a batch to Anthropic API for a single role.

        Args:
            role_name: Name of the role
            batch_requests: List of batch request dictionaries

        Returns:
            Batch tracking information dictionary, or None if submission failed
        """
        if not batch_requests:
            logger.warning(f"No requests to submit for role '{role_name}'")
            return None

        try:
            logger.info(f"Submitting batch for role '{role_name}' with {len(batch_requests)} requests...")

            # Submit batch to Anthropic API
            message_batch = self.client.messages.batches.create(requests=batch_requests)

            logger.info(f"Batch submitted successfully for role '{role_name}'")
            logger.info(f"  Batch ID: {message_batch.id}")
            logger.info(f"  Status: {message_batch.processing_status}")

            # Create tracking entry (convert datetime objects to ISO strings for JSON serialization)
            tracking_entry = {
                "batch_id": message_batch.id,
                "role_name": role_name,
                "created_at": message_batch.created_at.isoformat() if message_batch.created_at else None,
                "expires_at": message_batch.expires_at.isoformat() if message_batch.expires_at else None,
                "status": message_batch.processing_status,
                "request_count": len(batch_requests),
                "output_file": str(self.output_dir / f"{role_name}.jsonl"),
                "model_name": self.model_name,
                "question_count": self.question_count,
                "prompt_indices": self.prompt_indices
            }

            return tracking_entry

        except Exception as e:
            logger.error(f"Error submitting batch for role '{role_name}': {e}")
            return None

    def save_batch_tracking(self, tracking_entry: Dict):
        """
        Save batch tracking information to tracking file.

        Args:
            tracking_entry: Tracking information dictionary
        """
        tracking_file = self.output_dir / "batch_tracking.jsonl"

        try:
            with jsonlines.open(tracking_file, mode='a') as writer:
                writer.write(tracking_entry)

            logger.info(f"Saved batch tracking for {tracking_entry['role_name']} to {tracking_file}")

        except Exception as e:
            logger.error(f"Error saving batch tracking: {e}")

    def should_skip_role(self, role_name: str) -> bool:
        """
        Check if role should be skipped (already has pending/completed batch).

        Args:
            role_name: Name of the role

        Returns:
            True if role should be skipped
        """
        tracking_file = self.output_dir / "batch_tracking.jsonl"

        if not tracking_file.exists():
            return False

        try:
            with jsonlines.open(tracking_file, 'r') as reader:
                for entry in reader:
                    if entry['role_name'] == role_name:
                        status = entry.get('status', 'unknown')
                        if status in ['in_progress', 'ended']:
                            logger.info(f"Skipping role '{role_name}' (batch {entry['batch_id']} status: {status})")
                            return True
            return False

        except Exception as e:
            logger.error(f"Error reading batch tracking file: {e}")
            return False

    def process_all_roles(self, skip_existing: bool = True):
        """
        Process all roles and submit batches.

        Args:
            skip_existing: Skip roles with existing batches in tracking file
        """
        # Load role files
        role_files = self.load_role_files()

        if not role_files:
            logger.error("No role files found to process")
            return

        # Filter roles if skipping existing
        if skip_existing:
            roles_to_process = {}
            for role_name, role_data in role_files.items():
                if self.should_skip_role(role_name):
                    continue
                roles_to_process[role_name] = role_data
        else:
            roles_to_process = role_files

        logger.info(f"Processing {len(roles_to_process)} roles")

        # Process each role
        submitted_count = 0
        failed_count = 0

        for i, (role_name, role_data) in enumerate(roles_to_process.items(), 1):
            logger.info(f"\nProcessing role {i}/{len(roles_to_process)}: {role_name}")

            try:
                # Create batch requests
                batch_requests = self.create_batch_requests_for_role(role_name, role_data)

                if not batch_requests:
                    logger.warning(f"No batch requests created for role '{role_name}'")
                    failed_count += 1
                    continue

                # Submit batch
                tracking_entry = self.submit_batch_for_role(role_name, batch_requests)

                if tracking_entry:
                    # Save tracking information
                    self.save_batch_tracking(tracking_entry)
                    submitted_count += 1
                    logger.info(f"Successfully submitted batch for role '{role_name}'")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to submit batch for role '{role_name}'")

            except Exception as e:
                logger.error(f"Error processing role '{role_name}': {e}")
                failed_count += 1
                continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch submission completed!")
        logger.info(f"  Submitted: {submitted_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Total: {len(roles_to_process)}")
        logger.info(f"\nBatch tracking saved to: {self.output_dir / 'batch_tracking.jsonl'}")
        logger.info(f"Use 2_responses_api_retrieve.py to check status and retrieve results")
        logger.info(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Submit role response batches to Anthropic API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python roles/2_responses_api_submit.py

    # Custom model and directories
    python roles/2_responses_api_submit.py \\
        --model-name claude-sonnet-4-5-20250929 \\
        --roles-dir /path/to/roles \\
        --output-dir /path/to/output

    # Custom generation parameters
    python roles/2_responses_api_submit.py \\
        --temperature 0.8 \\
        --max-tokens 1024 \\
        --question-count 10
        """
    )

    # Model configuration
    parser.add_argument('--model-name', type=str, default='claude-sonnet-4-5-20250929',
                       help='Claude model name (default: claude-sonnet-4-5-20250929)')
    parser.add_argument('--roles-dir', type=str, default='/root/git/persona-subspace/roles/data/instructions',
                       help='Directory containing role JSON files')
    parser.add_argument('--output-dir', type=str, default='/workspace/roles/responses_api',
                       help='Output directory for batch tracking and JSONL files')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Anthropic API key (default: uses ANTHROPIC_API_KEY env var)')

    # Generation parameters
    parser.add_argument('--question-count', type=int, default=240,
                       help='Number of questions to process per role (default: 240)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter (default: 0.9)')

    # Instruction selection parameters
    parser.add_argument('--prompt-indices', type=str, default=None,
                       help='Comma-separated list of instruction prompt indices to process (e.g., "1,2,3,4")')
    parser.add_argument('--include-default', action='store_true', default=True,
                       help='Include default system prompts in addition to positive prompts (default: True)')
    parser.add_argument('--no-default', action='store_false', dest='include_default',
                       help='Disable default system prompts')

    # Questions source configuration
    parser.add_argument('--questions-file', type=str, default=None,
                       help='Path to central questions JSONL file (if not specified, uses role-specific questions)')

    # Role selection parameters
    parser.add_argument('--roles-subset', type=str, default=None,
                       help='Process subset of roles by index range (e.g., "0-120" or "121-240")')

    # Optional flags
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process all roles, even if batches already submitted')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse prompt indices
    prompt_indices = None
    if args.prompt_indices:
        try:
            prompt_indices = [int(x.strip()) for x in args.prompt_indices.split(',')]
            logger.info(f"Using specific prompt indices: {prompt_indices}")
        except ValueError as e:
            logger.error(f"Invalid prompt indices format: {args.prompt_indices}")
            return 1

    # Parse roles subset
    roles_subset = None
    if args.roles_subset:
        try:
            if '-' in args.roles_subset:
                start, end = args.roles_subset.split('-')
                roles_subset = (int(start.strip()), int(end.strip()))
                logger.info(f"Using roles subset: {roles_subset[0]} to {roles_subset[1]}")
            else:
                logger.error(f"Invalid roles subset format: {args.roles_subset}. Use format like '0-120'")
                return 1
        except ValueError as e:
            logger.error(f"Invalid roles subset format: {args.roles_subset}")
            return 1

    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Roles directory: {args.roles_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Question count: {args.question_count}")
    logger.info(f"  Questions file: {args.questions_file if args.questions_file else 'role-specific'}")
    logger.info(f"  Roles subset: {roles_subset if roles_subset else 'all roles'}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Prompt indices: {prompt_indices if prompt_indices else 'all (0-4)'}")
    logger.info(f"  Include default: {args.include_default}")
    logger.info(f"  Skip existing: {not args.no_skip_existing}")

    try:
        # Create submitter
        submitter = RoleBatchSubmitter(
            model_name=args.model_name,
            roles_dir=args.roles_dir,
            output_dir=args.output_dir,
            question_count=args.question_count,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            prompt_indices=prompt_indices,
            include_default=args.include_default,
            questions_file=args.questions_file,
            roles_subset=roles_subset,
            api_key=args.api_key
        )

        # Process all roles
        submitter.process_all_roles(skip_existing=not args.no_skip_existing)

        logger.info("\nBatch submission completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
