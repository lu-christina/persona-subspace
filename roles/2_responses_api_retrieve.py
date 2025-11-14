#!/usr/bin/env python3
"""
Batch retrieval script for fetching role responses from Anthropic Batch API.

This script polls batch status and retrieves completed results from Anthropic's API
for batches submitted via 2_responses_api_submit.py.

It reads batch tracking information from {output_dir}/batch_tracking.jsonl,
checks the status of each batch, and downloads completed results.

Results are saved in the same JSONL format as the vLLM version for compatibility.

Usage:
    # Check status only (no retrieval)
    uv run roles/2_responses_api_retrieve.py \
        --output-dir /workspace/roles/responses_api \
        --check-only

    # Retrieve all completed batches (single pass)
    uv run roles/2_responses_api_retrieve.py \
        --output-dir /workspace/roles/responses_api

    # Wait and poll until all batches complete
    uv run roles/2_responses_api_retrieve.py \
        --output-dir /workspace/roles/responses_api \
        --wait \
        --poll-interval 60
"""

import argparse
import json
import logging
import os
import sys
import time
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


class RoleBatchRetriever:
    """Retriever for role-based batch results from Anthropic API."""

    def __init__(
        self,
        output_dir: str = "/workspace/roles/responses_api",
        api_key: Optional[str] = None
    ):
        """
        Initialize the batch retriever.

        Args:
            output_dir: Directory containing batch tracking and for saving response files
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
        """
        self.output_dir = Path(output_dir)
        self.tracking_file = self.output_dir / "batch_tracking.jsonl"

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)

        logger.info(f"Initialized RoleBatchRetriever")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Tracking file: {self.tracking_file}")

    def load_tracking_entries(self) -> List[Dict]:
        """
        Load all batch tracking entries.

        Returns:
            List of tracking entry dictionaries
        """
        if not self.tracking_file.exists():
            logger.warning(f"Tracking file not found: {self.tracking_file}")
            return []

        entries = []
        try:
            with jsonlines.open(self.tracking_file, 'r') as reader:
                for entry in reader:
                    entries.append(entry)

            logger.info(f"Loaded {len(entries)} tracking entries")
            return entries

        except Exception as e:
            logger.error(f"Error loading tracking file: {e}")
            return []

    def update_tracking_entry(self, batch_id: str, updates: Dict):
        """
        Update a tracking entry with new information.

        Args:
            batch_id: Batch ID to update
            updates: Dictionary of fields to update
        """
        # Load all entries
        entries = self.load_tracking_entries()

        # Update matching entry
        updated = False
        for entry in entries:
            if entry['batch_id'] == batch_id:
                entry.update(updates)
                updated = True
                break

        if not updated:
            logger.warning(f"Batch ID {batch_id} not found in tracking file")
            return

        # Write back all entries
        try:
            with jsonlines.open(self.tracking_file, mode='w') as writer:
                for entry in entries:
                    writer.write(entry)

            logger.debug(f"Updated tracking entry for batch {batch_id}")

        except Exception as e:
            logger.error(f"Error updating tracking file: {e}")

    def check_batch_status(self, batch_id: str) -> Optional[Dict]:
        """
        Check the status of a batch.

        Args:
            batch_id: Batch ID to check

        Returns:
            Batch status information dictionary, or None if error
        """
        try:
            message_batch = self.client.messages.batches.retrieve(batch_id)

            # Convert datetime objects to ISO strings for JSON serialization
            status_info = {
                "batch_id": message_batch.id,
                "status": message_batch.processing_status,
                "request_counts": {
                    "processing": message_batch.request_counts.processing,
                    "succeeded": message_batch.request_counts.succeeded,
                    "errored": message_batch.request_counts.errored,
                    "canceled": message_batch.request_counts.canceled,
                    "expired": message_batch.request_counts.expired
                },
                "created_at": message_batch.created_at.isoformat() if message_batch.created_at else None,
                "expires_at": message_batch.expires_at.isoformat() if message_batch.expires_at else None,
                "ended_at": getattr(message_batch, 'ended_at', None).isoformat() if getattr(message_batch, 'ended_at', None) else None,
                "results_url": message_batch.results_url
            }

            return status_info

        except Exception as e:
            logger.error(f"Error checking batch status for {batch_id}: {e}")
            return None

    def retrieve_batch_results(self, batch_id: str, role_name: str, tracking_entry: Dict) -> bool:
        """
        Retrieve and save results for a completed batch.

        Args:
            batch_id: Batch ID to retrieve
            role_name: Role name for output file
            tracking_entry: Original tracking entry with metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Retrieving results for batch {batch_id} (role: {role_name})...")

            # Stream results from Anthropic API
            results = []
            succeeded_count = 0
            errored_count = 0
            first_error_logged = False  # Log first error for debugging

            for result in self.client.messages.batches.results(batch_id):
                custom_id = result.custom_id

                if result.result.type == "succeeded":
                    message = result.result.message
                    succeeded_count += 1

                    # Parse custom_id to extract metadata
                    # Format: {role_name}____{label}___{prompt_idx}___{question_idx}
                    parts = custom_id.split('___')
                    if len(parts) == 4:
                        _, label, prompt_idx_str, question_idx_str = parts
                        prompt_idx = int(prompt_idx_str)
                        question_idx = int(question_idx_str)
                    else:
                        logger.warning(f"Unexpected custom_id format: {custom_id}")
                        continue

                    # Get system prompt from request (if available)
                    system_prompt = ""
                    if message.model:
                        # Try to extract from the message (Anthropic doesn't return system in response)
                        # We'll need to reconstruct from our tracking data
                        system_prompt = self.reconstruct_system_prompt(
                            role_name, label, prompt_idx, tracking_entry
                        )

                    # Get the question from messages
                    question = ""
                    if len(message.content) > 0 and hasattr(result, 'custom_id'):
                        # The question was in the request, not the response
                        # We'll need to look it up from role files or tracking
                        question = self.reconstruct_question(
                            role_name, question_idx, tracking_entry
                        )

                    # Get the assistant's response
                    assistant_response = message.content[0].text if message.content else ""

                    # Build conversation format (matching vLLM output)
                    conversation = []

                    # Add system message if present
                    if system_prompt:
                        conversation.append({"role": "system", "content": system_prompt})

                    # Add user message (question)
                    conversation.append({"role": "user", "content": question})

                    # Add assistant response
                    conversation.append({"role": "assistant", "content": assistant_response})

                    # Create result object (matching vLLM format)
                    result_obj = {
                        "system_prompt": system_prompt,
                        "label": label,
                        "prompt_index": prompt_idx,
                        "conversation": conversation,
                        "question_index": question_idx,
                        "question": question
                    }

                    results.append(result_obj)

                elif result.result.type == "errored":
                    errored_count += 1
                    error = result.result.error
                    error_message = getattr(error, 'message', str(error))
                    error_type = getattr(error, 'type', 'unknown')

                    # Log first error in detail for debugging
                    if not first_error_logged:
                        logger.error(f"FIRST ERROR - Request {custom_id} failed:")
                        logger.error(f"  Error type: {error_type}")
                        logger.error(f"  Error message: {error_message}")
                        logger.error(f"  Full error object: {error}")
                        first_error_logged = True
                    else:
                        logger.debug(f"Request {custom_id} failed: {error_type} - {error_message}")

                elif result.result.type == "canceled":
                    logger.warning(f"Request {custom_id} was canceled")

                elif result.result.type == "expired":
                    logger.warning(f"Request {custom_id} expired")

            logger.info(f"Retrieved {len(results)} successful results ({errored_count} errors)")

            # Save results to JSONL file
            if results:
                output_file = Path(tracking_entry['output_file'])
                with jsonlines.open(output_file, mode='w') as writer:
                    for result in results:
                        writer.write(result)

                logger.info(f"Saved {len(results)} results to {output_file}")

            # Update tracking entry
            self.update_tracking_entry(batch_id, {
                "status": "completed",
                "retrieved_at": datetime.now().isoformat(),
                "results_count": len(results),
                "succeeded_count": succeeded_count,
                "errored_count": errored_count
            })

            return True

        except Exception as e:
            import traceback
            logger.error(f"Error retrieving results for batch {batch_id}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    def reconstruct_system_prompt(
        self, role_name: str, label: str, prompt_idx: int, tracking_entry: Dict
    ) -> str:
        """
        Reconstruct the system prompt used in the request.

        Since Anthropic doesn't return the system prompt in results,
        we need to reconstruct it from the role files.

        Args:
            role_name: Role name
            label: Instruction label ('pos' or 'default')
            prompt_idx: Prompt index
            tracking_entry: Tracking entry with metadata

        Returns:
            System prompt string
        """
        # Default system prompts
        default_instructions = [
            "",
            "You are an AI assistant.",
            "You are a large language model.",
            "You are Gemma.",
            "Respond as yourself."
        ]

        if label == "default":
            if prompt_idx < len(default_instructions):
                return default_instructions[prompt_idx]
            return ""

        # For 'pos' labels, need to load from role file
        try:
            # Get roles_dir from config or use default
            roles_dir = Path("/root/git/persona-subspace/roles/data/instructions")
            role_file = roles_dir / f"{role_name}.json"

            if not role_file.exists():
                logger.warning(f"Role file not found: {role_file}")
                return ""

            with open(role_file, 'r') as f:
                role_data = json.load(f)

            instructions = role_data.get('instruction', [])

            # Extract positive instructions
            pos_instructions = []
            for inst in instructions:
                if 'pos' in inst:
                    pos_instructions.append(inst['pos'])

            if prompt_idx < len(pos_instructions):
                return pos_instructions[prompt_idx]

            return ""

        except Exception as e:
            logger.error(f"Error reconstructing system prompt: {e}")
            return ""

    def reconstruct_question(
        self, role_name: str, question_idx: int, tracking_entry: Dict
    ) -> str:
        """
        Reconstruct the question used in the request.

        Args:
            role_name: Role name
            question_idx: Question index
            tracking_entry: Tracking entry with metadata

        Returns:
            Question string
        """
        try:
            # Check if using central questions file
            questions_file = tracking_entry.get('questions_file')

            if questions_file:
                # Load from central questions file
                with jsonlines.open(questions_file, 'r') as reader:
                    questions = [entry['question'] for entry in reader if 'question' in entry]
            else:
                # Load from role file
                roles_dir = Path("/root/git/persona-subspace/roles/data/instructions")
                role_file = roles_dir / f"{role_name}.json"

                if not role_file.exists():
                    logger.warning(f"Role file not found: {role_file}")
                    return ""

                with open(role_file, 'r') as f:
                    role_data = json.load(f)

                questions = role_data.get('questions', [])

            if question_idx < len(questions):
                return questions[question_idx]

            return ""

        except Exception as e:
            logger.error(f"Error reconstructing question: {e}")
            return ""

    def check_all_batches(self) -> Dict[str, List[Dict]]:
        """
        Check status of all batches in tracking file.

        Returns:
            Dictionary categorizing batches by status
        """
        entries = self.load_tracking_entries()

        if not entries:
            logger.warning("No tracking entries found")
            return {}

        status_summary = {
            "in_progress": [],
            "ended": [],
            "completed": [],
            "errored": [],
            "unknown": []
        }

        logger.info(f"\nChecking status of {len(entries)} batches...\n")

        for entry in entries:
            batch_id = entry['batch_id']
            role_name = entry['role_name']

            # Skip already completed retrievals
            if entry.get('status') == 'completed' and 'retrieved_at' in entry:
                logger.info(f"✓ {role_name}: Already retrieved (batch {batch_id})")
                status_summary["completed"].append(entry)
                continue

            # Check current status
            status_info = self.check_batch_status(batch_id)

            if not status_info:
                logger.warning(f"✗ {role_name}: Failed to check status (batch {batch_id})")
                status_summary["unknown"].append(entry)
                continue

            status = status_info['status']
            counts = status_info['request_counts']

            # Update tracking with current status
            self.update_tracking_entry(batch_id, {"status": status})

            # Log status
            if status == "in_progress":
                logger.info(f"⏳ {role_name}: Processing ({counts['succeeded']}/{entry['request_count']} done)")
                status_summary["in_progress"].append(entry)
            elif status == "ended":
                logger.info(f"✓ {role_name}: Complete ({counts['succeeded']} succeeded, {counts['errored']} errored)")
                status_summary["ended"].append(entry)
            else:
                logger.info(f"? {role_name}: Status {status}")
                status_summary["errored"].append(entry)

        return status_summary

    def retrieve_all_completed(self) -> Tuple[int, int]:
        """
        Retrieve results for all completed batches.

        Returns:
            Tuple of (succeeded_count, failed_count)
        """
        entries = self.load_tracking_entries()

        if not entries:
            logger.warning("No tracking entries found")
            return 0, 0

        succeeded_count = 0
        failed_count = 0

        logger.info(f"\nRetrieving completed batches...\n")

        for entry in entries:
            batch_id = entry['batch_id']
            role_name = entry['role_name']
            status = entry.get('status', 'unknown')

            # Skip if not ended or already retrieved
            if status == 'completed' and 'retrieved_at' in entry:
                logger.debug(f"Skipping {role_name} (already retrieved)")
                continue

            if status != 'ended':
                # Check current status
                status_info = self.check_batch_status(batch_id)
                if not status_info or status_info['status'] != 'ended':
                    logger.debug(f"Skipping {role_name} (not ready)")
                    continue

            # Retrieve results
            logger.info(f"Retrieving {role_name}...")
            success = self.retrieve_batch_results(batch_id, role_name, entry)

            if success:
                succeeded_count += 1
            else:
                failed_count += 1

        return succeeded_count, failed_count

    def wait_for_all_batches(self, poll_interval: int = 60, max_wait: int = 86400):
        """
        Wait for all batches to complete, polling periodically.

        Args:
            poll_interval: Seconds between status checks
            max_wait: Maximum time to wait in seconds (default: 24 hours)
        """
        start_time = time.time()

        logger.info(f"Waiting for all batches to complete (polling every {poll_interval}s)...\n")

        while True:
            elapsed = time.time() - start_time

            if elapsed > max_wait:
                logger.warning(f"Reached maximum wait time ({max_wait}s)")
                break

            # Check all batch statuses
            status_summary = self.check_all_batches()

            in_progress = len(status_summary.get("in_progress", []))
            ended = len(status_summary.get("ended", []))
            completed = len(status_summary.get("completed", []))

            logger.info(f"\nStatus: {in_progress} in progress, {ended} ended, {completed} retrieved")

            # If nothing in progress, we're done
            if in_progress == 0:
                logger.info("All batches completed!")

                # Retrieve any newly completed batches
                if ended > 0:
                    logger.info("\nRetrieving newly completed batches...")
                    self.retrieve_all_completed()

                break

            # Wait before next check
            logger.info(f"Waiting {poll_interval}s before next check...")
            time.sleep(poll_interval)

        logger.info("\nPolling complete!")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Retrieve role response batches from Anthropic API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check status only
    python roles/2_responses_api_retrieve.py --check-only

    # Retrieve completed batches (single pass)
    python roles/2_responses_api_retrieve.py

    # Wait and poll until all complete
    python roles/2_responses_api_retrieve.py --wait --poll-interval 60
        """
    )

    # Configuration
    parser.add_argument('--output-dir', type=str, default='/workspace/roles/responses_api',
                       help='Output directory containing batch tracking file')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Anthropic API key (default: uses ANTHROPIC_API_KEY env var)')

    # Operation modes
    parser.add_argument('--check-only', action='store_true',
                       help='Only check batch status, do not retrieve results')
    parser.add_argument('--wait', action='store_true',
                       help='Wait and poll until all batches complete')
    parser.add_argument('--poll-interval', type=int, default=60,
                       help='Seconds between status checks when using --wait (default: 60)')
    parser.add_argument('--max-wait', type=int, default=86400,
                       help='Maximum time to wait in seconds (default: 86400 = 24 hours)')

    # Optional flags
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Check only: {args.check_only}")
    logger.info(f"  Wait for completion: {args.wait}")
    if args.wait:
        logger.info(f"  Poll interval: {args.poll_interval}s")
        logger.info(f"  Max wait: {args.max_wait}s")

    try:
        # Create retriever
        retriever = RoleBatchRetriever(
            output_dir=args.output_dir,
            api_key=args.api_key
        )

        if args.check_only:
            # Only check status
            logger.info("\n" + "="*60)
            status_summary = retriever.check_all_batches()

            # Print summary
            logger.info("\n" + "="*60)
            logger.info("SUMMARY:")
            logger.info(f"  In Progress: {len(status_summary.get('in_progress', []))}")
            logger.info(f"  Ended (ready): {len(status_summary.get('ended', []))}")
            logger.info(f"  Retrieved: {len(status_summary.get('completed', []))}")
            logger.info(f"  Errors: {len(status_summary.get('errored', []))}")
            logger.info("="*60)

        elif args.wait:
            # Wait for all batches to complete
            retriever.wait_for_all_batches(
                poll_interval=args.poll_interval,
                max_wait=args.max_wait
            )

        else:
            # Single pass retrieval
            logger.info("\n" + "="*60)
            succeeded, failed = retriever.retrieve_all_completed()

            logger.info("\n" + "="*60)
            logger.info("RETRIEVAL SUMMARY:")
            logger.info(f"  Retrieved: {succeeded}")
            logger.info(f"  Failed: {failed}")
            logger.info("="*60)

        logger.info("\nBatch retrieval completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
