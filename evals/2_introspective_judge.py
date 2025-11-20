#!/usr/bin/env python3
"""
Script to evaluate introspective responses using a judge model via the Anthropic API.

This script takes introspective prompts and responses, and uses Claude
to evaluate and label them based on their type (purpose, traits, capabilities).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union

import anthropic
from dotenv import load_dotenv

# Import prompts from the introspective directory
sys.path.insert(0, str(Path(__file__).parent / "introspective"))
from prompts import PROMPTS

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, rate: float):
        self.rate = rate  # requests per second
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.time()
            # Add tokens based on elapsed time
            self.tokens = min(self.rate, self.tokens + (now - self.last_update) * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            # Wait until we can get a token
            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate introspective responses using Claude via Anthropic API"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input JSONL files containing introspective prompts and responses to evaluate"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSONL file path to save results"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Claude model to use for evaluation (default: claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens for judge responses (default: 500)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=100,
        help="Maximum number of concurrent workers (default: 10)"
    )
    parser.add_argument(
        "--requests-per-second",
        type=int,
        default=500,
        help="Rate limit for API requests per second across all workers (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of items to process in each batch (default: 100)"
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key (optional, will use environment if not provided)"
    )

    return parser.parse_args()


def load_input_data(input_files: List[str]) -> List[Dict[str, Any]]:
    """Load input data from JSONL files."""
    all_data = []

    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            continue

        logger.info(f"Loading input file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    # Validate required fields
                    required_fields = ['id', 'question', 'response', 'type']
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        logger.warning(f"Missing required fields {missing_fields} on line {line_num} in {input_file}")
                        continue

                    # Validate that type is valid
                    if data['type'] not in PROMPTS:
                        logger.warning(f"Invalid type '{data['type']}' on line {line_num} in {input_file}. Must be one of: {list(PROMPTS.keys())}")
                        continue

                    all_data.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {input_file}: {e}")
                    continue

    logger.info(f"Loaded {len(all_data)} input records from {len(input_files)} files")
    return all_data


def load_existing_results(output_file: Path) -> Set[Tuple[int, str, int]]:
    """
    Load existing results from output file to support restart.
    Returns set of (id, experiment_id, sample_id) tuples that were already processed.
    """
    if not output_file.exists():
        return set()

    existing = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if 'id' in data and 'experiment_id' in data:
                    request_key = data['id']
                    identifier = data['experiment_id']
                    sample_id = data.get('sample_id', 0)
                    existing.add((request_key, identifier, sample_id))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in existing output on line {line_num}: {e}")
                continue

    logger.info(f"Found {len(existing)} existing results in output file")
    return existing


async def write_result_to_file(
    output_file: Path,
    result_data: Dict[str, Any],
    write_lock: asyncio.Lock
) -> None:
    """Thread-safe function to write a single result to the output JSONL file."""
    async with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')


def create_evaluation_messages(
    data_item: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Create evaluation messages for the judge model from input data.

    Returns:
        List of message dicts for the API call
    """
    # Get the appropriate prompt template based on type
    prompt_type = data_item['type']
    prompt_template = PROMPTS[prompt_type]

    # Format the prompt with the prefill (question) and completion (response)
    formatted_content = prompt_template.format(
        prefill=data_item['question'],
        completion=data_item['response']
    )

    # Create messages list
    messages = [
        {
            'role': 'user',
            'content': formatted_content
        }
    ]

    return messages


def parse_judge_response(
    response_text: str,
    input_data: Dict[str, Any]
) -> Tuple[Optional[Union[str, List[str]]], Optional[str]]:
    """
    Parse the judge's response to extract the label and explanation.

    Args:
        response_text: The judge model's response
        input_data: The original input data for error logging

    Returns:
        Tuple of (label, explanation) or (None, None) if parsing fails
        label can be a string or list of strings depending on the prompt type
    """
    identifier = input_data.get('experiment_id', 'N/A')

    if not response_text:
        logger.error(
            f"Empty response from judge model - "
            f"id: {input_data.get('id')}, "
            f"identifier: {identifier}"
        )
        return None, None

    try:
        # Clean the response text by removing markdown code blocks
        cleaned_text = response_text.strip()

        # Remove ```json and ``` markers if present
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]  # Remove ```json
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]   # Remove ```

        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]  # Remove trailing ```

        cleaned_text = cleaned_text.strip()

        # Try to parse as JSON
        response_json = json.loads(cleaned_text)

        if 'label' not in response_json or 'explanation' not in response_json:
            logger.error(
                f"Missing label or explanation in judge response: {response_text} - "
                f"id: {input_data.get('id')}, "
                f"identifier: {identifier}"
            )
            return None, None

        return response_json['label'], response_json['explanation']

    except json.JSONDecodeError as e:
        logger.error(
            f"Could not parse JSON response from judge: {response_text} - "
            f"id: {input_data.get('id')}, "
            f"identifier: {identifier}, "
            f"error: {e}"
        )
        return None, None


async def call_judge_model_single(
    client: anthropic.Anthropic,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter
) -> Optional[str]:
    """Call the judge model with a single set of messages."""
    await rate_limiter.acquire()

    try:
        # Call the API in a thread pool since anthropic client is sync
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0  # For consistent evaluation
            )
        )

        if response.content and len(response.content) > 0:
            return response.content[0].text
        else:
            return None

    except Exception as e:
        logger.error(f"Error calling judge model: {e}")
        return None


async def process_item(
    client: anthropic.Anthropic,
    data_item: Dict[str, Any],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    output_file: Path,
    write_lock: asyncio.Lock
) -> bool:
    """
    Process a single evaluation item.

    Returns:
        True if successful, False otherwise
    """
    # Create evaluation messages
    messages = create_evaluation_messages(data_item)

    # Call the judge model
    response_text = await call_judge_model_single(
        client=client,
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter
    )

    if response_text is None:
        logger.error(
            f"No response received - "
            f"id: {data_item.get('id')}, "
            f"experiment_id: {data_item.get('experiment_id')}"
        )
        return False

    # Parse the response
    label, explanation = parse_judge_response(response_text, data_item)

    if label is None or explanation is None:
        return False

    # Create result object
    result_data = data_item.copy()
    result_data['label'] = label
    result_data['explanation'] = explanation

    # Write result immediately
    await write_result_to_file(output_file, result_data, write_lock)
    return True


async def batch_worker(
    client: anthropic.Anthropic,
    batch_data: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    output_file: Path,
    write_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    batch_id: int
) -> Tuple[int, int, int]:
    """
    Worker function to process a batch of evaluation data.

    Returns:
        Tuple of (batch_id, successful_count, total_count)
    """
    async with semaphore:
        try:
            logger.info(f"🚀 Starting batch {batch_id} with {len(batch_data)} items")

            # Process all items in the batch concurrently
            tasks = [
                process_item(
                    client=client,
                    data_item=item,
                    model=model,
                    max_tokens=max_tokens,
                    rate_limiter=rate_limiter,
                    output_file=output_file,
                    write_lock=write_lock
                )
                for item in batch_data
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes
            successful_count = sum(1 for r in results if r is True)

            logger.info(f"✅ Completed batch {batch_id}: {successful_count}/{len(batch_data)} items processed")
            return (batch_id, successful_count, len(batch_data))

        except Exception as e:
            error_msg = f"Unexpected error in batch {batch_id}: {e}"
            logger.error(error_msg)
            return (batch_id, 0, len(batch_data))


async def main():
    """Main async function."""
    args = parse_arguments()

    # Check for Anthropic API key
    api_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables or arguments")
        sys.exit(1)

    # Load input data
    logger.info(f"Loading input data from {len(args.input_files)} files")
    input_data = load_input_data(args.input_files)

    if not input_data:
        logger.error("No input data loaded")
        sys.exit(1)

    # Setup output file and load existing results for restart support
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checking for existing results in: {output_file}")
    existing_results = load_existing_results(output_file)

    # Filter out already-processed items
    items_to_process = []
    skipped_count = 0
    for item in input_data:
        key = (
            item['id'],
            item.get('experiment_id', ''),
            item.get('sample_id', 0)
        )
        if key in existing_results:
            skipped_count += 1
        else:
            items_to_process.append(item)

    logger.info(f"Found {len(items_to_process)} items to process, skipped {skipped_count} existing results")

    if not items_to_process:
        logger.info("No items to process (all already completed)")
        return

    logger.info(f"Results will be appended to: {output_file}")

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    logger.info(f"Initialized Anthropic client with model: {args.model}")

    # Initialize concurrency controls
    rate_limiter = RateLimiter(args.requests_per_second)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.max_workers)

    logger.info(f"Starting concurrent processing with {args.max_workers} workers, {args.requests_per_second} req/s limit")

    # Split data into batches for processing
    batch_size = args.batch_size
    batches = []
    for i in range(0, len(items_to_process), batch_size):
        batch = items_to_process[i:i + batch_size]
        batches.append(batch)

    logger.info(f"Split {len(items_to_process)} items into {len(batches)} batches of up to {batch_size} items each")

    # Create worker tasks for all batches
    tasks = []
    for batch_id, batch_data in enumerate(batches):
        task = batch_worker(
            client=client,
            batch_data=batch_data,
            model=args.model,
            max_tokens=args.max_tokens,
            rate_limiter=rate_limiter,
            output_file=output_file,
            write_lock=write_lock,
            semaphore=semaphore,
            batch_id=batch_id
        )
        tasks.append(task)

    # Process all batches concurrently with progress tracking
    start_time = time.time()
    logger.info(f"Launching {len(tasks)} concurrent batch workers...")

    # Track progress using asyncio.as_completed
    completed_count = 0
    total_count = len(tasks)
    total_successful = 0
    total_items = 0

    # Use as_completed to show progress as tasks finish
    for coro in asyncio.as_completed(tasks):
        try:
            batch_id, successful_count, batch_total = await coro
            completed_count += 1
            total_successful += successful_count
            total_items += batch_total

            logger.info(f"✅ [{completed_count}/{total_count}] Batch {batch_id} completed: {successful_count}/{batch_total} items")

        except Exception as e:
            completed_count += 1
            logger.error(f"❌ [{completed_count}/{total_count}] Batch exception: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"Processing complete: {total_successful}/{total_items} items processed successfully in {elapsed_time:.1f}s")

    if total_successful < total_items:
        failed_count = total_items - total_successful
        logger.warning(f"{failed_count} items failed to process - check logs for details")


if __name__ == "__main__":
    asyncio.run(main())
