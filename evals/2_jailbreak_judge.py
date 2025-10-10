#!/usr/bin/env python3
"""
Script to evaluate jailbreak responses using a judge model via the OpenRouter API.

This script takes jailbreak prompts and responses, and uses a judge model
to evaluate how successful the jailbreak attempt was at eliciting harmful content.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from httpx/openai (only show errors)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# Constants
DEFAULT_JUDGE_PROMPT_FILE = "/root/git/persona-subspace/evals/jailbreak/judge.json"
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-chat"


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
        description="Evaluate jailbreak responses using a judge model via OpenRouter API"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input JSONL files containing jailbreak prompts and responses to evaluate"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSONL file path to save results"
    )
    parser.add_argument(
        "--judge-prompt",
        default=str(DEFAULT_JUDGE_PROMPT_FILE),
        help=f"JSON file containing judge prompt template (default: {DEFAULT_JUDGE_PROMPT_FILE})"
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model to use for evaluation (default: {DEFAULT_JUDGE_MODEL})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=400,
        help="Batch size for API requests (default: 300)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens for judge responses (default: 500)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making API calls"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers (default: 10)"
    )
    parser.add_argument(
        "--requests-per-second",
        type=int,
        default=50,
        help="Rate limit for API requests per second across all workers (default: 50)"
    )
    
    return parser.parse_args()


def load_judge_prompt_template(prompt_file: Path) -> Dict[str, Any]:
    """Load judge prompt template from JSON file."""
    if not prompt_file.exists():
        raise FileNotFoundError(f"Judge prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'prompts' not in data:
        raise ValueError(f"Missing 'prompts' key in judge prompt file: {prompt_file}")
    
    return data


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
                    # Validate required fields for jailbreak data
                    required_fields = ['id', 'prompt', 'harm_category', 'persona', 'response']
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        logger.warning(f"Missing required fields {missing_fields} on line {line_num} in {input_file}")
                        continue
                    
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {input_file}: {e}")
                    continue
    
    logger.info(f"Loaded {len(all_data)} input records from {len(input_files)} files")
    return all_data


def load_existing_results(output_file: Path) -> Set[Tuple[int, float, int]]:
    """
    Load existing results from output file to support restart.
    Returns set of (id, magnitude_or_cap, sample_id) tuples that were already processed.
    For backwards compatibility, entries without sample_id are treated as sample_id=0.
    Tries 'magnitude' first, then falls back to 'cap' if magnitude is not available.
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
                if 'id' in data:
                    request_key = data['id']
                    # Try magnitude first, then fall back to cap
                    magnitude_or_cap = data.get('magnitude', data.get('cap'))
                    if magnitude_or_cap is not None:
                        # For backwards compatibility, treat missing sample_id as 0
                        sample_id = data.get('sample_id', 0)
                        existing.add((request_key, float(magnitude_or_cap), sample_id))
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


def create_evaluation_prompts(
    input_data: List[Dict[str, Any]],
    judge_template: Dict[str, Any],
    existing_results: Set[Tuple[int, float, int]]
) -> List[Tuple[Dict[str, Any], List[Dict[str, str]]]]:
    """
    Create evaluation prompts for the judge model from input data.

    Returns:
        List of tuples: (original_input_data, chat_messages)
    """
    evaluation_prompts = []
    skipped_count = 0

    for data_item in input_data:
        # Check if this item was already processed
        request_key = data_item['id']
        # Try magnitude first, then fall back to cap
        magnitude_or_cap = data_item.get('magnitude', data_item.get('cap', 0.0))
        # For backwards compatibility, treat missing sample_id as 0
        sample_id = data_item.get('sample_id', 0)
        if (request_key, float(magnitude_or_cap), sample_id) in existing_results:
            skipped_count += 1
            continue
        
        # Create the evaluation messages from the template
        messages = []
        for template_message in judge_template['prompts']:
            content = template_message['content'].format(
                request=data_item['prompt'],
                response=data_item['response'],
                behavior=data_item['persona'],
                action=data_item['harm_category']
            )
            
            message = {
                'role': template_message['role'],
                'content': content
            }
            messages.append(message)
        
        evaluation_prompts.append((data_item, messages))
    
    logger.info(f"Created {len(evaluation_prompts)} evaluation prompts, skipped {skipped_count} existing results")
    return evaluation_prompts


def parse_judge_response(response_text: str, input_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the judge's response to extract the score and analysis.

    Args:
        response_text: The judge model's response
        input_data: The original input data for error logging

    Returns:
        Tuple of (score, analysis) or (None, None) if parsing fails
    """
    # Get magnitude or cap for logging
    magnitude_or_cap = input_data.get('magnitude', input_data.get('cap', 'N/A'))

    if not response_text:
        logger.error(
            f"Empty response from judge model - "
            f"id: {input_data.get('id')}, "
            f"magnitude/cap: {magnitude_or_cap}"
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

        if 'score' not in response_json or 'analysis' not in response_json:
            logger.error(
                f"Missing score or analysis in judge response: {response_text} - "
                f"id: {input_data.get('id')}, "
                f"magnitude/cap: {magnitude_or_cap}"
            )
            return None, None

        return response_json['score'], response_json['analysis']

    except json.JSONDecodeError as e:
        # Try to fix common JSON issues and retry parsing
        try:
            fixed_json = fix_common_json_issues(cleaned_text)
            response_json = json.loads(fixed_json)

            if 'score' not in response_json or 'analysis' not in response_json:
                logger.error(
                    f"Missing score or analysis in judge response: {response_text} - "
                    f"id: {input_data.get('id')}, "
                    f"magnitude/cap: {magnitude_or_cap}"
                )
                return None, None

            return response_json['score'], response_json['analysis']

        except json.JSONDecodeError:
            logger.error(
                f"Could not parse JSON response from judge: {response_text} - "
                f"id: {input_data.get('id')}, "
                f"magnitude/cap: {magnitude_or_cap}, "
                f"error: {e}"
            )
            return None, None


def fix_common_json_issues(json_text: str) -> str:
    """
    Fix common JSON formatting issues that prevent parsing.
    
    Args:
        json_text: The potentially malformed JSON text
        
    Returns:
        Fixed JSON text
    """
    import re
    
    # Remove trailing commas before closing braces/brackets
    # This handles cases like: {"key": "value",} and ["item1", "item2",]
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
    
    return json_text


async def call_judge_model_single(
    client: openai.AsyncOpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter
) -> Optional[str]:
    """Call the judge model with a single set of messages."""
    await rate_limiter.acquire()
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0  # For consistent evaluation
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error calling judge model: {e}")
        return None


async def call_judge_model(
    client: openai.AsyncOpenAI,
    message_batches: List[List[Dict[str, str]]],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    batch_size: int = 50
) -> List[Optional[str]]:
    """Call the judge model with a list of message sets using true async concurrency."""
    results = []
    
    # Process in smaller concurrent batches
    for i in range(0, len(message_batches), batch_size):
        batch = message_batches[i:i + batch_size]
        
        # Create tasks for concurrent execution
        tasks = [
            call_judge_model_single(client, messages, model, max_tokens, rate_limiter)
            for messages in batch
        ]
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Exception in batch: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        results.extend(processed_results)
        logger.info(f"📦 Batch {i//batch_size + 1}/{(len(message_batches) - 1)//batch_size + 1} ({len(batch)} requests)")
    
    return results


async def process_batch(
    batch_data: List[Tuple[Dict[str, Any], List[Dict[str, str]]]],
    client: openai.AsyncOpenAI,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    rate_limiter: RateLimiter,
    output_file: Path,
    write_lock: asyncio.Lock
) -> int:
    """
    Process a batch of evaluation prompts and write results immediately.
    
    Returns:
        Number of successfully processed items
    """
    # Extract just the message sets for the API call
    message_batches = [messages for _, messages in batch_data]
    
    # Call the judge model
    logger.info(f"Sending batch of {len(message_batches)} prompts to judge model")
    responses = await call_judge_model(
        client=client,
        message_batches=message_batches,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )
    
    logger.info(f"Received {len(responses)} responses from judge model")
    
    # Parse scores and write results immediately
    successful_count = 0
    
    for i, (input_data, messages) in enumerate(batch_data):
        if i < len(responses) and responses[i] is not None:
            response_text = responses[i]
            score, analysis = parse_judge_response(response_text, input_data)
            
            if score is not None and analysis is not None:
                # Create result object with score and analysis
                result_data = input_data.copy()
                result_data['score'] = score
                result_data['analysis'] = analysis
                
                # Write result immediately
                await write_result_to_file(output_file, result_data, write_lock)
                successful_count += 1
            # If parsing failed, error was already logged in parse_judge_response
        else:
            magnitude_or_cap = input_data.get('magnitude', input_data.get('cap', 'N/A'))
            logger.error(
                f"No response received from judge model - "
                f"id: {input_data.get('id')}, "
                f"magnitude/cap: {magnitude_or_cap}"
            )
    
    logger.info(f"Successfully processed {successful_count}/{len(batch_data)} items in batch")
    return successful_count


async def batch_worker(
    batch_data: List[Tuple[Dict[str, Any], List[Dict[str, str]]]],
    client: openai.AsyncOpenAI,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
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
            
            successful_count = await process_batch(
                batch_data=batch_data,
                client=client,
                judge_model=judge_model,
                max_tokens=max_tokens,
                batch_size=batch_size,
                rate_limiter=rate_limiter,
                output_file=output_file,
                write_lock=write_lock
            )
            
            logger.info(f"✅ Completed batch {batch_id}: {successful_count}/{len(batch_data)} items processed")
            return (batch_id, successful_count, len(batch_data))
            
        except Exception as e:
            error_msg = f"Unexpected error in batch {batch_id}: {e}"
            logger.error(error_msg)
            return (batch_id, 0, len(batch_data))


async def main():
    """Main async function."""
    args = parse_arguments()
    
    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Load judge prompt template
    judge_prompt_file = Path(args.judge_prompt)
    if not judge_prompt_file.exists():
        logger.error(f"Judge prompt file not found: {judge_prompt_file}")
        sys.exit(1)
    
    logger.info(f"Loading judge prompt template from {judge_prompt_file}")
    judge_template = load_judge_prompt_template(judge_prompt_file)
    
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
    
    # Create evaluation prompts
    logger.info(f"Creating evaluation prompts")
    evaluation_prompts = create_evaluation_prompts(input_data, judge_template, existing_results)
    
    if not evaluation_prompts:
        logger.error("No evaluation prompts created (all items may already be processed)")
        sys.exit(1)
    
    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        logger.info(f"Total prompts that would be sent to API: {len(evaluation_prompts)}")
        
        # Show one example API call request
        if evaluation_prompts:
            _, example_messages = evaluation_prompts[0]
            logger.info("\nExample API call request:")
            logger.info(f"  Model: {args.judge_model}")
            logger.info(f"  Max tokens: {args.max_tokens}")
            logger.info(f"  Temperature: 0.0")
            logger.info(f"  Messages: {len(example_messages)} messages")
            for i, msg in enumerate(example_messages):
                logger.info(f"    Message {i+1} ({msg['role']}):")
                logger.info(f"{msg['content']}")
                logger.info("")
        return
    
    logger.info(f"Results will be appended to: {output_file}")
    
    # Initialize OpenRouter client
    client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    logger.info(f"Initialized OpenRouter client with judge model: {args.judge_model}")
    
    # Initialize concurrency controls
    rate_limiter = RateLimiter(args.requests_per_second)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.max_workers)
    
    logger.info(f"Starting concurrent processing with {args.max_workers} workers, {args.requests_per_second} req/s limit")
    
    # Split data into batches for processing
    batch_size = args.batch_size
    batches = []
    for i in range(0, len(evaluation_prompts), batch_size):
        batch = evaluation_prompts[i:i + batch_size]
        batches.append(batch)
    
    logger.info(f"Split {len(evaluation_prompts)} items into {len(batches)} batches of up to {batch_size} items each")
    
    # Create worker tasks for all batches
    tasks = []
    for batch_id, batch_data in enumerate(batches):
        task = batch_worker(
            batch_data=batch_data,
            client=client,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            batch_size=50,  # API batch size within each worker batch
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