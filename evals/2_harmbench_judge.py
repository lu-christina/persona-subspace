#!/usr/bin/env python3
"""
Script to evaluate responses using a judge model via the OpenAI API for HarmBench data.

This script takes input JSONL files containing prompts and responses,
applies a single judge prompt template to all evaluations, and outputs
scores for each response. Expects yes/no or integer responses from the judge.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import re

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
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"


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
        description="Evaluate responses using a judge model with a single prompt template for HarmBench data"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input JSONL files containing prompts and responses to evaluate"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSONL file path to save results"
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Python file containing PROMPTS dictionary with judge prompt template"
    )
    parser.add_argument(
        "--prompt-key",
        default="judge",
        help="Key in PROMPTS dictionary to use as judge prompt template (default: judge)"
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model to use for evaluation (default: {DEFAULT_JUDGE_MODEL})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="Batch size for API requests (default: 300)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Maximum tokens for judge responses (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making API calls"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Maximum number of concurrent workers (default: 20)"
    )
    parser.add_argument(
        "--requests-per-second",
        type=int,
        default=400,
        help="Rate limit for API requests per second across all workers (default: 400)"
    )
    
    return parser.parse_args()


def read_existing_scores(output_file: str) -> set:
    """Read existing scores to avoid duplicate evaluations."""
    existing = set()
    output_path = Path(output_file)
    
    if not output_path.exists():
        return existing
        
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Create unique key from id, role_id, and magnitude to match baseline script
                    if all(key in data for key in ['id', 'role_id', 'magnitude']):
                        key = f"{data['id']}_{data['role_id']}_{data['magnitude']}"
                        existing.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        logger.warning(f"Could not read existing scores: {e}")
        
    return existing


def load_prompt_template(prompt_file: str, prompt_key: str = "judge") -> str:
    """Load prompt template from Python file containing PROMPTS dict."""
    prompt_path = Path(prompt_file)
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    # Execute the Python file to get the PROMPTS dict
    namespace = {}
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(code, namespace)
    except Exception as e:
        raise ValueError(f"Failed to execute prompt file {prompt_path}: {e}")
    
    if 'PROMPTS' not in namespace:
        raise ValueError(f"No PROMPTS dictionary found in {prompt_path}")
    
    prompts = namespace['PROMPTS']
    if not isinstance(prompts, dict):
        raise ValueError(f"PROMPTS is not a dictionary in {prompt_path}")
    
    if prompt_key not in prompts:
        available_keys = list(prompts.keys())
        raise ValueError(f"Key '{prompt_key}' not found in PROMPTS. Available keys: {available_keys}")
    
    return prompts[prompt_key]


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
                    required_fields = ['id', 'prompt', 'response']
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
    prompt_template: str
) -> List[Tuple[Dict[str, Any], str]]:
    """
    Create evaluation prompts using single template for all input data.
    
    Returns:
        List of tuples: (original_input_data, filled_prompt)
    """
    evaluation_prompts = []
    
    for data_item in input_data:
        # Fill in the template using the prompt and response from input data
        # Use double braces in template file ({{question}}, {{answer}}, {{question_label}}) which become single braces
        try:
            filled_prompt = prompt_template.format(
                question=data_item['prompt'],
                answer=data_item['response'],
                question_label=data_item.get('question_label', '')
            )
            evaluation_prompts.append((data_item, filled_prompt))
        except KeyError as e:
            logger.error(f"Missing placeholder in prompt template: {e}. Template should use {{question}}, {{answer}}, and {{question_label}} placeholders.")
            continue
    
    logger.info(f"Created {len(evaluation_prompts)} evaluation prompts using single template")
    return evaluation_prompts


def parse_judge_score(response_text: str, input_data: Dict[str, Any]) -> Optional[Union[int, str]]:
    """
    Parse the judge's response to extract the score (integer or yes/no).
    
    Args:
        response_text: The judge model's response
        input_data: The original input data for error logging
    
    Returns:
        Integer score, "yes", "no", or None if parsing fails
    """
    if not response_text:
        logger.error(
            f"Empty response from judge model - "
            f"id: {input_data.get('id')}, "
            f"question_label: {input_data.get('question_label')}"
        )
        return None
    
    response_clean = response_text.strip().lower()
    
    # Check for yes/no responses first (case insensitive)
    if response_clean in ['yes', 'no']:
        return response_clean
    
    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())
    
    if numbers:
        # Take the first number found
        try:
            score = int(numbers[0])
            return score
        except ValueError:
            pass
    
    # If neither yes/no nor valid integer found, log error
    logger.error(
        f"Invalid response format (expected integer or 'yes'/'no'): {response_text} - "
        f"id: {input_data.get('id')}, "
        f"question_label: {input_data.get('question_label')}"
    )
    return None


async def call_judge_model_single(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter
) -> Optional[str]:
    """Call the judge model with a single prompt."""
    await rate_limiter.acquire()
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
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
    prompts: List[str],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    batch_size: int = 50
) -> List[Optional[str]]:
    """Call the judge model with a list of prompts using true async concurrency."""
    results = []
    
    # Process in smaller concurrent batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # Create tasks for concurrent execution
        tasks = [
            call_judge_model_single(client, prompt, model, max_tokens, rate_limiter)
            for prompt in batch
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
        logger.info(f"=� Batch {i//batch_size + 1}/{(len(prompts) - 1)//batch_size + 1} ({len(batch)} requests)")
    
    return results


async def process_batch(
    batch_data: List[Tuple[Dict[str, Any], str]],
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
    # Extract just the prompt texts for the API call
    prompt_texts = [prompt_text for _, prompt_text in batch_data]
    
    # Call the judge model
    logger.info(f"Sending batch of {len(prompt_texts)} prompts to judge model")
    responses = await call_judge_model(
        client=client,
        prompts=prompt_texts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )
    
    logger.info(f"Received {len(responses)} responses from judge model")
    
    # Parse scores and write results immediately
    successful_count = 0
    
    for i, (input_data, prompt_text) in enumerate(batch_data):
        if i < len(responses) and responses[i] is not None:
            response_text = responses[i]
            score = parse_judge_score(response_text, input_data)
            
            if score is not None:
                # Create result object with score
                result_data = input_data.copy()
                result_data['score'] = score
                
                # Write result immediately
                await write_result_to_file(output_file, result_data, write_lock)
                successful_count += 1
            # If score parsing failed, error was already logged in parse_judge_score
        else:
            logger.error(
                f"No response received from judge model - "
                f"id: {input_data.get('id')}, "
                f"question_label: {input_data.get('question_label')}"
            )
    
    logger.info(f"Successfully processed {successful_count}/{len(batch_data)} items in batch")
    return successful_count


async def batch_worker(
    batch_data: List[Tuple[Dict[str, Any], str]],
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
            logger.info(f"=� Starting batch {batch_id} with {len(batch_data)} items")
            
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
            
            logger.info(f" Completed batch {batch_id}: {successful_count}/{len(batch_data)} items processed")
            return (batch_id, successful_count, len(batch_data))
            
        except Exception as e:
            error_msg = f"Unexpected error in batch {batch_id}: {e}"
            logger.error(error_msg)
            return (batch_id, 0, len(batch_data))


async def main():
    """Main async function."""
    args = parse_arguments()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Load prompt template
    try:
        logger.info(f"Loading prompt template from {args.prompt_file}, key '{args.prompt_key}'")
        prompt_template = load_prompt_template(args.prompt_file, args.prompt_key)
        logger.info(f"Successfully loaded prompt template ({len(prompt_template)} chars)")
    except Exception as e:
        logger.error(f"Failed to load prompt template: {e}")
        sys.exit(1)
    
    # Load input data
    logger.info(f"Loading input data from {len(args.input_files)} files")
    input_data = load_input_data(args.input_files)
    
    if not input_data:
        logger.error("No input data loaded")
        sys.exit(1)
    
    # Check for existing results to enable restart functionality
    existing_scores = read_existing_scores(args.output)
    if existing_scores:
        logger.info(f"Found {len(existing_scores)} existing scores, will skip duplicates")
    
    # Create evaluation prompts
    logger.info("Creating evaluation prompts using single template")
    evaluation_prompts = create_evaluation_prompts(input_data, prompt_template)
    
    if not evaluation_prompts:
        logger.error("No evaluation prompts created")
        sys.exit(1)
    
    # Filter out already-processed items
    original_count = len(evaluation_prompts)
    if existing_scores:
        filtered_prompts = []
        for input_data_item, prompt_text in evaluation_prompts:
            # Create unique key from input data to match existing scores
            if all(key in input_data_item for key in ['id', 'role_id', 'magnitude']):
                key = f"{input_data_item['id']}_{input_data_item['role_id']}_{input_data_item['magnitude']}"
                if key not in existing_scores:
                    filtered_prompts.append((input_data_item, prompt_text))
        
        evaluation_prompts = filtered_prompts
        skipped_count = original_count - len(evaluation_prompts)
        
        if skipped_count > 0:
            logger.info(f"Filtered out {skipped_count} already-processed items")
    
    if not evaluation_prompts:
        logger.info("No new items to process - all evaluations already complete!")
        return
    
    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        logger.info(f"Total prompts that would be sent to API: {len(evaluation_prompts)}")
        
        # Show one example API call request
        if evaluation_prompts:
            _, example_prompt = evaluation_prompts[0]
            logger.info("\nExample API call request:")
            logger.info(f"  Model: {args.judge_model}")
            logger.info(f"  Max tokens: {args.max_tokens}")
            logger.info(f"  Temperature: 0.0")
            logger.info(f"  Message content preview: {example_prompt[:1000]}...")
        return
    
    # Setup output file
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results will be written to: {output_file}")
    
    # Initialize OpenAI client
    client = openai.AsyncOpenAI()
    logger.info(f"Initialized OpenAI client with judge model: {args.judge_model}")
    
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
            
            logger.info(f" [{completed_count}/{total_count}] Batch {batch_id} completed: {successful_count}/{batch_total} items")
                
        except Exception as e:
            completed_count += 1
            logger.error(f"L [{completed_count}/{total_count}] Batch exception: {e}")
    
    elapsed_time = time.time() - start_time
    
    # Calculate final statistics
    skipped_count = original_count - total_items if existing_scores else 0
    logger.info(f"Processing complete: {total_successful}/{total_items} new items processed successfully in {elapsed_time:.1f}s")
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} items that were already completed")
        logger.info(f"Total items in dataset: {original_count}")
    
    if total_successful < total_items:
        failed_count = total_items - total_successful
        logger.warning(f"{failed_count} items failed to process - check logs for details")


if __name__ == "__main__":
    asyncio.run(main())