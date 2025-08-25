#!/usr/bin/env python3
"""
Script to evaluate role responses using a judge model via the OpenAI API.

This script takes evaluation prompts for roles from the instructions directory,
pairs them with responses from the responses directory, and uses a judge model
to score how well each response exhibits the target role.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
DEFAULT_INSTRUCTIONS_DIR = Path(__file__).parent / "data_generation" / "instructions"
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
        description="Evaluate role responses using a judge model"
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
        "--instructions-dir",
        default=str(DEFAULT_INSTRUCTIONS_DIR),
        help=f"Directory containing judge prompt instruction files (default: {DEFAULT_INSTRUCTIONS_DIR})"
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


def load_judge_prompt(question_label: str, instructions_dir: Path) -> str:
    """Load judge prompt for a given question label."""
    instruction_file = instructions_dir / f"{question_label}.json"
    
    if not instruction_file.exists():
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
    
    with open(instruction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'eval_prompt' not in data:
        raise ValueError(f"Missing 'eval_prompt' key in instruction file: {instruction_file}")
    
    return data['eval_prompt']


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
                    required_fields = ['id', 'role_id', 'role_label', 'question_label', 'prompt', 'response']
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
    instructions_dir: Path
) -> List[Tuple[Dict[str, Any], str]]:
    """
    Create evaluation prompts for the judge model from input data.
    
    Returns:
        List of tuples: (original_input_data, filled_prompt)
    """
    evaluation_prompts = []
    judge_prompt_cache = {}  # Cache judge prompts to avoid repeated file reads
    
    for data_item in input_data:
        question_label = data_item['question_label']
        
        # Load judge prompt (with caching)
        if question_label not in judge_prompt_cache:
            try:
                judge_prompt_cache[question_label] = load_judge_prompt(question_label, instructions_dir)
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load judge prompt for question_label '{question_label}': {e}")
                continue
        
        eval_prompt_template = judge_prompt_cache[question_label]
        
        # Fill in the template using the prompt and response from input data
        filled_prompt = eval_prompt_template.format(
            question=data_item['prompt'],
            answer=data_item['response']
        )
        
        evaluation_prompts.append((data_item, filled_prompt))
    
    logger.info(f"Created {len(evaluation_prompts)} evaluation prompts using {len(judge_prompt_cache)} unique judge prompt templates")
    return evaluation_prompts




def parse_judge_score(response_text: str, input_data: Dict[str, Any]) -> Optional[int]:
    """
    Parse the judge's response to extract the numerical score.
    
    Args:
        response_text: The judge model's response
        input_data: The original input data for error logging
    
    Returns:
        Integer score between 0-3, or None if parsing fails
    """
    if not response_text:
        logger.error(
            f"Empty response from judge model - "
            f"magnitude: {input_data.get('magnitude')}, "
            f"question_id: {input_data.get('id')}, "
            f"question_label: {input_data.get('question_label')}, "
            f"role_id: {input_data.get('role_id')}"
        )
        return None
    
    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())
    
    if not numbers:
        logger.error(
            f"No numbers found in judge response: {response_text} - "
            f"magnitude: {input_data.get('magnitude')}, "
            f"question_id: {input_data.get('id')}, "
            f"question_label: {input_data.get('question_label')}, "
            f"role_id: {input_data.get('role_id')}"
        )
        return None
    
    # Take the first number found
    try:
        score = int(numbers[0])
        if 0 <= score <= 3:
            return score
        else:
            logger.error(
                f"Score out of range (0-3): {score} from response: {response_text} - "
                f"magnitude: {input_data.get('magnitude')}, "
                f"question_id: {input_data.get('id')}, "
                f"question_label: {input_data.get('question_label')}, "
                f"role_id: {input_data.get('role_id')}"
            )
            return None
    except ValueError:
        logger.error(
            f"Could not parse score from: {numbers[0]} in response: {response_text} - "
            f"magnitude: {input_data.get('magnitude')}, "
            f"question_id: {input_data.get('id')}, "
            f"question_label: {input_data.get('question_label')}, "
            f"role_id: {input_data.get('role_id')}"
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
        logger.info(f"📦 Batch {i//batch_size + 1}/{(len(prompts) - 1)//batch_size + 1} ({len(batch)} requests)")
    
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
                f"magnitude: {input_data.get('magnitude')}, "
                f"question_id: {input_data.get('id')}, "
                f"question_label: {input_data.get('question_label')}, "
                f"role_id: {input_data.get('role_id')}"
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
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Load input data
    logger.info(f"Loading input data from {len(args.input_files)} files")
    input_data = load_input_data(args.input_files)
    
    if not input_data:
        logger.error("No input data loaded")
        sys.exit(1)
    
    # Create evaluation prompts
    instructions_dir = Path(args.instructions_dir)
    if not instructions_dir.exists():
        logger.error(f"Instructions directory not found: {instructions_dir}")
        sys.exit(1)
    
    logger.info(f"Creating evaluation prompts using instructions from {instructions_dir}")
    evaluation_prompts = create_evaluation_prompts(input_data, instructions_dir)
    
    if not evaluation_prompts:
        logger.error("No evaluation prompts created")
        sys.exit(1)
    
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
            logger.info(f"  Message content preview: {example_prompt[:200]}...")
        return
    
    # Setup output file
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file if it exists
    if output_file.exists():
        logger.info(f"Clearing existing output file: {output_file}")
        output_file.unlink()
    
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