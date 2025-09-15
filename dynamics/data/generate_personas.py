#!/usr/bin/env python3
"""
Script to generate conversation topics for personas using OpenRouter API.

This script takes persona JSONL files and uses the generate_persona prompt template
to send prompts to an LLM and save outputs to a new JSONL file where every line
is the output JSON object from the LLM response.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

import openai
from dotenv import load_dotenv

# Add the parent directory to sys.path to import prompts
sys.path.append(str(Path(__file__).parent / "data"))
from prompts import PROMPTS

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
DEFAULT_MODEL = "moonshotai/kimi-k2-0905"
DEFAULT_MAX_TOKENS = 4000


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
        description="Generate conversation topics for personas using OpenAI API"
    )
    parser.add_argument(
        "input_file",
        help="Input JSONL file containing personas (e.g., coding.jsonl)"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSONL file path to save results"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to use for generation (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens per response (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for API requests (default: 50)"
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
        type=float,
        default=100,
        help="Rate limit for API requests per second across all workers (default: 10.0)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only first 3 personas for testing"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Process personas regardless of if they already exist in output (skip existing work check)"
    )
    
    return parser.parse_args()


def load_personas(input_file: Path) -> List[Dict[str, Any]]:
    """Load personas from JSONL file."""
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return []
    
    personas = []
    logger.info(f"Loading personas from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                # Validate required fields
                if 'domain' not in data or 'persona' not in data:
                    logger.warning(f"Missing required fields on line {line_index + 1}")
                    continue

                # Add ID field based on line index (0-based)
                data['id'] = line_index
                personas.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_index + 1}: {e}")
                continue
    
    logger.info(f"Loaded {len(personas)} personas")
    return personas


def load_existing_results(output_file: Path) -> Set[int]:
    """
    Load existing results from output file to support restart.
    Returns set of IDs that were already processed.
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
                    existing.add(data['id'])
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in existing output on line {line_num}: {e}")
                continue

    logger.info(f"Found {len(existing)} existing results in output file")
    return existing


def format_persona_prompt(domain: str, persona: str) -> str:
    """Format a prompt for persona topic generation using the generate_persona template."""
    prompt_template = PROMPTS["generate_persona"]
    
    # Format the prompt with domain and persona information
    formatted_prompt = prompt_template.format(
        domain=domain,
        persona=persona
    )
    
    return formatted_prompt


async def write_result_to_file(
    output_file: Path,
    result_data: Dict[str, Any],
    write_lock: asyncio.Lock
) -> None:
    """Thread-safe function to write a single result to the output JSONL file."""
    async with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')


def parse_llm_response(response_text: str, persona_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse the LLM's response to extract the JSON object.
    
    Args:
        response_text: The LLM model's response
        persona_data: The original persona data for error logging
    
    Returns:
        Parsed JSON object or None if parsing fails
    """
    if not response_text:
        logger.error(f"Empty response from LLM for persona: {persona_data.get('persona', '')[:100]}...")
        return None
    
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
        
        # Validate expected structure
        if 'domain' not in response_json or 'persona' not in response_json or 'topics' not in response_json:
            logger.error(f"Missing expected fields in LLM response for persona: {persona_data.get('persona', '')[:100]}...")
            return None
        
        return response_json
        
    except json.JSONDecodeError as e:
        logger.error(f"Could not parse JSON response from LLM for persona: {persona_data.get('persona', '')[:100]}..., error: {e}")
        logger.debug(f"Raw response: {response_text[:500]}...")
        return None


async def call_openrouter_single(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter
) -> Optional[str]:
    """Call OpenRouter API with a single prompt."""
    await rate_limiter.acquire()
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7  # Some creativity for topic generation
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        return None


async def process_persona_batch(
    batch_personas: List[Dict[str, Any]],
    client: openai.AsyncOpenAI,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    output_file: Path,
    write_lock: asyncio.Lock
) -> int:
    """
    Process a batch of personas and write results immediately.
    
    Returns:
        Number of successfully processed personas
    """
    logger.info(f"Processing batch of {len(batch_personas)} personas")
    
    successful_count = 0
    
    # Process each persona in the batch
    tasks = []
    for persona_data in batch_personas:
        domain = persona_data['domain']
        persona = persona_data['persona']
        formatted_prompt = format_persona_prompt(domain, persona)
        
        task = call_openrouter_single(client, formatted_prompt, model, max_tokens, rate_limiter)
        tasks.append((persona_data, task))
    
    # Wait for all API calls to complete
    for persona_data, task in tasks:
        response_text = await task
        
        if response_text is not None:
            parsed_response = parse_llm_response(response_text, persona_data)
            
            if parsed_response is not None:
                # Create new result with ID first
                result = {'id': persona_data['id']}
                result.update(parsed_response)
                # Write result immediately
                await write_result_to_file(output_file, result, write_lock)
                successful_count += 1
            # If parsing failed, error was already logged in parse_llm_response
        else:
            logger.error(f"No response received from OpenRouter for persona: {persona_data.get('persona', '')[:100]}...")
    
    logger.info(f"Successfully processed {successful_count}/{len(batch_personas)} personas in batch")
    return successful_count


async def batch_worker(
    batch_personas: List[Dict[str, Any]],
    client: openai.AsyncOpenAI,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    output_file: Path,
    write_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    batch_id: int
) -> tuple[int, int, int]:
    """
    Worker function to process a batch of personas.
    
    Returns:
        Tuple of (batch_id, successful_count, total_count)
    """
    async with semaphore:
        try:
            logger.info(f"🚀 Starting batch {batch_id} with {len(batch_personas)} personas")
            
            successful_count = await process_persona_batch(
                batch_personas=batch_personas,
                client=client,
                model=model,
                max_tokens=max_tokens,
                rate_limiter=rate_limiter,
                output_file=output_file,
                write_lock=write_lock
            )
            
            logger.info(f"✅ Completed batch {batch_id}: {successful_count}/{len(batch_personas)} personas processed")
            return (batch_id, successful_count, len(batch_personas))
            
        except Exception as e:
            error_msg = f"Unexpected error in batch {batch_id}: {e}"
            logger.error(error_msg)
            return (batch_id, 0, len(batch_personas))


def sort_output_file(output_file: Path) -> None:
    """
    Sort the output JSONL file by ID field.
    Reads all results, sorts by ID, and rewrites the file.
    """
    if not output_file.exists():
        logger.warning(f"Output file does not exist for sorting: {output_file}")
        return

    logger.info(f"Sorting output file by ID: {output_file}")

    # Read all results
    results = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                results.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON while sorting on line {line_num}: {e}")
                continue

    logger.info(f"Read {len(results)} results for sorting")

    # Sort by ID field
    results.sort(key=lambda x: x.get('id', -1))

    # Write sorted results back to file with ID first
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            # Ensure ID is first in the output
            ordered_result = {'id': result.get('id')}
            ordered_result.update({k: v for k, v in result.items() if k != 'id'})
            f.write(json.dumps(ordered_result, ensure_ascii=False) + '\n')

    logger.info(f"Sorted and rewrote {len(results)} results to {output_file}")


def filter_personas_for_processing(
    personas: List[Dict[str, Any]],
    existing_results: Set[int],
    overwrite: bool = False,
    test_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Filter personas based on existing results and processing options.

    Args:
        personas: List of persona dictionaries with ID fields
        existing_results: Set of already processed IDs
        overwrite: If True, process all personas regardless of existing results
        test_mode: If True, limit to first 3 personas

    Returns:
        Filtered list of personas to process
    """
    if test_mode:
        personas = personas[:3]
        logger.info(f"Test mode: Limited to {len(personas)} personas")

    if not overwrite:
        original_count = len(personas)
        personas = [p for p in personas if p['id'] not in existing_results]
        skipped_count = original_count - len(personas)
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already processed personas")

    logger.info(f"Will process {len(personas)} personas")
    return personas


async def main():
    """Main async function."""
    args = parse_arguments()
    
    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Load input personas
    input_file = Path(args.input_file)
    logger.info(f"Loading personas from {input_file}")
    personas = load_personas(input_file)
    
    if not personas:
        logger.error("No personas loaded")
        sys.exit(1)
    
    # Setup output file and load existing results for restart support
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Checking for existing results in: {output_file}")
    existing_results = load_existing_results(output_file)
    
    # Filter personas for processing
    personas_to_process = filter_personas_for_processing(
        personas, existing_results, args.overwrite, args.test_mode
    )
    
    if not personas_to_process:
        logger.info("No personas to process (all may already be completed)")
        return
    
    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        logger.info(f"Total personas that would be processed: {len(personas_to_process)}")
        
        # Show complete example prompt
        if personas_to_process:
            example_persona = personas_to_process[0]
            domain = example_persona['domain']
            persona = example_persona['persona']
            formatted_prompt = format_persona_prompt(domain, persona)
            
            logger.info(f"\nAPI call configuration:")
            logger.info(f"  Model: {args.model}")
            logger.info(f"  Max tokens: {args.max_tokens}")
            logger.info(f"  Temperature: 0.7")
            logger.info(f"  Batch size: {args.batch_size}")
            logger.info(f"  Max workers: {args.max_workers}")
            logger.info(f"  Requests per second: {args.requests_per_second}")
            
            logger.info(f"\nExample persona data:")
            logger.info(f"  Domain: {domain}")
            logger.info(f"  Persona: {persona[:200]}{'...' if len(persona) > 200 else ''}")
            
            logger.info(f"\nComplete formatted prompt that would be sent:")
            logger.info(f"Messages: [")
            logger.info(f"  {{")
            logger.info(f"    'role': 'user',")
            logger.info(f"    'content': '''")
            logger.info(formatted_prompt)
            logger.info(f"    '''")
            logger.info(f"  }}")
            logger.info(f"]")
        return
    
    logger.info(f"Results will be appended to: {output_file}")
    
    # Initialize OpenRouter client
    client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    logger.info(f"Initialized OpenRouter client with model: {args.model}")
    
    # Initialize concurrency controls
    rate_limiter = RateLimiter(args.requests_per_second)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.max_workers)
    
    logger.info(f"Starting concurrent processing with {args.max_workers} workers, {args.requests_per_second} req/s limit")
    
    # Split personas into batches for processing
    batch_size = args.batch_size
    batches = []
    for i in range(0, len(personas_to_process), batch_size):
        batch = personas_to_process[i:i + batch_size]
        batches.append(batch)
    
    logger.info(f"Split {len(personas_to_process)} personas into {len(batches)} batches of up to {batch_size} personas each")
    
    # Create worker tasks for all batches
    tasks = []
    for batch_id, batch_personas in enumerate(batches):
        task = batch_worker(
            batch_personas=batch_personas,
            client=client,
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
            
            logger.info(f"✅ [{completed_count}/{total_count}] Batch {batch_id} completed: {successful_count}/{batch_total} personas")
                
        except Exception as e:
            completed_count += 1
            logger.error(f"❌ [{completed_count}/{total_count}] Batch exception: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing complete: {total_successful}/{total_items} personas processed successfully in {elapsed_time:.1f}s")
    
    if total_successful < total_items:
        failed_count = total_items - total_successful
        logger.warning(f"{failed_count} personas failed to process - check logs for details")

    # Sort the output file by ID
    if total_successful > 0:
        logger.info("Sorting output file by ID...")
        sort_output_file(output_file)
        logger.info("Output file sorting complete")


if __name__ == "__main__":
    asyncio.run(main())