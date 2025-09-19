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

# Default constants
DEFAULT_INSTRUCTIONS_DIR = Path(__file__).parent / "data" / "instructions"
DEFAULT_RESPONSES_DIR = Path("/workspace/roles_240/responses")
DEFAULT_OUTPUT_DIR = Path("/workspace/roles_240/extract_labels")
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
        "--roles",
        nargs="+",
        help="Specific roles to process. If not provided, processes all available roles."
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
        help="Batch size for API requests (default: 100)"
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
        help="Maximum number of concurrent role workers (default: 10)"
    )
    parser.add_argument(
        "--requests-per-second",
        type=int,
        default=400,
        help="Rate limit for API requests per second across all workers (default: 400)"
    )
    parser.add_argument(
        "--instructions-dir",
        type=Path,
        default=DEFAULT_INSTRUCTIONS_DIR,
        help=f"Directory containing instruction files (default: {DEFAULT_INSTRUCTIONS_DIR})"
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=DEFAULT_RESPONSES_DIR,
        help=f"Directory containing response files (default: {DEFAULT_RESPONSES_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output files (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    return parser.parse_args()


def get_available_roles(instructions_dir: Path, responses_dir: Path) -> List[str]:
    """Get list of available roles from the instructions directory."""
    if not instructions_dir.exists():
        logger.error(f"Instructions directory not found: {instructions_dir}")
        return []
    
    roles = []
    for file_path in instructions_dir.glob("*.json"):
        role_name = file_path.stem
        # Check if corresponding response file exists
        response_file = responses_dir / f"{role_name}.jsonl"
        if response_file.exists():
            roles.append(role_name)
        else:
            logger.warning(f"No response file found for role: {role_name}")
    
    return sorted(roles)


def load_instruction_data(role: str, instructions_dir: Path) -> Dict[str, Any]:
    """Load instruction data for a given role."""
    instruction_file = instructions_dir / f"{role}.json"
    
    if not instruction_file.exists():
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
    
    with open(instruction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required_keys = ['eval_prompt', 'questions']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in instruction file: {instruction_file}")
    
    return data


def load_response_data(role: str, responses_dir: Path) -> List[Dict[str, Any]]:
    """Load response data for a given role."""
    response_file = responses_dir / f"{role}.jsonl"
    
    if not response_file.exists():
        raise FileNotFoundError(f"Response file not found: {response_file}")
    
    responses = []
    with open(response_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                response_data = json.loads(line)
                responses.append(response_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_num} in {response_file}: {e}")
                continue
    
    return responses


def organize_responses_by_question_and_label(responses: List[Dict[str, Any]]) -> Dict[Tuple[int, int, str], Tuple[str, str]]:
    """
    Organize responses by (prompt_index, question_index, label) for easy lookup.
    
    Returns:
        Dict mapping (prompt_index, question_index, label) to (question_text, assistant_response)
    """
    organized = {}
    
    for response_data in responses:
        question_index = response_data.get('question_index')
        label = response_data.get('label')
        # Handle both old format (no prompt_index) and new format (with prompt_index)
        prompt_index = response_data.get('prompt_index', 0)  # Default to 0 for backward compatibility
        conversation = response_data.get('conversation', [])
        question_text = response_data.get('question', '')
        
        # Normalize neutral labels to default for consistency
        if label == 'neutral':
            label = 'default'
        
        if question_index is None or label is None:
            logger.warning(f"Missing question_index or label in response: {response_data}")
            continue
        
        if not question_text:
            logger.warning(f"Missing question text for prompt {prompt_index}, question {question_index}, label {label}")
            continue
        
        # Extract assistant's response from conversation
        assistant_response = None
        for message in conversation:
            if message.get('role') == 'assistant':
                assistant_response = message.get('content', '')
                break
        
        if assistant_response is None:
            logger.warning(f"No assistant response found for prompt {prompt_index}, question {question_index}, label {label}")
            continue
        
        organized[(prompt_index, question_index, label)] = (question_text, assistant_response)
    
    return organized


def create_evaluation_prompts(
    role: str,
    instruction_data: Dict[str, Any], 
    organized_responses: Dict[Tuple[int, int, str], Tuple[str, str]]
) -> List[Tuple[str, int, int, str, str]]:
    """
    Create evaluation prompts for the judge model.
    
    Returns:
        List of tuples: (role, prompt_index, question_index, label, prompt_text)
    """
    eval_prompt_template = instruction_data['eval_prompt']
    
    evaluation_prompts = []
    
    # Process each response combination directly
    for (prompt_index, question_index, label), (question_text, answer) in organized_responses.items():
        # Fill in the template using the question text from the response
        filled_prompt = eval_prompt_template.format(
            question=question_text,
            answer=answer
        )
        
        evaluation_prompts.append((role, prompt_index, question_index, label, filled_prompt))
    
    return evaluation_prompts


def convert_old_scores_format(old_scores: Dict[str, int]) -> Dict[str, int]:
    """
    Convert old format scores (label_question) to new format (label_p0_q{question}).
    
    Args:
        old_scores: Dict with keys like "pos_0", "neg_1", "default_2"
        
    Returns:
        Dict with keys like "pos_p0_q0", "neg_p0_q1", "default_p0_q2"
    """
    new_scores = {}
    
    for old_key, score in old_scores.items():
        # Parse old key format: label_question_index
        parts = old_key.split('_')
        if len(parts) == 2:
            label, question_index = parts
            # Normalize neutral to default
            if label == 'neutral':
                label = 'default'
            new_key = f"{label}_p0_q{question_index}"
            new_scores[new_key] = score
        else:
            # If it's already in new format, normalize neutral to default
            if old_key.startswith('neutral_'):
                normalized_key = old_key.replace('neutral_', 'default_', 1)
                new_scores[normalized_key] = score
            else:
                new_scores[old_key] = score
    
    return new_scores


def parse_judge_score(response_text: str):
    """
    Parse the judge's response to extract the numerical score.
    
    Returns:
        Integer score between 0-3, or None if parsing fails
    """
    if not response_text:
        return None
    
    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())
    
    if not numbers:
        logger.warning(f"No numbers found in judge response: {response_text}")
        return None
    
    # Take the first number found
    try:
        score = int(numbers[0])
        if 0 <= score <= 3:
            return score
        else:
            logger.warning(f"Score out of range (0-3): {score}")
            return None
    except ValueError:
        logger.warning(f"Could not parse score from: {numbers[0]}")
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
            max_completion_tokens=max_tokens,
            temperature=1  # For consistent evaluation
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


async def process_role(
    role: str,
    client: openai.AsyncOpenAI,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    rate_limiter: RateLimiter,
    instructions_dir: Path,
    responses_dir: Path,
    output_dir: Path
) -> Optional[Dict[str, int]]:
    """
    Process a single role and return the organized scores.
    
    Returns:
        Dict with keys like "pos_0", "neg_1", "default_2", etc. mapping to scores
    """
    try:
        # Load data
        logger.info(f"Loading data for role: {role}")
        instruction_data = load_instruction_data(role, instructions_dir)
        response_data = load_response_data(role, responses_dir)
        
        # Load existing scores first to check what we already have
        output_file = output_dir / f"{role}.json"
        existing_scores = {}
        
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_scores = json.load(f)
                
                # Convert old format to new format if needed
                existing_scores = convert_old_scores_format(existing_scores)
                logger.info(f"Loaded and converted {len(existing_scores)} existing scores for role: {role}")
            except Exception as e:
                logger.warning(f"Failed to load existing scores for role {role}: {e}")
        
        # Organize responses
        organized_responses = organize_responses_by_question_and_label(response_data)
        logger.info(f"Found {len(organized_responses)} response entries for role: {role}")
        
        # Create evaluation prompts
        evaluation_prompts = create_evaluation_prompts(role, instruction_data, organized_responses)
        logger.info(f"Created {len(evaluation_prompts)} evaluation prompts for role: {role}")
        
        if not evaluation_prompts:
            logger.warning(f"No evaluation prompts created for role: {role}")
            return existing_scores if existing_scores else None
        
        # Filter out prompts that already have scores
        prompts_to_evaluate = []
        for prompt_data in evaluation_prompts:
            role_name, prompt_index, question_index, label, prompt_text = prompt_data
            key = f"{label}_p{prompt_index}_q{question_index}"
            
            if key not in existing_scores:
                prompts_to_evaluate.append(prompt_data)
        
        if not prompts_to_evaluate:
            logger.info(f"All scores already processed for role: {role}")
            return existing_scores
        
        logger.info(f"Need to evaluate {len(prompts_to_evaluate)} new prompts (skipping {len(evaluation_prompts) - len(prompts_to_evaluate)} existing scores)")
        
        # Extract just the prompt texts for the API call
        prompt_texts = [prompt_text for _, _, _, _, prompt_text in prompts_to_evaluate]
        
        # Call the judge model
        logger.info(f"Sending {len(prompt_texts)} prompts to judge model for role: {role}")
        responses = await call_judge_model(
            client=client,
            prompts=prompt_texts,
            model=judge_model,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
            batch_size=batch_size
        )
        
        logger.info(f"Received {len(responses)} responses for role: {role}")
        
        # Parse scores and organize results (only for new prompts)
        organized_scores = {}
        
        for i, (role_name, prompt_index, question_index, label, prompt_text) in enumerate(prompts_to_evaluate):
            if i < len(responses) and responses[i] is not None:
                response_text = responses[i]
                score = parse_judge_score(response_text)
                
                if score is not None:
                    key = f"{label}_p{prompt_index}_q{question_index}"
                    organized_scores[key] = score
                else:
                    logger.warning(f"Failed to parse score for {role}, prompt {prompt_index}, question {question_index}, label {label}")
            else:
                logger.warning(f"No response received for {role}, prompt {prompt_index}, question {question_index}, label {label}")
        
        logger.info(f"Successfully parsed {len(organized_scores)} new scores for role: {role}")
        
        # Merge existing scores with new scores (new scores take precedence)
        final_scores = existing_scores.copy()
        final_scores.update(organized_scores)
        
        logger.info(f"Final scores for role {role}: {len(final_scores)} total ({len(existing_scores)} existing + {len(organized_scores)} new)")
        return final_scores
        
    except Exception as e:
        logger.error(f"Error processing role {role}: {e}")
        return None


async def process_role_worker(
    role: str,
    client: openai.AsyncOpenAI,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    instructions_dir: Path,
    responses_dir: Path,
    output_dir: Path
) -> Tuple[str, bool, Optional[str]]:
    """
    Worker function to process a single role.
    
    Returns:
        Tuple of (role_name, success, error_message)
    """
    async with semaphore:
        try:
            logger.info(f"🚀 Starting: {role}")
            
            scores = await process_role(
                role=role,
                client=client,
                judge_model=judge_model,
                max_tokens=max_tokens,
                batch_size=batch_size,
                rate_limiter=rate_limiter,
                instructions_dir=instructions_dir,
                responses_dir=responses_dir,
                output_dir=output_dir
            )
            
            if scores:
                # Save results
                output_file = output_dir / f"{role}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Worker completed role: {role} ({len(scores)} scores)")
                return (role, True, None)
            else:
                error_msg = f"Failed to process role: {role}"
                logger.error(error_msg)
                return (role, False, error_msg)
                
        except Exception as e:
            error_msg = f"Unexpected error processing role {role}: {e}"
            logger.error(error_msg)
            return (role, False, error_msg)


async def main():
    """Main async function."""
    args = parse_arguments()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Get roles to process
    if args.roles:
        roles_to_process = args.roles
    else:
        roles_to_process = get_available_roles(args.instructions_dir, args.responses_dir)
    
    if not roles_to_process:
        logger.error("No roles to process")
        sys.exit(1)
    
    logger.info(f"Processing {len(roles_to_process)} roles: {', '.join(roles_to_process[:5])}{'...' if len(roles_to_process) > 5 else ''}")
    
    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        total_prompts = 0
        example_call_shown = False
        
        for role in roles_to_process:
            try:
                # Load data to count prompts
                instruction_data = load_instruction_data(role, args.instructions_dir)
                response_data = load_response_data(role, args.responses_dir)
                organized_responses = organize_responses_by_question_and_label(response_data)
                evaluation_prompts = create_evaluation_prompts(role, instruction_data, organized_responses)
                
                role_prompt_count = len(evaluation_prompts)
                total_prompts += role_prompt_count
                
                logger.info(f"Would process role: {role} ({role_prompt_count} prompts)")
                
                # Show one example API call request
                if not example_call_shown and evaluation_prompts:
                    _, _, _, _, example_prompt = evaluation_prompts[0]
                    logger.info("\n" + "=" * 80)
                    logger.info("SAMPLE JUDGE MESSAGE:")
                    logger.info("=" * 80)
                    logger.info(f"Model: {args.judge_model}")
                    logger.info(f"Max tokens: {args.max_tokens}")
                    logger.info(f"Temperature: 0.0")
                    logger.info("\nFull message content:")
                    logger.info("-" * 40)
                    logger.info(example_prompt)
                    logger.info("-" * 40)
                    logger.info("=" * 80)
                    example_call_shown = True
                    
            except Exception as e:
                logger.warning(f"Could not count prompts for role {role}: {e}")
        
        logger.info(f"\nTotal prompts that would be sent to API: {total_prompts}")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize OpenAI client
    client = openai.AsyncOpenAI()
    logger.info(f"Initialized OpenAI client with judge model: {args.judge_model}")
    
    # Initialize concurrency controls
    rate_limiter = RateLimiter(args.requests_per_second)
    semaphore = asyncio.Semaphore(args.max_workers)
    
    logger.info(f"Starting concurrent processing with {args.max_workers} workers, {args.requests_per_second} req/s limit")
    
    # Create worker tasks for all roles
    tasks = []
    for role in roles_to_process:
        task = process_role_worker(
            role=role,
            client=client,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            rate_limiter=rate_limiter,
            semaphore=semaphore,
            instructions_dir=args.instructions_dir,
            responses_dir=args.responses_dir,
            output_dir=args.output_dir
        )
        tasks.append(task)
    
    # Process all roles concurrently with progress tracking
    start_time = time.time()
    logger.info(f"Launching {len(tasks)} concurrent workers...")
    
    # Track progress using asyncio.as_completed
    completed_count = 0
    total_count = len(tasks)
    results = []
    
    # Use as_completed to show progress as tasks finish
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            completed_count += 1
            results.append(result)
            
            if isinstance(result, tuple):
                role_name, success, error_msg = result
                if success:
                    logger.info(f"✅ [{completed_count}/{total_count}] Completed: {role_name}")
                else:
                    logger.info(f"❌ [{completed_count}/{total_count}] Failed: {role_name}")
            else:
                logger.info(f"⚠️ [{completed_count}/{total_count}] Unexpected result format")
                
        except Exception as e:
            completed_count += 1
            results.append(e)
            logger.error(f"❌ [{completed_count}/{total_count}] Exception: {e}")
    
    # Process results
    successful_roles = 0
    failed_roles = 0
    errors = []
    
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Worker exception: {result}")
            failed_roles += 1
            errors.append(str(result))
        else:
            role_name, success, error_msg = result
            if success:
                successful_roles += 1
            else:
                failed_roles += 1
                if error_msg:
                    errors.append(error_msg)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing complete: {successful_roles} successful, {failed_roles} failed in {elapsed_time:.1f}s")
    
    if errors:
        logger.info("Errors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            logger.info(f"  - {error}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    asyncio.run(main())