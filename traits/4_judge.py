#!/usr/bin/env python3
"""
Script to evaluate trait responses using a judge model via the OpenAI API.

This script takes evaluation prompts for traits from the instructions directory,
pairs them with responses from the responses directory, and uses a judge model
to score how well each response exhibits the target trait.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
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

# Constants
INSTRUCTIONS_DIR = Path(__file__).parent / "data" / "instructions"
RESPONSES_DIR = Path("/workspace/traits/responses")
OUTPUT_DIR = Path(__file__).parent / "data" / "extract_scores"
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trait responses using a judge model"
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        help="Specific traits to process. If not provided, processes all available traits."
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model to use for evaluation (default: {DEFAULT_JUDGE_MODEL})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
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
    
    return parser.parse_args()


def get_available_traits() -> List[str]:
    """Get list of available traits from the instructions directory."""
    if not INSTRUCTIONS_DIR.exists():
        logger.error(f"Instructions directory not found: {INSTRUCTIONS_DIR}")
        return []
    
    traits = []
    for file_path in INSTRUCTIONS_DIR.glob("*.json"):
        trait_name = file_path.stem
        # Check if corresponding response file exists
        response_file = RESPONSES_DIR / f"{trait_name}.jsonl"
        if response_file.exists():
            traits.append(trait_name)
        else:
            logger.warning(f"No response file found for trait: {trait_name}")
    
    return sorted(traits)


def load_instruction_data(trait: str) -> Dict[str, Any]:
    """Load instruction data for a given trait."""
    instruction_file = INSTRUCTIONS_DIR / f"{trait}.json"
    
    if not instruction_file.exists():
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
    
    with open(instruction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required_keys = ['eval_prompt', 'questions']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in instruction file: {instruction_file}")
    
    return data


def load_response_data(trait: str) -> List[Dict[str, Any]]:
    """Load response data for a given trait."""
    response_file = RESPONSES_DIR / f"{trait}.jsonl"
    
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


def organize_responses_by_question_and_label(responses: List[Dict[str, Any]]) -> Dict[Tuple[int, str], str]:
    """
    Organize responses by (question_index, label) for easy lookup.
    
    Returns:
        Dict mapping (question_index, label) to the assistant's response text
    """
    organized = {}
    
    for response_data in responses:
        question_index = response_data.get('question_index')
        label = response_data.get('label')
        conversation = response_data.get('conversation', [])
        
        if question_index is None or label is None:
            logger.warning(f"Missing question_index or label in response: {response_data}")
            continue
        
        # Extract assistant's response from conversation
        assistant_response = None
        for message in conversation:
            if message.get('role') == 'assistant':
                assistant_response = message.get('content', '')
                break
        
        if assistant_response is None:
            logger.warning(f"No assistant response found for question {question_index}, label {label}")
            continue
        
        organized[(question_index, label)] = assistant_response
    
    return organized


def create_evaluation_prompts(
    trait: str,
    instruction_data: Dict[str, Any], 
    organized_responses: Dict[Tuple[int, str], str]
) -> List[Tuple[str, int, str, str]]:
    """
    Create evaluation prompts for the judge model.
    
    Returns:
        List of tuples: (trait, question_index, label, prompt_text)
    """
    eval_prompt_template = instruction_data['eval_prompt']
    questions = instruction_data['questions']
    
    evaluation_prompts = []
    
    for question_index, question in enumerate(questions):
        for label in ['pos', 'neg', 'default']:
            key = (question_index, label)
            
            if key not in organized_responses:
                logger.warning(f"No response found for trait {trait}, question {question_index}, label {label}")
                continue
            
            answer = organized_responses[key]
            
            # Fill in the template
            filled_prompt = eval_prompt_template.format(
                question=question,
                answer=answer
            )
            
            evaluation_prompts.append((trait, question_index, label, filled_prompt))
    
    return evaluation_prompts


def parse_judge_score(response_text: str):
    """
    Parse the judge's response to extract the numerical score or REFUSAL.
    
    Returns:
        Integer score between 0-100, "REFUSAL" string, or None if parsing fails
    """
    if not response_text:
        return None
    
    # Handle REFUSAL case
    if "REFUSAL" in response_text.upper():
        return "REFUSAL"
    
    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())
    
    if not numbers:
        logger.warning(f"No numbers found in judge response: {response_text}")
        return None
    
    # Take the first number found
    try:
        score = int(numbers[0])
        if 0 <= score <= 100:
            return score
        else:
            logger.warning(f"Score out of range (0-100): {score}")
            return None
    except ValueError:
        logger.warning(f"Could not parse score from: {numbers[0]}")
        return None


async def call_judge_model(
    client: openai.AsyncOpenAI,
    prompts: List[str],
    model: str,
    max_tokens: int,
    batch_size: int = 100
) -> List[Optional[str]]:
    """Call the judge model with a list of prompts."""
    results = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_results = []
        
        # Process each prompt in the batch
        for prompt in batch:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0  # For consistent evaluation
                )
                
                if response.choices and response.choices[0].message.content:
                    batch_results.append(response.choices[0].message.content)
                else:
                    batch_results.append(None)
                    
            except Exception as e:
                logger.error(f"Error calling judge model: {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(prompts) - 1)//batch_size + 1}")
    
    return results


async def process_trait(
    trait: str,
    client: openai.AsyncOpenAI,
    judge_model: str,
    max_tokens: int,
    batch_size: int
) -> Optional[Dict[str, int]]:
    """
    Process a single trait and return the organized scores.
    
    Returns:
        Dict with keys like "pos_0", "neg_1", "default_2", etc. mapping to scores
    """
    try:
        # Load data
        logger.info(f"Loading data for trait: {trait}")
        instruction_data = load_instruction_data(trait)
        response_data = load_response_data(trait)
        
        # Organize responses
        organized_responses = organize_responses_by_question_and_label(response_data)
        logger.info(f"Found {len(organized_responses)} response entries for trait: {trait}")
        
        # Create evaluation prompts
        evaluation_prompts = create_evaluation_prompts(trait, instruction_data, organized_responses)
        logger.info(f"Created {len(evaluation_prompts)} evaluation prompts for trait: {trait}")
        
        if not evaluation_prompts:
            logger.warning(f"No evaluation prompts created for trait: {trait}")
            return None
        
        # Extract just the prompt texts for the API call
        prompt_texts = [prompt_text for _, _, _, prompt_text in evaluation_prompts]
        
        # Call the judge model
        logger.info(f"Sending {len(prompt_texts)} prompts to judge model for trait: {trait}")
        responses = await call_judge_model(
            client=client,
            prompts=prompt_texts,
            model=judge_model,
            max_tokens=max_tokens,
            batch_size=batch_size
        )
        
        logger.info(f"Received {len(responses)} responses for trait: {trait}")
        
        # Parse scores and organize results
        organized_scores = {}
        
        for i, (trait_name, question_index, label, prompt_text) in enumerate(evaluation_prompts):
            if i < len(responses) and responses[i] is not None:
                response_text = responses[i]
                score = parse_judge_score(response_text)
                
                if score is not None:
                    key = f"{label}_{question_index}"
                    organized_scores[key] = score
                else:
                    logger.warning(f"Failed to parse score for {trait}, question {question_index}, label {label}")
            else:
                logger.warning(f"No response received for {trait}, question {question_index}, label {label}")
        
        logger.info(f"Successfully parsed {len(organized_scores)} scores for trait: {trait}")
        return organized_scores
        
    except Exception as e:
        logger.error(f"Error processing trait {trait}: {e}")
        return None


async def main():
    """Main async function."""
    args = parse_arguments()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Get traits to process
    if args.traits:
        traits_to_process = args.traits
    else:
        traits_to_process = get_available_traits()
    
    if not traits_to_process:
        logger.error("No traits to process")
        sys.exit(1)
    
    logger.info(f"Processing {len(traits_to_process)} traits: {', '.join(traits_to_process[:5])}{'...' if len(traits_to_process) > 5 else ''}")
    
    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        total_prompts = 0
        example_call_shown = False
        
        for trait in traits_to_process:
            try:
                # Load data to count prompts
                instruction_data = load_instruction_data(trait)
                response_data = load_response_data(trait)
                organized_responses = organize_responses_by_question_and_label(response_data)
                evaluation_prompts = create_evaluation_prompts(trait, instruction_data, organized_responses)
                
                trait_prompt_count = len(evaluation_prompts)
                total_prompts += trait_prompt_count
                
                logger.info(f"Would process trait: {trait} ({trait_prompt_count} prompts)")
                
                # Show one example API call request
                if not example_call_shown and evaluation_prompts:
                    _, _, _, example_prompt = evaluation_prompts[0]
                    logger.info("\nExample API call request:")
                    logger.info(f"  Model: {args.judge_model}")
                    logger.info(f"  Max tokens: {args.max_tokens}")
                    logger.info(f"  Temperature: 0.0")
                    logger.info(f"  Message content preview: {example_prompt}...")
                    example_call_shown = True
                    
            except Exception as e:
                logger.warning(f"Could not count prompts for trait {trait}: {e}")
        
        logger.info(f"\nTotal prompts that would be sent to API: {total_prompts}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Initialize OpenAI client
    client = openai.AsyncOpenAI()
    logger.info(f"Initialized OpenAI client with judge model: {args.judge_model}")
    
    # Process traits
    successful_traits = 0
    failed_traits = 0
    
    for trait in traits_to_process:
        logger.info(f"Processing trait: {trait}")
        
        try:
            scores = await process_trait(
                trait=trait,
                client=client,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size
            )
            
            if scores:
                # Save results
                output_file = OUTPUT_DIR / f"{trait}.json"  # Changed to .json for better readability
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {len(scores)} scores to {output_file}")
                successful_traits += 1
            else:
                logger.error(f"Failed to process trait: {trait}")
                failed_traits += 1
                
        except Exception as e:
            logger.error(f"Unexpected error processing trait {trait}: {e}")
            failed_traits += 1
    
    logger.info(f"Processing complete: {successful_traits} successful, {failed_traits} failed")


if __name__ == "__main__":
    asyncio.run(main())