#!/usr/bin/env python3
"""
Jailbreak Evaluation Script with Prefills

This script runs evaluations using conversation histories from prefill files as context,
ignoring system prompts and posing questions directly to models after prefilling with
each conversation history.

Output: JSONL with evaluation results including prefill context

Example usage:
uv run 3_jailbreak_prefills.py \
    --prefill_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/prefills/role_pc1_prefills.jsonl \
    --questions_file /root/git/persona-subspace/evals/data/harmbench/harmbench.jsonl \
    --output_jsonl /root/git/persona-subspace/dynamics/results/jailbreak_prefills_eval.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))

from utils.inference_utils import load_vllm_model, batch_conversation_chat, close_vllm_model


def read_existing_results(output_jsonl: str, samples_per_prompt: int = 1) -> Set[str]:
    """Read existing results to avoid duplicates."""
    existing = set()
    if not os.path.exists(output_jsonl):
        return existing

    try:
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Create unique key from id, prefill_id, and sample_id
                    if samples_per_prompt > 1:
                        # When multiple samples per prompt, use (id, prefill_id, sample_id) as key
                        sample_id = data.get('sample_id', 0)
                        key = (data['id'], data.get('prefill_id', 0), sample_id)
                    else:
                        # Backwards compatibility: just use (id, prefill_id) for single samples
                        key = (data['id'], data.get('prefill_id', 0))
                    existing.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Could not read existing results: {e}")

    return existing


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate baseline (non-steered) responses using vLLM batch processing with prefill conversations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--questions_file",
        type=str,
        required=True,
        help="Path to questions JSONL file with id and semantic_category fields"
    )

    parser.add_argument(
        "--prefill_file",
        type=str,
        required=True,
        help="Path to prefill JSONL file with id and conversation fields"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-27b-it",
        help="Model name for inference"
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation"
    )

    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Test mode: only process first question with all prefills"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="Specific GPU ID to use (if not specified, will auto-detect and use all available GPUs)"
    )

    parser.add_argument(
        "--samples_per_prompt",
        type=int,
        default=1,
        help="Number of samples to generate for each unique question x prefill combination"
    )

    return parser.parse_args()


def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file."""
    print(f"Loading questions from {questions_file}")

    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    questions = []
    with open(questions_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                question_obj = json.loads(line.strip())

                # Only validate presence of question field
                if 'question' not in question_obj:
                    print(f"Warning: No 'question' field found on line {line_num}")
                    continue

                question_text = question_obj['question']

                # Store the original object with the identified text field
                question_obj['_question_text'] = question_text
                questions.append(question_obj)

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")

    print(f"Loaded {len(questions)} questions")
    return questions


def load_prefills(prefill_file: str) -> List[Dict[str, Any]]:
    """Load prefills from JSONL file."""
    print(f"Loading prefills from {prefill_file}")

    if not os.path.exists(prefill_file):
        raise FileNotFoundError(f"Prefill file not found: {prefill_file}")

    prefills = []
    with open(prefill_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                prefill_obj = json.loads(line.strip())

                # Validate presence of required fields
                if 'id' not in prefill_obj or 'conversation' not in prefill_obj:
                    print(f"Warning: Missing 'id' or 'conversation' field on line {line_num}")
                    continue

                prefills.append(prefill_obj)

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")

    print(f"Loaded {len(prefills)} prefills")
    return prefills


def generate_all_prompts(prefills: List[Dict[str, Any]], questions: List[Dict[str, Any]], existing_results: Set[str], samples_per_prompt: int = 1) -> List[Dict[str, Any]]:
    """Generate all prefill-question combination prompts, filtering out existing ones."""
    prompts_data = []
    skipped_count = 0

    for prefill in prefills:
        for question in questions:
            question_text = question['_question_text']

            # Generate multiple samples for this prompt
            for sample_id in range(samples_per_prompt):
                # Create unique key for this sample
                if samples_per_prompt > 1:
                    unique_key = (question.get('id', 0), prefill['id'], sample_id)
                else:
                    unique_key = (question.get('id', 0), prefill['id'])

                # Check if this specific sample already exists
                if unique_key in existing_results:
                    skipped_count += 1
                    continue

                # Combine all fields from both prefill and question
                prompt_data = {}
                prompt_data.update(prefill)
                prompt_data.update(question)

                # Store question text and prefill info
                prompt_data['_user_message'] = question_text
                prompt_data['_conversation_prefill'] = prefill['conversation']
                prompt_data['prefill_id'] = prefill['id']

                # Add sample_id if multiple samples per prompt
                if samples_per_prompt > 1:
                    prompt_data['sample_id'] = sample_id

                # Clean up temporary fields
                if '_question_text' in prompt_data:
                    del prompt_data['_question_text']

                prompts_data.append(prompt_data)

    if skipped_count > 0:
        print(f"Filtered out {skipped_count} existing combinations")

    return prompts_data


def format_messages_for_chat(prompts_data: List[Dict[str, Any]], model_name: str) -> List[List[Dict[str, str]]]:
    """Format prompts into chat messages for vLLM batch processing."""
    formatted_messages = []

    for prompt_data in prompts_data:
        user_message = prompt_data.get('_user_message', '')
        conversation_prefill = prompt_data.get('_conversation_prefill', [])

        # Start with conversation prefill
        messages = []
        if conversation_prefill:
            messages.extend(conversation_prefill)

        # Add the current question as a user message
        messages.append({"role": "user", "content": user_message})

        formatted_messages.append(messages)

    return formatted_messages


def write_results_to_jsonl(prompts_data: List[Dict[str, Any]], responses: List[str], output_jsonl: str, existing_results: Set[str], samples_per_prompt: int = 1):
    """Write results to JSONL file, appending new results only."""
    print(f"Writing results to {output_jsonl}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    written_count = 0
    skipped_count = 0

    # Use append mode to add new results
    with open(output_jsonl, 'a', encoding='utf-8') as f:
        for prompt_data, response in zip(prompts_data, responses):
            # Create unique key for this sample (matching logic in read_existing_results)
            if samples_per_prompt > 1:
                sample_id = prompt_data.get('sample_id', 0)
                unique_key = (prompt_data['id'], prompt_data.get('prefill_id', 0), sample_id)
            else:
                unique_key = (prompt_data['id'], prompt_data.get('prefill_id', 0))

            # Check if this specific sample already exists
            if unique_key in existing_results:
                skipped_count += 1
                continue

            row_data = prompt_data.copy()  # Copy all original fields
            row_data['response'] = response
            row_data['magnitude'] = prompt_data.get('pc1', 0.0)

            # Remove underscore fields from output
            fields_to_remove = [k for k in row_data.keys() if k.startswith('_')]
            for field in fields_to_remove:
                del row_data[field]

            f.write(json.dumps(row_data) + '\n')
            written_count += 1

    print(f"Successfully wrote {written_count} new results to {output_jsonl}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing results")


def main():
    """Main function to generate jailbreak evaluation responses with prefills."""
    args = parse_arguments()

    print("="*60)
    print("Jailbreak Evaluation with Prefill Conversations")
    print("="*60)

    # Check for existing results to enable restart functionality
    existing_results = read_existing_results(args.output_jsonl, args.samples_per_prompt)
    if existing_results:
        print(f"Found {len(existing_results)} existing results, will skip duplicates")

    # Load and validate inputs
    questions = load_questions(args.questions_file)
    prefills = load_prefills(args.prefill_file)

    # Apply test mode filtering
    if args.test_mode:
        if len(questions) > 0:
            questions = questions[:1]  # Keep only first question
            print(f"TEST MODE: Using only first question with all {len(prefills)} prefills")
        else:
            print("Warning: No questions available for test mode")

    # Generate all prompts
    print(f"Generating prompts for {len(questions)} questions and {len(prefills)} prefills")
    prompts_data = generate_all_prompts(prefills, questions, existing_results, args.samples_per_prompt)
    print(f"Generated {len(prompts_data)} total prompts")

    # Check if we have any new prompts to process after filtering
    if len(prompts_data) == 0:
        print("No new prompts to process - all combinations already exist!")
        return

    # Format messages for chat-based processing with conversation prefill
    messages_list = format_messages_for_chat(prompts_data, args.model_name)

    # Load vLLM model with GPU configuration
    print(f"Loading vLLM model: {args.model_name}")

    # Set GPU configuration based on gpu_id flag
    if args.gpu_id is not None:
        # Use specific GPU
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        tensor_parallel_size = 1
        print(f"Using GPU {args.gpu_id}")
    else:
        # Auto-detect all available GPUs
        tensor_parallel_size = None
        print("Auto-detecting available GPUs")

    model_wrapper = load_vllm_model(
        args.model_name,
        max_model_len=8192,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9
    )

    try:
        # Generate responses using batch processing with formatted messages
        print(f"Generating responses with batch processing...")
        responses = batch_conversation_chat(
            model_wrapper=model_wrapper,
            conversations=messages_list,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            progress=True
        )

        # Write results to JSONL
        write_results_to_jsonl(prompts_data, responses, args.output_jsonl, existing_results, args.samples_per_prompt)

        print(f"\nJailbreak evaluation completed!")
        print(f"Results saved to: {args.output_jsonl}")
        print(f"Processed {len(prefills)} prefill conversations with {len(questions)} questions each")

    finally:
        # Clean up model
        close_vllm_model(model_wrapper)


if __name__ == "__main__":
    main()