#!/usr/bin/env python3
"""
Baseline Response Generation Script

This script generates default (non-steered) responsesusing vLLM's batch processing 
for efficient inference across multiple GPUs.

Output: JSONL with role_id, role_label, question_id, question_label, prompt, response

uv run baseline.py \
	--questions_file /root/git/persona-subspace/evals/data/questions/harmbench.jsonl \
	--roles_file /root/git/persona-subspace/evals/data/roles/good_evil.jsonl \
	--output_jsonl /root/git/persona-subspace/evals/results/roles_traits/harmbench_baseline.jsonl \

uv run baseline.py \
    --prompts_file /root/git/persona-subspace/evals/data/roles_20.jsonl \
    --output_jsonl /root/git/persona-subspace/evals/results/roles_traits/roles_20_baseline.jsonl
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

from utils.inference_utils import load_vllm_model, batch_chat, close_vllm_model


def read_existing_results(output_jsonl: str) -> Set[str]:
    """Read existing results to avoid duplicates."""
    existing = set()
    if not os.path.exists(output_jsonl):
        return existing
        
    try:
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Create unique key from prompt content
                    key = data['prompt']
                    existing.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Could not read existing results: {e}")
        
    return existing


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate baseline (non-steered) responses using vLLM batch processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--questions_file",
        type=str,
        help="Path to questions JSONL file with id and semantic_category fields"
    )
    
    parser.add_argument(
        "--roles_file", 
        type=str,
        help="Path to roles JSONL file with id and type fields"
    )
    
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="Path to combined prompts file (alternative to separate questions/roles files)"
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
        help="Test mode: only process first question with all roles"
    )
    
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="Specific GPU ID to use (if not specified, will auto-detect and use all available GPUs)"
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive arguments
    if args.prompts_file:
        if args.questions_file or args.roles_file:
            parser.error("--prompts_file cannot be used with --questions_file or --roles_file")
    else:
        if not args.questions_file or not args.roles_file:
            parser.error("Either --prompts_file OR both --questions_file and --roles_file must be provided")
    
    return args


def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file with id and semantic_category fields."""
    print(f"Loading questions from {questions_file}")
    
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    questions = []
    with open(questions_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                question_obj = json.loads(line.strip())
                
                # Validate required fields
                if 'id' not in question_obj:
                    print(f"Warning: Missing 'id' field on line {line_num}")
                    continue
                    
                if 'semantic_category' not in question_obj:
                    print(f"Warning: Missing 'semantic_category' field on line {line_num}")
                    continue
                
                # Extract question text (try multiple possible field names)
                question_text = None
                for field in ['question', 'text', 'prompt']:
                    if field in question_obj:
                        question_text = question_obj[field]
                        break
                
                if question_text is None:
                    print(f"Warning: No question text found on line {line_num}")
                    continue
                
                questions.append({
                    'id': question_obj['id'],
                    'semantic_category': question_obj['semantic_category'],
                    'text': question_text
                })
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"Loaded {len(questions)} questions")
    return questions


def load_roles(roles_file: str) -> List[Dict[str, Any]]:
    """Load roles from JSONL file with id and type fields."""
    print(f"Loading roles from {roles_file}")
    
    if not os.path.exists(roles_file):
        raise FileNotFoundError(f"Roles file not found: {roles_file}")
    
    roles = []
    with open(roles_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                role_obj = json.loads(line.strip())
                
                # Validate required fields
                if 'id' not in role_obj:
                    print(f"Warning: Missing 'id' field on line {line_num}")
                    continue
                    
                if 'type' not in role_obj:
                    print(f"Warning: Missing 'type' field on line {line_num}")
                    continue
                
                # Extract role text
                role_text = None
                for field in ['role', 'text', 'prompt']:
                    if field in role_obj:
                        role_text = role_obj[field]
                        break
                
                if role_text is None:
                    print(f"Warning: No role text found on line {line_num}")
                    continue
                
                roles.append({
                    'id': role_obj['id'],
                    'type': role_obj['type'],
                    'text': role_text
                })
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"Loaded {len(roles)} roles")
    return roles


def load_prompts_file(prompts_file: str) -> List[Dict[str, Any]]:
    """Load prompts from combined JSONL file with role, prompt_id, question_id, prompt, question fields."""
    print(f"Loading prompts from {prompts_file}")
    
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    prompts = []
    unique_roles = set()
    
    with open(prompts_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                prompt_obj = json.loads(line.strip())
                
                # Validate required fields
                required_fields = ['role', 'prompt_id', 'question_id', 'prompt', 'question']
                for field in required_fields:
                    if field not in prompt_obj:
                        print(f"Warning: Missing '{field}' field on line {line_num}")
                        continue
                
                # Track unique roles for alphabetical ordering
                unique_roles.add(prompt_obj['role'])
                
                prompts.append(prompt_obj)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    
    # Create alphabetical ordering for role_id
    sorted_roles = sorted(unique_roles)
    role_to_id = {role: idx for idx, role in enumerate(sorted_roles)}
    
    # Process prompts with proper mappings
    processed_prompts = []
    for prompt_obj in prompts:
        role = prompt_obj['role']
        
        processed_prompt = {
            'role_id': role_to_id[role],
            'role_label': 'role' if role != 'default' else 'default',
            'question_id': prompt_obj['question_id'],
            'question_label': role,
            'prompt': f"{prompt_obj['prompt']} {prompt_obj['question']}".strip() if prompt_obj['prompt'] else prompt_obj['question']
        }
        
        processed_prompts.append(processed_prompt)
    
    print(f"Loaded {len(processed_prompts)} prompts from {len(sorted_roles)} unique roles: {sorted_roles}")
    return processed_prompts


def generate_all_prompts(roles: List[Dict[str, Any]], questions: List[Dict[str, Any]], existing_results: Set[str]) -> List[Dict[str, Any]]:
    """Generate all role-question combination prompts, filtering out existing ones."""
    prompts_data = []
    skipped_count = 0
    
    for role in roles:
        for question in questions:
            # Concatenate role and question
            prompt = f"{role['text']} {question['text']}".strip() if role['text'] else question['text']
            
            # Check if this prompt already exists
            if prompt in existing_results:
                skipped_count += 1
                continue
            
            prompts_data.append({
                'role_id': role['id'],
                'role_label': role['type'],
                'question_id': question['id'],
                'question_label': question['semantic_category'],
                'prompt': prompt
            })
    
    if skipped_count > 0:
        print(f"Filtered out {skipped_count} existing combinations")
    
    return prompts_data


def write_results_to_jsonl(prompts_data: List[Dict[str, Any]], responses: List[str], output_jsonl: str, existing_results: Set[str]):
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
            # Check if this prompt already exists
            if prompt_data['prompt'] in existing_results:
                skipped_count += 1
                continue
                
            row_data = {
                'role_id': prompt_data['role_id'],
                'role_label': prompt_data['role_label'],
                'question_id': prompt_data['question_id'],
                'question_label': prompt_data['question_label'],
                'prompt': prompt_data['prompt'],
                'response': response,
                'magnitude': 0.0
            }
            f.write(json.dumps(row_data) + '\n')
            written_count += 1
    
    print(f"Successfully wrote {written_count} new results to {output_jsonl}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing results")


def main():
    """Main function to generate baseline responses."""
    args = parse_arguments()
    
    print("="*60)
    print("Baseline Response Generation")
    print("="*60)
    
    # Check for existing results to enable restart functionality
    existing_results = read_existing_results(args.output_jsonl)
    if existing_results:
        print(f"Found {len(existing_results)} existing results, will skip duplicates")
    
    # Load and validate inputs
    if args.prompts_file:
        # Load combined prompts file
        prompts_data = load_prompts_file(args.prompts_file)
        
        # Apply test mode filtering
        if args.test_mode:
            if len(prompts_data) > 0:
                # In test mode, keep only first unique question_id
                first_question_id = prompts_data[0]['question_id']
                prompts_data = [p for p in prompts_data if p['question_id'] == first_question_id]
                print(f"TEST MODE: Using only question_id {first_question_id} ({len(prompts_data)} prompts)")
            else:
                print("Warning: No prompts available for test mode")
        
        # Filter out existing results
        original_count = len(prompts_data)
        prompts_data = [p for p in prompts_data if p['prompt'] not in existing_results]
        filtered_count = original_count - len(prompts_data)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} existing combinations from combined prompts")
        
        n_unique_roles = len(set(p['role_id'] for p in prompts_data))
        n_unique_questions = len(set(p['question_id'] for p in prompts_data))
        print(f"Using combined prompts format: {len(prompts_data)} prompts from {n_unique_roles} roles and {n_unique_questions} questions")
        
    else:
        # Load separate questions and roles files
        questions = load_questions(args.questions_file)
        roles = load_roles(args.roles_file)
        
        # Apply test mode filtering
        if args.test_mode:
            if len(questions) > 0:
                questions = questions[:1]  # Keep only first question
                print("TEST MODE: Using only the first question")
            else:
                print("Warning: No questions available for test mode")
        
        # Generate all prompts
        print(f"Generating prompts for {len(questions)} questions and {len(roles)} roles")
        prompts_data = generate_all_prompts(roles, questions, existing_results)
        print(f"Generated {len(prompts_data)} total prompts")
    
    # Check if we have any new prompts to process after filtering
    if len(prompts_data) == 0:
        print("No new prompts to process - all combinations already exist!")
        return
        
    # Extract just the prompt strings for batch processing
    prompts = [data['prompt'] for data in prompts_data]
    
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
        # Generate responses using batch processing
        print(f"Generating responses with batch processing...")
        responses = batch_chat(
            model_wrapper=model_wrapper,
            messages=prompts,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            progress=True
        )
        
        # Write results to JSONL
        write_results_to_jsonl(prompts_data, responses, args.output_jsonl, existing_results)
        
        print(f"\nBaseline generation completed!")
        print(f"Results saved to: {args.output_jsonl}")
        
    finally:
        # Clean up model
        close_vllm_model(model_wrapper)


if __name__ == "__main__":
    main()