#!/usr/bin/env python3
"""
Jailbreak Evaluation Script

This script runs evaluations with conversation history as context, ignoring system prompts
and posing questions directly to models after prefilling with conversation turns.

Output: JSONL with evaluation results including conversation context

Example usage:
uv run jailbreak_eval.py \
    --conversation_file /root/git/persona-subspace/dynamics/results/qwen-3-32b/interactive/spiral.json \
    --prefill_turns 5 \
    --questions_file /root/git/persona-subspace/evals/data/harmbench/harmbench.jsonl \
    --output_jsonl /root/git/persona-subspace/dynamics/results/jailbreak_eval.jsonl
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
                    # Create unique key from id, magnitude, and sample_id
                    if samples_per_prompt > 1:
                        # When multiple samples per prompt, use (id, magnitude, sample_id) as key
                        sample_id = data.get('sample_id', 0)
                        key = (data['id'], data.get('magnitude', 0.0), sample_id)
                    else:
                        # Backwards compatibility: just use (id, magnitude) for single samples
                        key = (data['id'], data.get('magnitude', 0.0))
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
    
    
    parser.add_argument(
        "--samples_per_prompt",
        type=int,
        default=1,
        help="Number of samples to generate for each unique id x magnitude combination"
    )

    parser.add_argument(
        "--conversation_file",
        type=str,
        help="Path to conversation JSON file to use as context"
    )

    parser.add_argument(
        "--prefill_turns",
        type=int,
        default=0,
        help="Number of user+assistant turn pairs to use as conversation prefill"
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
                question_text = None
                for field in ['question', 'text', 'prompt']:
                    if field in question_obj:
                        question_text = question_obj[field]
                        break
                
                if question_text is None:
                    print(f"Warning: No question text found on line {line_num}")
                    continue
                
                # Store the original object with the identified text field
                question_obj['_question_text'] = question_text
                questions.append(question_obj)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"Loaded {len(questions)} questions")
    return questions


def load_roles(roles_file: str) -> List[Dict[str, Any]]:
    """Load roles from JSONL file."""
    print(f"Loading roles from {roles_file}")
    
    if not os.path.exists(roles_file):
        raise FileNotFoundError(f"Roles file not found: {roles_file}")
    
    roles = []
    with open(roles_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                role_obj = json.loads(line.strip())
                
                # Only validate presence of role field
                role_text = None
                for field in ['role', 'text', 'prompt']:
                    if field in role_obj:
                        role_text = role_obj[field]
                        break
                
                if role_text is None:
                    print(f"Warning: No role text found on line {line_num}")
                    continue
                
                # Store the original object with the identified text field
                role_obj['_role_text'] = role_text
                roles.append(role_obj)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"Loaded {len(roles)} roles")
    return roles


def load_prompts_file(prompts_file: str) -> List[Dict[str, Any]]:
    """Load prompts from combined JSONL file."""
    print(f"Loading prompts from {prompts_file}")

    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    prompts = []

    with open(prompts_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                prompt_obj = json.loads(line.strip())

                # Only validate presence of prompt and question fields
                if 'prompt' not in prompt_obj or 'question' not in prompt_obj:
                    print(f"Warning: Missing 'prompt' or 'question' field on line {line_num}")
                    continue

                # Create combined prompt text
                combined_prompt = f"{prompt_obj['prompt']} {prompt_obj['question']}".strip() if prompt_obj['prompt'] else prompt_obj['question']
                prompt_obj['_combined_prompt'] = combined_prompt

                prompts.append(prompt_obj)

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")

    print(f"Loaded {len(prompts)} prompts")
    return prompts


def load_conversation_file(conversation_file: str) -> List[Dict[str, str]]:
    """Load conversation history from JSON file."""
    print(f"Loading conversation from {conversation_file}")

    if not os.path.exists(conversation_file):
        raise FileNotFoundError(f"Conversation file not found: {conversation_file}")

    with open(conversation_file, 'r') as f:
        conversation_data = json.load(f)

    if 'conversation' not in conversation_data:
        raise ValueError("Conversation file must contain 'conversation' field")

    conversation = conversation_data['conversation']
    print(f"Loaded conversation with {len(conversation)} messages")
    return conversation


def extract_conversation_turns(conversation: List[Dict[str, str]], prefill_turns: int) -> List[Dict[str, str]]:
    """Extract conversation turns as user+assistant pairs."""
    if prefill_turns <= 0:
        return []

    turns = []
    i = 0
    turn_count = 0

    while i < len(conversation) and turn_count < prefill_turns:
        # Look for user message
        if i < len(conversation) and conversation[i]['role'] == 'user':
            user_message = conversation[i]
            i += 1

            # Look for assistant response
            if i < len(conversation) and conversation[i]['role'] == 'assistant':
                assistant_message = conversation[i]
                i += 1

                # Add both messages as a turn
                turns.append(user_message)
                turns.append(assistant_message)
                turn_count += 1
            else:
                # User message without assistant response - still count as partial turn
                turns.append(user_message)
                break
        else:
            # Skip non-user messages at the start
            i += 1

    print(f"Extracted {turn_count} conversation turns ({len(turns)} messages)")
    return turns


def generate_all_prompts(roles: List[Dict[str, Any]], questions: List[Dict[str, Any]], existing_results: Set[str], company_name: str = "Acme Corp", name_value: str = "Alex", no_system_prompt: bool = False, samples_per_prompt: int = 1) -> List[Dict[str, Any]]:
    """Generate all role-question combination prompts, filtering out existing ones."""
    prompts_data = []
    skipped_count = 0
    
    for role in roles:
        for question in questions:
            # Format company name and name in role and question text
            role_text = role['_role_text'].format(company=company_name, name=name_value)
            question_text = question['_question_text'].format(company=company_name, name=name_value)
            
            if no_system_prompt:
                # In no-system-prompt mode, only use the question text
                prompt = question_text
                system_prompt = ''
                user_message = question_text
            else:
                # Normal mode: combine role and question
                prompt = f"{role_text} {question_text}".strip() if role_text else question_text
                system_prompt = role_text
                user_message = question_text
            
            # Generate multiple samples for this prompt
            for sample_id in range(samples_per_prompt):
                # Create unique key for this sample
                if samples_per_prompt > 1:
                    unique_key = (role.get('id', 0), 0.0, sample_id)
                else:
                    unique_key = (role.get('id', 0), 0.0)
                
                # Check if this specific sample already exists
                if unique_key in existing_results:
                    skipped_count += 1
                    continue
                
                # Combine all fields from both role and question
                prompt_data = {}
                prompt_data.update(role)
                prompt_data.update(question)
                prompt_data['prompt'] = prompt  # Formatted version for output
                prompt_data['_system_prompt'] = system_prompt
                prompt_data['_user_message'] = user_message
                
                # Add sample_id if multiple samples per prompt
                if samples_per_prompt > 1:
                    prompt_data['sample_id'] = sample_id
                
                # Clean up temporary fields
                if '_role_text' in prompt_data:
                    del prompt_data['_role_text']
                if '_question_text' in prompt_data:
                    del prompt_data['_question_text']
                
                prompts_data.append(prompt_data)
    
    if skipped_count > 0:
        print(f"Filtered out {skipped_count} existing combinations")
    
    return prompts_data


def format_messages_for_chat(prompts_data: List[Dict[str, Any]], model_name: str, conversation_prefill: List[Dict[str, str]] = None, no_system_prompt: bool = True) -> List[List[Dict[str, str]]]:
    """Format prompts into chat messages for vLLM batch processing."""
    # Force no system prompt mode for jailbreak evaluation
    formatted_messages = []

    for prompt_data in prompts_data:
        user_message = prompt_data.get('_user_message', '')

        # Start with conversation prefill if provided
        messages = []
        if conversation_prefill:
            messages.extend(conversation_prefill)

        # Add the current question as a user message
        messages.append({"role": "user", "content": user_message})

        formatted_messages.append(messages)

    return formatted_messages


def write_results_to_jsonl(prompts_data: List[Dict[str, Any]], responses: List[str], output_jsonl: str, existing_results: Set[str], samples_per_prompt: int = 1, prefill_turns: int = 0):
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
                unique_key = (prompt_data['id'], prompt_data.get('magnitude', 0.0), sample_id)
            else:
                unique_key = (prompt_data['id'], prompt_data.get('magnitude', 0.0))
            
            # Check if this specific sample already exists
            if unique_key in existing_results:
                skipped_count += 1
                continue
                
            row_data = prompt_data.copy()  # Copy all original fields
            row_data['response'] = response
            row_data['magnitude'] = 0.0
            row_data['turn'] = prefill_turns
            
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
    """Main function to generate jailbreak evaluation responses."""
    args = parse_arguments()

    print("="*60)
    print("Jailbreak Evaluation with Conversation Context")
    print("="*60)
    
    # Check for existing results to enable restart functionality
    existing_results = read_existing_results(args.output_jsonl, args.samples_per_prompt)
    if existing_results:
        print(f"Found {len(existing_results)} existing results, will skip duplicates")

    # Load conversation context if provided
    conversation_prefill = []
    if args.conversation_file and args.prefill_turns > 0:
        conversation = load_conversation_file(args.conversation_file)
        conversation_prefill = extract_conversation_turns(conversation, args.prefill_turns)
    elif args.conversation_file:
        print("Warning: conversation_file provided but prefill_turns is 0 - no conversation context will be used")

    # Force no system prompt mode for jailbreak evaluation
    print("Forcing no-system-prompt mode for direct question evaluation")
    
    # Load and validate inputs
    if args.prompts_file:
        # Load combined prompts file
        prompts_data = load_prompts_file(args.prompts_file)
        
        # Apply test mode filtering
        if args.test_mode:
            if len(prompts_data) > 0:
                # In test mode, keep only first 10 prompts
                prompts_data = prompts_data[:10]
                print(f"TEST MODE: Using only first 10 prompts ({len(prompts_data)} prompts)")
            else:
                print("Warning: No prompts available for test mode")
        
        # Generate multiple samples for each unique prompt if needed
        original_prompts = prompts_data.copy()
        prompts_data = []
        skipped_count = 0
        
        for p in original_prompts:
            # Get raw prompt texts without formatting
            system_prompt_text = p.get('prompt', '')
            user_message = p.get('question', '')
            
            # Force no-system-prompt mode: only use the question text
            formatted_combined = user_message
            system_prompt = ''
            
            # Generate multiple samples for this prompt
            for sample_id in range(args.samples_per_prompt):
                # Create unique key for this sample
                if args.samples_per_prompt > 1:
                    unique_key = (p['id'], 0.0, sample_id)
                else:
                    unique_key = (p['id'], 0.0)
                
                # Check if this specific sample already exists
                if unique_key in existing_results:
                    skipped_count += 1
                    continue
                
                # Create new prompt data for this sample
                prompt_data = p.copy()
                prompt_data['prompt'] = formatted_combined  # Formatted version for output
                prompt_data['_system_prompt'] = system_prompt
                prompt_data['_user_message'] = user_message
                
                # Add sample_id if multiple samples per prompt
                if args.samples_per_prompt > 1:
                    prompt_data['sample_id'] = sample_id
                
                # Clean up temporary fields
                if '_combined_prompt' in prompt_data:
                    del prompt_data['_combined_prompt']
                
                prompts_data.append(prompt_data)
        
        if skipped_count > 0:
            print(f"Filtered out {skipped_count} existing combinations from combined prompts")
        
        n_unique_roles = len(set(p.get('role', '') for p in prompts_data))
        n_unique_questions = len(set(p.get('id', p.get('question_id', '')) for p in prompts_data))
        print(f"Using combined prompts format: {len(prompts_data)} prompts from {n_unique_roles} roles and {n_unique_questions} questions")
        
    else:
        # Load separate questions and roles files
        questions = load_questions(args.questions_file)
        roles = load_roles(args.roles_file)
        
        # Apply test mode filtering
        if args.test_mode:
            if len(questions) > 0:
                questions = questions[:10]  # Keep only first 10 questions
                print(f"TEST MODE: Using only first 10 questions ({len(questions)} questions)")
            else:
                print("Warning: No questions available for test mode")
        
        # Generate all prompts (force no system prompt)
        print(f"Generating prompts for {len(questions)} questions and {len(roles)} roles")
        prompts_data = generate_all_prompts(roles, questions, existing_results, "Acme Corp", "Alex", True, args.samples_per_prompt)
        print(f"Generated {len(prompts_data)} total prompts")
    
    # Check if we have any new prompts to process after filtering
    if len(prompts_data) == 0:
        print("No new prompts to process - all combinations already exist!")
        return
        
    # Format messages for chat-based processing with conversation prefill
    messages_list = format_messages_for_chat(prompts_data, args.model_name, conversation_prefill, True)
    
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
        write_results_to_jsonl(prompts_data, responses, args.output_jsonl, existing_results, args.samples_per_prompt, args.prefill_turns)
        
        print(f"\nJailbreak evaluation completed!")
        print(f"Results saved to: {args.output_jsonl}")
        if conversation_prefill:
            print(f"Used {len(conversation_prefill)} messages as conversation context")
        
    finally:
        # Clean up model
        close_vllm_model(model_wrapper)


if __name__ == "__main__":
    main()