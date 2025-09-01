#!/usr/bin/env python3
"""
Multi-GPU PCA Component Steering Script

This script performs activation steering on multiple PCA components in parallel
using multiple GPUs. Each GPU processes different PC components to maximize
hardware utilization while working within ActivationSteerer's single-GPU constraint.
"""

import argparse
import json
import os
import sys
import multiprocessing as mp
import threading
import fcntl
import time
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import torch

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))
torch.set_float32_matmul_precision('high')

from utils.steering_utils import ActivationSteering
from utils.probing_utils import load_model, generate_text


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU PCA component steering script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--pca_filepath",
        type=str,
        required=True,
        help="Path to PCA results file (.pt format)"
    )
    
    # Create mutually exclusive group for questions input
    questions_group = parser.add_mutually_exclusive_group(required=True)
    questions_group.add_argument(
        "--questions_filepath", 
        type=str,
        help="Path to questions JSONL file"
    )
    questions_group.add_argument(
        "--questions_dir",
        type=str,
        help="Path to directory containing JSONL question files"
    )
    
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        required=True,
        help="List of PC component indices to process (0-indexed)"
    )
    
    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[-1000.0, 0.0, 1000.0],
        help="List of steering magnitudes"
    )
    
    parser.add_argument(
        "--layer",
        type=int,
        default=22,
        help="Layer index for steering"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-27b-it",
        help="Model name for steering"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/git/persona-subspace/steering/results/roles_240",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--test_questions",
        type=int,
        default=0,
        help="Number of questions to use for testing (0 = all questions)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing questions"
    )
    
    parser.add_argument(
        "--question_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of question indices to use (0-indexed, inclusive start, exclusive end)"
    )
    
    
    return parser.parse_args()


def load_pca_results(pca_filepath: str) -> Dict[str, Any]:
    """Load and validate PCA results file."""
    print(f"Loading PCA results from {pca_filepath}")
    
    if not os.path.exists(pca_filepath):
        raise FileNotFoundError(f"PCA results file not found: {pca_filepath}")
    
    try:
        pca_results = torch.load(pca_filepath, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load PCA results: {e}")
    
    # Validate PCA structure
    if 'pca' not in pca_results:
        raise ValueError("PCA results must contain 'pca' key")
    
    if not hasattr(pca_results['pca'], 'components_'):
        raise ValueError("PCA object must have 'components_' attribute")
    
    n_components = pca_results['pca'].components_.shape[0]
    print(f"Found PCA with {n_components} components")
    
    return pca_results


def load_questions_from_directory(questions_dir: str, test_questions: int = 0, question_range: List[int] = None) -> List[Dict]:
    """Load questions from all JSONL files in a directory with deduplication."""
    print(f"Loading questions from directory {questions_dir}")
    
    if not os.path.exists(questions_dir):
        raise FileNotFoundError(f"Questions directory not found: {questions_dir}")
    
    if not os.path.isdir(questions_dir):
        raise ValueError(f"Path is not a directory: {questions_dir}")
    
    # Find all JSONL files in the directory
    jsonl_files = [f for f in os.listdir(questions_dir) if f.endswith('.jsonl')]
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in directory: {questions_dir}")
    
    print(f"Found {len(jsonl_files)} JSONL files: {jsonl_files}")
    
    # Load questions from all files with deduplication
    seen_pairs = set()
    questions = []
    duplicates_removed = 0
    
    for filename in sorted(jsonl_files):  # Sort for consistent ordering
        filepath = os.path.join(questions_dir, filename)
        print(f"  Loading {filename}...")
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    question_obj = json.loads(line.strip())
                    
                    # Extract prompt and question fields, with fallbacks
                    prompt = question_obj.get('prompt', '')
                    question = question_obj.get('question', '')
                    
                    if not question:
                        print(f"    Warning: No question field found on line {line_num}")
                        continue
                    
                    # Create tuple for deduplication based on both fields
                    dedup_key = (prompt, question)
                    
                    if dedup_key not in seen_pairs:
                        seen_pairs.add(dedup_key)
                        
                        # Create combined field
                        combined = f"{prompt} {question}" if prompt else question
                        
                        # Create structured entry
                        entry = {
                            'prompt': prompt,
                            'question': question,
                            'combined': combined
                        }
                        
                        # Preserve any other fields from original data
                        for key, value in question_obj.items():
                            if key not in ['prompt', 'question']:
                                entry[key] = value
                        
                        questions.append(entry)
                    else:
                        duplicates_removed += 1
                        
                except json.JSONDecodeError as e:
                    print(f"    Warning: Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"Loaded {len(questions)} unique questions")
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate questions")
    
    # Apply filtering logic
    if question_range is not None:
        start_idx, end_idx = question_range
        if start_idx < 0 or end_idx > len(questions) or start_idx >= end_idx:
            raise ValueError(f"Invalid question range [{start_idx}, {end_idx}) for {len(questions)} questions")
        questions = questions[start_idx:end_idx]
        print(f"Using questions {start_idx} to {end_idx-1} ({len(questions)} questions)")
    elif test_questions > 0:
        questions = questions[:test_questions]
        print(f"Using first {len(questions)} questions for testing")
    else:
        print(f"Using all {len(questions)} questions")
    
    return questions


def load_questions(questions_filepath: str, test_questions: int = 0, question_range: List[int] = None) -> List[Dict]:
    """Load questions from JSONL file, preserving role and question fields."""
    print(f"Loading questions from {questions_filepath}")
    
    if not os.path.exists(questions_filepath):
        raise FileNotFoundError(f"Questions file not found: {questions_filepath}")
    
    questions = []
    seen_pairs = set()
    duplicates_removed = 0
    
    with open(questions_filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                question_obj = json.loads(line.strip())
                
                # Extract prompt and question fields, with fallbacks
                prompt = question_obj.get('prompt', '')
                question = question_obj.get('question', '')
                
                if not question:
                    print(f"Warning: No question field found on line {line_num}")
                    continue
                
                # Create tuple for deduplication based on both fields
                dedup_key = (prompt, question)
                
                if dedup_key not in seen_pairs:
                    seen_pairs.add(dedup_key)
                    
                    # Create structured entry
                    entry = {
                        'prompt': prompt,
                        'question': question,
                        'combined': f"{prompt} {question}".strip() if prompt else question
                    }
                    
                    # Preserve any other fields from original data
                    for key, value in question_obj.items():
                        if key not in ['prompt', 'question']:
                            entry[key] = value
                    
                    questions.append(entry)
                else:
                    duplicates_removed += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    
    print(f"Loaded {len(questions)} unique questions")
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate questions")
    
    if question_range is not None:
        start_idx, end_idx = question_range
        if start_idx < 0 or end_idx > len(questions) or start_idx >= end_idx:
            raise ValueError(f"Invalid question range [{start_idx}, {end_idx}) for {len(questions)} questions")
        questions = questions[start_idx:end_idx]
        print(f"Using questions {start_idx} to {end_idx-1} ({len(questions)} questions)")
    elif test_questions > 0:
        questions = questions[:test_questions]
        print(f"Using first {len(questions)} questions for testing")
    else:
        print(f"Loaded {len(questions)} questions")
    
    return questions



def generate_batched_responses(
    model, 
    tokenizer, 
    questions, 
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    max_length: int = 2048
) -> List[str]:
    """
    Generate responses for a batch of questions efficiently using real batch inference.
    Questions can be strings or dicts with 'role', 'question', 'combined' keys.
    """
    try:
        if not questions:
            return []
        
        # Determine if this is a Gemma model (no system prompt support)
        is_gemma = 'gemma' in model.config.name_or_path.lower() if hasattr(model.config, 'name_or_path') else False
        
        # Format prompts for chat
        formatted_prompts = []
        for question_item in questions:
            # Handle both string and dict formats
            if isinstance(question_item, str):
                prompt = ''
                question = question_item
                combined = question_item
            else:
                prompt = question_item.get('prompt', '')
                question = question_item.get('question', '')
                combined = question_item.get('combined', question)
            
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                if is_gemma or not prompt:
                    # Gemma model or no prompt: use concatenated format in user message
                    messages = [{"role": "user", "content": combined}]
                else:
                    # Other models with prompt: use system prompt + user message
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question}
                    ]
                
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                formatted_prompts.append(formatted_prompt)
            else:
                # Fallback: use combined format
                formatted_prompts.append(combined)
        
        # Set padding side to left for decoder-only models
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        
        # Tokenize all prompts at once
        batch_inputs = tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(model.device)
        
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        
        # Set up generation parameters
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': True if temperature > 0 else False,
            'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'use_cache': True
        }
        
        # Generate responses for entire batch at once
        with torch.no_grad():
            batch_outputs = model.generate(
                input_ids=batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                **generation_config
            )
        
        # Decode responses
        batch_responses = []
        input_lengths = batch_inputs.input_ids.shape[1]
        
        for i, output in enumerate(batch_outputs):
            # Extract only the generated part (after input)
            generated_tokens = output[input_lengths:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            batch_responses.append(response.strip())
        
        return batch_responses
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Fallback to sequential processing
        print("Falling back to sequential processing")
        try:
            batch_responses = []
            for question in questions:
                response = generate_text(
                    model, tokenizer, question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    chat_format=True
                )
                batch_responses.append(response)
            return batch_responses
        except Exception as fallback_e:
            print(f"Fallback processing also failed: {fallback_e}")
            return [""] * len(questions)


class ThreadSafeJSONWriter:
    """Thread-safe JSON file writer with file locking."""
    
    def __init__(self):
        self._locks = {}
        self._global_lock = threading.Lock()
    
    def _get_file_lock(self, filepath: str) -> threading.Lock:
        """Get or create a lock for a specific file."""
        with self._global_lock:
            if filepath not in self._locks:
                self._locks[filepath] = threading.Lock()
            return self._locks[filepath]
    
    def write_json(self, filepath: str, data: dict, max_retries: int = 3) -> bool:
        """Write JSON data to file with thread safety and file locking."""
        file_lock = self._get_file_lock(filepath)
        
        for attempt in range(max_retries):
            try:
                with file_lock:
                    with open(filepath, 'w') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                        json.dump(data, f, indent=2)
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to write JSON after {max_retries} attempts: {e}")
                    return False
                time.sleep(0.1)  # Brief delay before retry
        return False
    
    def load_json(self, filepath: str) -> dict:
        """Load JSON data from file with thread safety."""
        if not os.path.exists(filepath):
            return {}
        
        file_lock = self._get_file_lock(filepath)
        
        try:
            with file_lock:
                with open(filepath, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load JSON from {filepath}: {e}")
            return {}
    
    def atomic_update_json(self, filepath: str, update_func, max_retries: int = 3) -> bool:
        """Atomically read, modify, and write JSON data with exclusive locking."""
        file_lock = self._get_file_lock(filepath)
        
        for attempt in range(max_retries):
            try:
                with file_lock:
                    # Read current state
                    current_data = {}
                    if os.path.exists(filepath):
                        try:
                            with open(filepath, 'r') as f:
                                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for read too
                                current_data = json.load(f)
                        except Exception as e:
                            print(f"Warning: Could not read existing data: {e}")
                            current_data = {}
                    
                    # Apply update function
                    updated_data = update_func(current_data)
                    
                    # Write updated data
                    with open(filepath, 'w') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                        json.dump(updated_data, f, indent=2)
                
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed atomic update after {max_retries} attempts: {e}")
                    return False
                time.sleep(0.1)  # Brief delay before retry
        return False


# Global JSON writer instance
json_writer = ThreadSafeJSONWriter()


def create_work_units(components: List[int], questions: List[str], magnitudes: List[float], batch_size: int):
    """Create work units for queue-based processing."""
    work_units = []
    
    # Group questions into batches
    question_batches = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        question_batches.append(batch)
    
    # Create work units as (component, magnitude, question_batch) combinations
    for component in components:
        for magnitude in magnitudes:
            for question_batch in question_batches:
                work_units.append({
                    'component': component,
                    'magnitude': magnitude,
                    'questions': question_batch
                })
    
    print(f"Created {len(work_units)} work units from {len(components)} components, {len(magnitudes)} magnitudes, {len(question_batches)} question batches")
    return work_units


def worker_process(
    gpu_id: int,
    work_queue: mp.Queue,
    pca_filepath: str,
    layer: int,
    model_name: str,
    output_dir: str,
    total_work_units: int
):
    """
    Worker process that pulls work units from queue and processes them on a single GPU.
    
    Args:
        gpu_id: CUDA device ID to use
        work_queue: Queue containing work units to process
        pca_filepath: Path to PCA results file
        layer: Layer index for steering
        model_name: Model name to load
        output_dir: Output directory for results
        total_work_units: Total number of work units for progress tracking
    """
    try:
        print(f"Worker GPU {gpu_id}: Starting work queue consumer")
        
        # Load model on assigned GPU
        device = f"cuda:{gpu_id}"
        model, tokenizer = load_model(model_name, device=device)
        print(f"Worker GPU {gpu_id}: Model loaded on {device}")
        
        # Load PCA results
        pca_results = torch.load(pca_filepath, weights_only=False)
        
        processed_work_units = 0
        
        # Process work units from queue
        while True:
            try:
                # Get work unit from queue with timeout
                try:
                    work_unit = work_queue.get(timeout=10)  # 10 second timeout
                except:
                    # Queue is empty or timeout reached
                    print(f"Worker GPU {gpu_id}: No more work available, exiting")
                    break
                
                if work_unit is None:  # Sentinel value to indicate end
                    print(f"Worker GPU {gpu_id}: Received end signal, exiting")
                    work_queue.put(None)  # Re-add sentinel for other workers
                    break
                
                pc_idx = work_unit['component']
                magnitude = work_unit['magnitude']
                questions_batch = work_unit['questions']
                
                processed_work_units += 1
                print(f"Worker GPU {gpu_id}: Processing PC{pc_idx+1}, magnitude {magnitude}, {len(questions_batch)} questions ({processed_work_units}/{total_work_units})")
                
                output_file = os.path.join(output_dir, f"pc{pc_idx+1}.json")
                
                # Get steering vector and ensure correct device/dtype
                steering_vector = torch.from_numpy(pca_results['pca'].components_[pc_idx])
                steering_vector = steering_vector.to(device=device, dtype=model.dtype)
                
                # Generate responses first (outside of file lock)
                try:
                    with ActivationSteering(
                        model=model,
                        steering_vectors=[steering_vector],
                        coefficients=magnitude,
                        layer_indices=layer,
                        intervention_type="addition",
                        positions="all"
                    ) as steerer:
                        # Generate responses in batch for all questions
                        batch_responses = generate_batched_responses(
                            model, tokenizer, questions_batch
                        )
                        
                except Exception as e:
                    print(f"Worker GPU {gpu_id}: Error generating responses: {e}")
                    continue
                
                # Atomically update file with new responses
                def update_file_data(current_data):
                    """Update function for atomic file operation."""
                    # Filter questions that haven't been processed yet for this magnitude
                    questions_to_add = []
                    responses_to_add = []
                    question_items_to_add = []
                    
                    for question_item, response in zip(questions_batch, batch_responses):
                        # Create deduplication key from prompt and question fields
                        if isinstance(question_item, dict):
                            prompt = question_item.get('prompt', '')
                            question = question_item.get('question', '')
                            question_key = f"{prompt} {question}".strip()
                        else:
                            prompt = ''
                            question = question_item
                            question_key = question
                        
                        if question_key not in current_data:
                            current_data[question_key] = {
                                'prompt': question_item.get('prompt', '') if isinstance(question_item, dict) else '',
                                'question': question_item.get('question', question_item) if isinstance(question_item, dict) else question_item,
                                'magnitudes': {}
                            }
                        
                        if magnitude not in current_data[question_key]['magnitudes']:
                            current_data[question_key]['magnitudes'][magnitude] = []
                            questions_to_add.append(question_key)
                            responses_to_add.append(response)
                            question_items_to_add.append(question_item)
                        # If already processed, skip this question
                    
                    # Add new responses
                    for question_key, response in zip(questions_to_add, responses_to_add):
                        current_data[question_key]['magnitudes'][magnitude].append(response)
                    
                    if questions_to_add:
                        print(f"Worker GPU {gpu_id}: Added {len(questions_to_add)} new responses for PC{pc_idx+1}, magnitude {magnitude}")
                    else:
                        print(f"Worker GPU {gpu_id}: All questions already processed for PC{pc_idx+1}, magnitude {magnitude}")
                    
                    return current_data
                
                # Perform atomic update
                if not json_writer.atomic_update_json(output_file, update_file_data):
                    print(f"Worker GPU {gpu_id}: Error updating results file {output_file}")
                    
            except Exception as e:
                print(f"Worker GPU {gpu_id}: Error processing work unit: {e}")
                continue
        
        print(f"Worker GPU {gpu_id}: Completed {processed_work_units} work units")
        
    except Exception as e:
        print(f"Worker GPU {gpu_id}: Fatal error: {e}")
        raise


def main():
    """Main function to orchestrate multi-GPU steering."""
    args = parse_arguments()
    
    print("="*50)
    print("Multi-GPU PCA Component Steering")
    print("="*50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and validate inputs
    pca_results = load_pca_results(args.pca_filepath)
    
    # Load questions from either file or directory
    if args.questions_filepath:
        questions = load_questions(args.questions_filepath, args.test_questions, args.question_range)
    else:
        questions = load_questions_from_directory(args.questions_dir, args.test_questions, args.question_range)
    
    # Questions are now loaded with role and question fields already structured
    print(f"Using {len(questions)} questions")
    
    # Validate component indices
    n_components = pca_results['pca'].components_.shape[0]
    for comp in args.components:
        if comp < 0 or comp >= n_components:
            raise ValueError(f"Component index {comp} out of range [0, {n_components-1}]")
    
    # Create work units
    work_units = create_work_units(args.components, questions, args.magnitudes, args.batch_size)
    
    if not work_units:
        print("No work units to process")
        return
    
    print(f"Total work units: {len(work_units)}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs available")
    
    # Create work queue
    work_queue = mp.Queue()
    
    # Populate work queue
    for work_unit in work_units:
        work_queue.put(work_unit)
    
    # Add sentinel values for workers
    for _ in range(n_gpus):
        work_queue.put(None)
    
    # Launch worker processes
    processes = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                work_queue,
                args.pca_filepath,
                args.layer,
                args.model_name,
                args.output_dir,
                len(work_units)
            )
        )
        p.start()
        processes.append(p)
    
    print(f"\nLaunched {len(processes)} worker processes for {len(work_units)} work units")
    
    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            print(f"Warning: Process {i} exited with code {p.exitcode}")
        else:
            print(f"Process {i} completed successfully")
    
    print("\nAll workers completed!")
    print(f"Results saved to: {args.output_dir}")
    
    # Print summary of component files created
    component_files = []
    for comp in args.components:
        output_file = os.path.join(args.output_dir, f"pc{comp+1}.json")
        if os.path.exists(output_file):
            component_files.append(output_file)
    
    if component_files:
        print(f"Component files created: {len(component_files)}")
        for f in component_files:
            print(f"  {f}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()