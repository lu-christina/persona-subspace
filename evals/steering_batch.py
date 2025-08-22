#!/usr/bin/env python3
"""
Batch-Based GPU Parallelized Steering Script

This script performs activation steering using batched processing for better GPU utilization.
It distributes work units (role, question, magnitude combinations) across multiple GPUs,
with each GPU processing multiple prompts simultaneously in batches.

Based on the pattern from 3_response_activations.py for optimal GPU utilization.

Output: JSONL with role_id, role_label, question_id, question_label, prompt, response, magnitude
"""

import argparse
import json
import os
import sys
import multiprocessing as mp
import csv
import threading
import gc
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fcntl
import time
from collections import defaultdict
from tqdm import tqdm

import torch

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))
torch.set_float32_matmul_precision('high')

from utils.steering_utils import ActivationSteering
from utils.probing_utils import load_model, generate_text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JSONLHandler:
    """Thread-safe JSONL handler with file locking for concurrent access."""
    
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self._lock = threading.Lock()
    
    def read_existing_combinations(self) -> set:
        """Read existing combinations to avoid duplicates."""
        existing = set()
        if not os.path.exists(self.jsonl_path):
            return existing
            
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Create unique key from prompt text and magnitude
                        key = (data['prompt'], float(data['magnitude']))
                        existing.add(key)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            print(f"Warning: Could not read existing JSONL: {e}")
            
        return existing
    
    def write_rows(self, row_data_list: List[Dict[str, Any]]) -> bool:
        """Write multiple rows to JSONL with file locking."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                    for row_data in row_data_list:
                        f.write(json.dumps(row_data) + '\n')
                    return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to write rows after {max_retries} attempts: {e}")
                    return False
                time.sleep(0.1)  # Brief delay before retry
        return False
    
    def write_row(self, row_data: Dict[str, Any]) -> bool:
        """Write a single row to JSONL with file locking."""
        return self.write_rows([row_data])


def generate_batched_responses(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    batch_size: int = 16,
    max_length: int = 2048
) -> List[str]:
    """
    Generate responses for a batch of prompts efficiently.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum new tokens to generate
        temperature: Temperature for generation
        batch_size: Maximum batch size for processing
        max_length: Maximum total sequence length
    
    Returns:
        List of generated responses, one per prompt
    """
    all_responses = []
    num_prompts = len(prompts)
    
    # Process prompts in batches
    for batch_start in range(0, num_prompts, batch_size):
        batch_end = min(batch_start + batch_size, num_prompts)
        batch_prompts = prompts[batch_start:batch_end]
        
        try:
            # For now, process each prompt individually within the batch
            # This maintains compatibility with the existing generate_text function
            batch_responses = []
            for prompt in batch_prompts:
                response = generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    chat_format=True
                )
                batch_responses.append(response)
            
            all_responses.extend(batch_responses)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
            # Add empty responses for failed batch
            all_responses.extend([""] * len(batch_prompts))
    
    return all_responses


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch-based GPU parallelized steering script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--pca_filepath",
        type=str,
        required=True,
        help="Path to PCA results file (.pt format)"
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
        "--component",
        type=int,
        default=0,
        help="PC component index to process (0-indexed)"
    )
    
    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[2000.0, 4000.0],
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
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing prompts"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Test mode: only process first question with all roles"
    )
    
    parser.add_argument(
        "--gpu_id",
        type=int,
        help="Specific GPU ID to use (0-indexed). If not specified, all available GPUs will be used."
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
            'combined_prompt': f"{prompt_obj['prompt']} {prompt_obj['question']}"
        }
        
        processed_prompts.append(processed_prompt)
    
    print(f"Loaded {len(processed_prompts)} prompts from {len(sorted_roles)} unique roles: {sorted_roles}")
    return processed_prompts


def create_work_units(prompts_data, magnitudes, is_combined_format=False):
    """Create work units from prompts and magnitudes."""
    work_units = []
    
    if is_combined_format:
        # prompts_data contains pre-processed combined prompts
        for prompt_data in prompts_data:
            for magnitude in magnitudes:
                work_unit = {
                    'role_id': prompt_data['role_id'],
                    'role_label': prompt_data['role_label'],
                    'question_id': prompt_data['question_id'],
                    'question_label': prompt_data['question_label'],
                    'prompt': prompt_data['combined_prompt'],
                    'magnitude': magnitude
                }
                work_units.append(work_unit)
    else:
        # prompts_data contains [questions, roles]
        questions, roles = prompts_data
        for role in roles:
            for question in questions:
                for magnitude in magnitudes:
                    work_unit = {
                        'role_id': role['id'],
                        'role_label': role['type'],
                        'question_id': question['id'],
                        'question_label': question['semantic_category'],
                        'prompt': f"{role['text']} {question['text']}",
                        'magnitude': magnitude
                    }
                    work_units.append(work_unit)
    
    return work_units


def distribute_work_units(work_units: List[Dict], n_gpus: int) -> List[List[Dict]]:
    """Distribute work units across GPUs as evenly as possible."""
    if n_gpus <= 0:
        raise ValueError("Number of GPUs must be positive")
    
    if len(work_units) == 0:
        return [[] for _ in range(n_gpus)]
    
    # Calculate base number of work units per GPU
    base_count = len(work_units) // n_gpus
    remainder = len(work_units) % n_gpus
    
    assignments = []
    start_idx = 0
    
    for gpu_id in range(n_gpus):
        # Some GPUs get one extra work unit if there's a remainder
        count = base_count + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + count
        
        assignments.append(work_units[start_idx:end_idx])
        start_idx = end_idx
    
    return assignments


def worker_process(
    gpu_id: int,
    assigned_work_units: List[Dict],
    pca_filepath: str,
    component: int,
    layer: int,
    model_name: str,
    output_jsonl: str,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    max_length: int
):
    """
    Worker process that handles steering for assigned work units on a single GPU using batching.
    """
    torch.set_float32_matmul_precision('high')
    
    try:
        logger = logging.getLogger(f"GPU-{gpu_id}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - GPU-{gpu_id} - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info(f"Starting with {len(assigned_work_units)} work units")
        
        # Load model on assigned GPU
        device = f"cuda:{gpu_id}"
        model, tokenizer = load_model(model_name, device=device)
        logger.info(f"Model loaded on {device}")
        
        # Load PCA results
        pca_results = torch.load(pca_filepath, weights_only=False)
        
        # Get steering vector for the specified component
        steering_vector = torch.from_numpy(pca_results['pca'].components_[component])
        steering_vector = steering_vector.to(device=device, dtype=model.dtype)
        
        logger.info(f"Using PC{component+1}, steering vector shape: {steering_vector.shape}")
        
        # Initialize JSONL handler
        jsonl_handler = JSONLHandler(output_jsonl)
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        
        # Get existing combinations to avoid duplicates
        existing_combinations = jsonl_handler.read_existing_combinations()
        
        # Filter work units to exclude existing combinations
        work_units_to_process = []
        for work_unit in assigned_work_units:
            combination_key = (work_unit['prompt'], work_unit['magnitude'])
            if combination_key not in existing_combinations:
                work_units_to_process.append(work_unit)
        
        logger.info(f"Processing {len(work_units_to_process)} new work units out of {len(assigned_work_units)} total")
        logger.info(f"Skipping {len(assigned_work_units) - len(work_units_to_process)} already completed work units")
        
        if not work_units_to_process:
            logger.info("No new work units to process")
            return
        
        # Group work units by magnitude for efficient processing
        work_by_magnitude = defaultdict(list)
        for work_unit in work_units_to_process:
            work_by_magnitude[work_unit['magnitude']].append(work_unit)
        
        completed_work_units = 0
        total_work_units = len(work_units_to_process)
        
        # Process each magnitude group
        with tqdm(work_by_magnitude.items(), desc=f"GPU-{gpu_id} magnitudes", unit="mag", position=gpu_id) as mag_pbar:
            for magnitude, magnitude_work_units in mag_pbar:
                mag_pbar.set_postfix(magnitude=f"{magnitude:.0f}", work_units=len(magnitude_work_units))
                
                try:
                    with ActivationSteering(
                        model=model,
                        steering_vectors=[steering_vector],
                        coefficients=magnitude,
                        layer_indices=layer,
                        intervention_type="addition",
                        positions="all"
                    ) as steerer:
                        
                        # Process work units in batches
                        for batch_start in range(0, len(magnitude_work_units), batch_size):
                            batch_end = min(batch_start + batch_size, len(magnitude_work_units))
                            batch_work_units = magnitude_work_units[batch_start:batch_end]
                            
                            # Extract prompts from work units
                            batch_prompts = [wu['prompt'] for wu in batch_work_units]
                            
                            # Generate responses for the batch
                            batch_responses = generate_batched_responses(
                                model, tokenizer, batch_prompts,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                batch_size=len(batch_prompts),  # Process all at once within steering context
                                max_length=max_length
                            )
                            
                            # Prepare row data for batch
                            batch_row_data = []
                            for work_unit, response in zip(batch_work_units, batch_responses):
                                row_data = {
                                    'role_id': work_unit['role_id'],
                                    'role_label': work_unit['role_label'],
                                    'question_id': work_unit['question_id'],
                                    'question_label': work_unit['question_label'],
                                    'prompt': work_unit['prompt'],
                                    'response': response,
                                    'magnitude': work_unit['magnitude']
                                }
                                batch_row_data.append(row_data)
                            
                            # Write batch to JSONL
                            if jsonl_handler.write_rows(batch_row_data):
                                completed_work_units += len(batch_work_units)
                                
                                if completed_work_units % (batch_size * 2) == 0:  # Log every few batches
                                    progress = completed_work_units / total_work_units * 100
                                    mag_pbar.set_postfix(
                                        magnitude=f"{magnitude:.0f}",
                                        progress=f"{progress:.1f}%",
                                        completed=f"{completed_work_units}/{total_work_units}"
                                    )
                            else:
                                logger.error(f"Failed to write batch of {len(batch_work_units)} work units")
                
                except Exception as e:
                    logger.error(f"Error with magnitude {magnitude}: {e}")
                    continue
                finally:
                    # Clear cache after each magnitude to prevent memory buildup
                    torch.cuda.empty_cache()
                    
                    # Periodic garbage collection
                    if len([m for m in work_by_magnitude.keys() if m <= magnitude]) % 3 == 0:
                        gc.collect()
        
        logger.info(f"Completed processing. Total work units processed: {completed_work_units}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Final cleanup
        if 'model' in locals():
            del model
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


def main():
    """Main function to orchestrate batch-based GPU steering."""
    args = parse_arguments()
    
    print("="*60)
    print("Batch-Based GPU Parallelized Steering")
    print("="*60)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load and validate inputs
    pca_results = load_pca_results(args.pca_filepath)
    
    if args.prompts_file:
        # Load combined prompts file
        combined_prompts = load_prompts_file(args.prompts_file)
        is_combined_format = True
        
        # Apply test mode filtering
        if args.test_mode:
            if len(combined_prompts) > 0:
                # In test mode, keep only first unique question_id
                first_question_id = combined_prompts[0]['question_id']
                combined_prompts = [p for p in combined_prompts if p['question_id'] == first_question_id]
                print(f"TEST MODE: Using only question_id {first_question_id} ({len(combined_prompts)} prompts)")
            else:
                print("Warning: No prompts available for test mode")
        
        prompts_data = combined_prompts
        n_unique_roles = len(set(p['role_id'] for p in combined_prompts))
        n_unique_questions = len(set(p['question_id'] for p in combined_prompts))
        
    else:
        # Load separate questions and roles files
        questions = load_questions(args.questions_file)
        roles = load_roles(args.roles_file)
        is_combined_format = False
        
        # Apply test mode filtering
        if args.test_mode:
            if len(questions) > 0:
                questions = questions[:1]  # Keep only first question
                print("TEST MODE: Using only the first question")
            else:
                print("Warning: No questions available for test mode")
        
        prompts_data = [questions, roles]
        n_unique_roles = len(roles)
        n_unique_questions = len(questions)
    
    # Validate component index
    n_components = pca_results['pca'].components_.shape[0]
    if args.component < 0 or args.component >= n_components:
        raise ValueError(f"Component index {args.component} out of range [0, {n_components-1}]")
    
    # Create work units
    work_units = create_work_units(prompts_data, args.magnitudes, is_combined_format)
    
    mode_str = "TEST MODE - " if args.test_mode else ""
    format_str = "combined prompts" if is_combined_format else f"{n_unique_questions} questions and {n_unique_roles} roles"
    print(f"{mode_str}Using PC{args.component+1} with {format_str}")
    print(f"Total work units: {len(work_units)} ({len(args.magnitudes)} magnitudes)")
    print(f"Batch size: {args.batch_size}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs available")
    
    # Validate GPU ID if specified
    if args.gpu_id is not None:
        if args.gpu_id < 0 or args.gpu_id >= n_gpus:
            raise ValueError(f"GPU ID {args.gpu_id} is out of range [0, {n_gpus-1}]")
        print(f"Using specific GPU: {args.gpu_id}")
        # Use only the specified GPU
        gpu_ids = [args.gpu_id]
        work_unit_assignments = [work_units]  # All work units go to the single GPU
    else:
        print("Using all available GPUs")
        gpu_ids = list(range(n_gpus))
        # Distribute work units across all GPUs
        work_unit_assignments = distribute_work_units(work_units, n_gpus)
    
    print("\nGPU work unit assignments:")
    for i, (gpu_id, assigned_work_units) in enumerate(zip(gpu_ids, work_unit_assignments)):
        if assigned_work_units:
            print(f"  GPU {gpu_id}: {len(assigned_work_units)} work units")
    
    # Launch worker processes
    processes = []
    for i, (gpu_id, assigned_work_units) in enumerate(zip(gpu_ids, work_unit_assignments)):
        if not assigned_work_units:  # Skip GPUs with no assigned work units
            continue
            
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                assigned_work_units,
                args.pca_filepath,
                args.component,
                args.layer,
                args.model_name,
                args.output_jsonl,
                args.max_new_tokens,
                args.temperature,
                args.batch_size,
                args.max_length
            )
        )
        p.start()
        processes.append(p)
    
    print(f"\nLaunched {len(processes)} worker processes")
    
    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            print(f"Warning: Process {i} exited with code {p.exitcode}")
        else:
            print(f"Process {i} completed successfully")
    
    print(f"\nAll workers completed!")
    print(f"Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()