#!/usr/bin/env python3
"""
Steering Script

This script steers prompts with a PC:
- Work units created in batches of the same magnitude
- Parallelized across all available GPU
- Won't repeat work on restart

Output: JSONL with role_id, role_label, question_id, question_label, prompt, response, magnitude

uv run steering.py \
    --pca_filepath /workspace/roles_traits/pca/layer22_roles_pos23_traits_pos40-100.pt \
    --questions_file /root/git/persona-subspace/evals/data/questions/harmbench.jsonl \
    --roles_file /root/git/persona-subspace/evals/data/roles/good_evil.jsonl \
    --output_jsonl /root/git/persona-subspace/evals/results/roles_traits/harmbench.jsonl 

uv run steering.py \
    --pca_filepath /workspace/roles_traits/pca/layer22_roles_pos23_traits_pos40-100.pt \
    --prompts_file /root/git/persona-subspace/evals/data/roles_20.jsonl \
    --magnitudes -4000.0 -2000.0 \
    --output_jsonl /root/git/persona-subspace/evals/results/roles_traits/roles_20.jsonl

uv run steering.py \
    --pca_filepath /workspace/roles_traits/pca/layer22_roles_pos23_traits_pos40-100.pt \
    --prompts_file /root/git/persona-subspace/evals/data/default_20.jsonl \
    --output_jsonl /root/git/persona-subspace/evals/results/roles_traits/default_20.jsonl

"""

import argparse
import json
import os
import sys
import multiprocessing as mp
import threading
import gc
import logging
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import fcntl
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


class WorkQueueManager:
    """Manages work queue creation, persistence, and progress tracking."""
    
    def __init__(self, queue_state_file: Optional[str] = None):
        self.queue_state_file = queue_state_file
        
    def create_work_batches(
        self, 
        work_units: List[Dict], 
        batch_size: int,
        existing_combinations: set = None
    ) -> List[List[Dict]]:
        """Create batches of work units for queue processing."""
        if existing_combinations is None:
            existing_combinations = set()
            
        # Filter out existing combinations
        filtered_work_units = []
        for work_unit in work_units:
            combination_key = (work_unit['prompt'], work_unit['magnitude'])
            if combination_key not in existing_combinations:
                filtered_work_units.append(work_unit)
        
        logger.info(f"Filtered {len(work_units)} work units to {len(filtered_work_units)} new units")
        
        # Group by magnitude for more efficient processing
        work_by_magnitude = defaultdict(list)
        for work_unit in filtered_work_units:
            work_by_magnitude[work_unit['magnitude']].append(work_unit)
        
        # Create batches within each magnitude group
        all_batches = []
        for magnitude, magnitude_work_units in work_by_magnitude.items():
            # Create batches of the specified size
            for i in range(0, len(magnitude_work_units), batch_size):
                batch = magnitude_work_units[i:i + batch_size]
                all_batches.append(batch)
        
        logger.info(f"Created {len(all_batches)} work batches from {len(filtered_work_units)} work units")
        return all_batches
    
    def save_queue_state(self, remaining_batches: List[List[Dict]], completed_count: int):
        """Save current queue state for restart capability."""
        if not self.queue_state_file:
            return
            
        state = {
            'remaining_batches': remaining_batches,
            'completed_count': completed_count,
            'timestamp': time.time()
        }
        
        try:
            with open(self.queue_state_file, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Saved queue state: {len(remaining_batches)} batches remaining")
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")
    
    def load_queue_state(self) -> Tuple[List[List[Dict]], int]:
        """Load queue state from file if it exists."""
        if not self.queue_state_file or not os.path.exists(self.queue_state_file):
            return [], 0
            
        try:
            with open(self.queue_state_file, 'rb') as f:
                state = pickle.load(f)
            
            remaining_batches = state.get('remaining_batches', [])
            completed_count = state.get('completed_count', 0)
            timestamp = state.get('timestamp', 0)
            
            logger.info(f"Loaded queue state: {len(remaining_batches)} batches remaining, "
                       f"{completed_count} completed, saved at {time.ctime(timestamp)}")
            
            return remaining_batches, completed_count
        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")
            return [], 0


def generate_batched_responses(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    max_length: int = 2048
) -> List[str]:
    """
    Generate responses for a batch of prompts efficiently using real batch inference.
    All prompts in the batch should have the same steering magnitude for safety.
    """
    try:
        if not prompts:
            return []
        
        # Format prompts for chat if needed
        formatted_prompts = []
        for prompt in prompts:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                # Apply chat template
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted_prompt)
            else:
                formatted_prompts.append(prompt)
        
        # Tokenize all prompts at once
        batch_inputs = tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(model.device)
        
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
        logger.error(f"Error processing batch: {e}")
        # Fallback to sequential processing if batch fails
        logger.info("Falling back to sequential processing")
        try:
            batch_responses = []
            for prompt in prompts:
                response = generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    chat_format=True
                )
                batch_responses.append(response)
            return batch_responses
        except Exception as fallback_e:
            logger.error(f"Fallback processing also failed: {fallback_e}")
            return [""] * len(prompts)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimized GPU parallelized steering script with hybrid queue-batch processing",
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
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing prompts"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--queue_state_file",
        type=str,
        help="File to save/load queue state for restart capability"
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
            'combined_prompt': f"{prompt_obj['prompt']} {prompt_obj['question']}".strip() if prompt_obj['prompt'] else prompt_obj['question']
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
                        'prompt': f"{role['text']} {question['text']}".strip() if role['text'] else question['text'],
                        'magnitude': magnitude
                    }
                    work_units.append(work_unit)
    
    return work_units


def worker_process(
    gpu_id: int,
    work_queue: mp.Queue,
    results_queue: mp.Queue,
    pca_filepath: str,
    component: int,
    layer: int,
    model_name: str,
    output_jsonl: str,
    max_new_tokens: int,
    temperature: float,
    max_length: int,
    total_batches: int
):
    """
    Optimized worker process that pulls work batches from queue and processes them with batch efficiency.
    """
    torch.set_float32_matmul_precision('high')
    
    try:
        logger = logging.getLogger(f"GPU-{gpu_id}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - GPU-{gpu_id} - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info(f"Starting optimized work queue consumer")
        
        # Load model on assigned GPU
        device = f"cuda:{gpu_id}"
        model, tokenizer = load_model(model_name, device=device)
        model.eval()  # Set to evaluation mode for inference
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
        
        processed_batches = 0
        processed_work_units = 0
        
        # Process work batches from queue
        while True:
            try:
                # Get work batch from queue with timeout
                try:
                    work_batch = work_queue.get(timeout=10)  # 10 second timeout
                except:
                    # Queue is empty or timeout reached
                    logger.info("No more work available, exiting")
                    break
                
                if work_batch is None:  # Sentinel value to indicate end
                    logger.info("Received end signal, exiting")
                    work_queue.put(None)  # Re-add sentinel for other workers
                    break
                
                logger.info(f"Processing batch {processed_batches + 1}/{total_batches} ({len(work_batch)} work units)")
                
                # Group work units by magnitude for efficient processing
                work_by_magnitude = defaultdict(list)
                for work_unit in work_batch:
                    work_by_magnitude[work_unit['magnitude']].append(work_unit)
                
                batch_processed_count = 0
                
                # Process each magnitude group
                for magnitude, magnitude_work_units in work_by_magnitude.items():
                    try:
                        with ActivationSteering(
                            model=model,
                            steering_vectors=[steering_vector],
                            coefficients=magnitude,
                            layer_indices=layer,
                            intervention_type="addition",
                            positions="all"
                        ) as steerer:
                            
                            # Extract prompts from work units
                            batch_prompts = [wu['prompt'] for wu in magnitude_work_units]
                            
                            # Generate responses for the batch
                            batch_responses = generate_batched_responses(
                                model, tokenizer, batch_prompts,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                max_length=max_length
                            )
                            
                            # Prepare row data for batch (skip empty responses)
                            batch_row_data = []
                            for work_unit, response in zip(magnitude_work_units, batch_responses):
                                if response.strip():  # Only save non-empty responses
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
                                else:
                                    logger.warning(f"Skipping empty response for role_id={work_unit['role_id']}, question_id={work_unit['question_id']}, magnitude={work_unit['magnitude']}")
                            
                            # Write batch to JSONL
                            if batch_row_data:  # Only write if there's data to write
                                if jsonl_handler.write_rows(batch_row_data):
                                    batch_processed_count += len(batch_row_data)
                                else:
                                    logger.error(f"Failed to write batch of {len(batch_row_data)} work units")
                            else:
                                logger.info("No valid responses to write for this batch")
                                batch_processed_count += len(magnitude_work_units)  # Still count as processed
                    
                    except Exception as e:
                        logger.error(f"Error processing magnitude {magnitude}: {e}")
                        continue
                    finally:
                        # Clear cache after each magnitude
                        torch.cuda.empty_cache()
                
                processed_batches += 1
                processed_work_units += batch_processed_count
                
                # Report progress to main process
                results_queue.put({
                    'gpu_id': gpu_id,
                    'processed_work_units': batch_processed_count,
                    'batch_count': 1
                })
                
                logger.info(f"Completed batch {processed_batches}/{total_batches}, total work units: {processed_work_units}")
                
                # Periodic garbage collection
                if processed_batches % 2 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing work batch: {e}")
                continue
        
        logger.info(f"Worker completed. Processed {processed_batches} batches, {processed_work_units} work units")
        
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


def progress_monitor(results_queue: mp.Queue, total_work_units: int, n_workers: int):
    """Monitor progress from worker processes and display unified progress."""
    completed_work_units = 0
    active_workers = n_workers
    
    with tqdm(total=total_work_units, desc="Overall progress", unit="work_units") as pbar:
        while active_workers > 0:
            try:
                result = results_queue.get(timeout=30)  # 30 second timeout
                
                if result is None:  # Worker finished
                    active_workers -= 1
                    continue
                
                processed_count = result.get('processed_work_units', 0)
                gpu_id = result.get('gpu_id', 'unknown')
                
                completed_work_units += processed_count
                pbar.update(processed_count)
                pbar.set_postfix(
                    completed=completed_work_units,
                    workers=active_workers,
                    last_gpu=gpu_id
                )
                
            except:
                # Timeout or queue closed
                break
    
    logger.info(f"Progress monitoring completed. Total processed: {completed_work_units}")


def main():
    """Main function to orchestrate optimized hybrid queue-batch GPU steering."""
    args = parse_arguments()
    
    print("="*60)
    print("Optimized Hybrid Queue-Batch GPU Parallelized Steering")
    print("="*60)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize work queue manager
    queue_manager = WorkQueueManager(args.queue_state_file)
    
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
    
    # Get existing combinations to filter work units
    jsonl_handler = JSONLHandler(args.output_jsonl)
    existing_combinations = jsonl_handler.read_existing_combinations()
    
    # Try to load previous queue state
    remaining_batches, completed_count = queue_manager.load_queue_state()
    
    if remaining_batches:
        print(f"Resuming from previous state: {len(remaining_batches)} batches remaining")
        work_batches = remaining_batches
    else:
        # Create new work batches
        work_batches = queue_manager.create_work_batches(
            work_units, args.batch_size, existing_combinations
        )
    
    if not work_batches:
        print("No work to process")
        return
    
    total_work_units = sum(len(batch) for batch in work_batches)
    print(f"Processing {total_work_units} work units in {len(work_batches)} batches")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs available")
    
    # Determine GPU IDs to use
    if args.gpu_id is not None:
        if args.gpu_id < 0 or args.gpu_id >= n_gpus:
            raise ValueError(f"GPU ID {args.gpu_id} is out of range [0, {n_gpus-1}]")
        gpu_ids = [args.gpu_id]
    else:
        gpu_ids = list(range(n_gpus))
    
    print(f"Using GPUs: {gpu_ids}")
    
    # Create work queue and results queue
    work_queue = mp.Queue()
    results_queue = mp.Queue()
    
    # Populate work queue with batches
    for batch in work_batches:
        work_queue.put(batch)
    
    # Add sentinel values for workers to know when to stop
    for _ in gpu_ids:
        work_queue.put(None)
    
    # Launch worker processes
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                work_queue,
                results_queue,
                args.pca_filepath,
                args.component,
                args.layer,
                args.model_name,
                args.output_jsonl,
                args.max_new_tokens,
                args.temperature,
                args.max_length,
                len(work_batches)
            )
        )
        p.start()
        processes.append(p)
    
    print(f"\nLaunched {len(processes)} worker processes")
    
    # Start progress monitor in separate process
    monitor_process = mp.Process(
        target=progress_monitor,
        args=(results_queue, total_work_units, len(processes))
    )
    monitor_process.start()
    
    # Wait for all worker processes to complete
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            print(f"Warning: Worker process {i} exited with code {p.exitcode}")
        else:
            print(f"Worker process {i} completed successfully")
    
    # Signal monitor to stop
    results_queue.put(None)
    monitor_process.join()
    
    # Clean up queue state file if all work completed
    if args.queue_state_file and os.path.exists(args.queue_state_file):
        os.remove(args.queue_state_file)
        print("Removed queue state file (all work completed)")
    
    print(f"\nAll workers completed!")
    print(f"Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()