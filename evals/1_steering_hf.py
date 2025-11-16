#!/usr/bin/env python3
"""
Multi-Vector Additive Steering Script with HuggingFace

This script applies multiple steering vectors simultaneously with predefined coefficient configurations
using HuggingFace Transformers for inference:
- Loads experiment configurations with complete intervention specifications
- Supports tensor parallelism across multiple GPUs using accelerate
- Each worker processes ALL prompts for its assigned experiments in a single steering context
- Always uses chat template formatting
- Thread-safe output writing won't repeat work on restart

Example usage:
uv run 1_steering_hf.py \
    --config_filepath /workspace/qwen-3-32b/evals/steering_config.pt \
    --prompts_file /root/git/persona-subspace/evals/data/default_20.jsonl \
    --output_jsonl /root/git/persona-subspace/evals/results/steering_hf.jsonl \
    --model_name meta-llama/Llama-3.1-8B-Instruct

"""

import argparse
import json
import os
import sys
import multiprocessing as mp
import threading
import gc
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
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
from utils.internals import ProbingModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HFSteeringExperiment:
    """Represents a single steering experiment adapted for HuggingFace ActivationSteering."""
    id: str
    vectors: List[torch.Tensor]  # List of steering vectors
    coefficients: List[float]    # List of coefficients
    layer_indices: List[int]     # List of layer indices


class JSONLHandler:
    """Thread-safe JSONL handler with file locking for concurrent access."""

    def __init__(self, jsonl_path: str, samples_per_prompt: int = 1):
        self.jsonl_path = jsonl_path
        self.samples_per_prompt = samples_per_prompt
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
                        # Create unique key from id, experiment_id, and sample_id
                        if self.samples_per_prompt > 1:
                            key = (
                                data.get('id'),
                                data.get('experiment_id'),
                                data.get('sample_id', 0)
                            )
                        else:
                            # Single sample: omit sample_id
                            key = (
                                data.get('id'),
                                data.get('experiment_id')
                            )
                        existing.add(key)
                    except (json.JSONDecodeError, KeyError):
                        continue
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"Warning: Could not read existing JSONL: {e}")

        return existing

    def write_rows(self, row_data_list: List[Dict[str, Any]]) -> bool:
        """Write multiple rows to JSONL with file locking."""
        if not row_data_list:
            return True

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                    for row_data in row_data_list:
                        f.write(json.dumps(row_data) + '\n')
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return True
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to write rows after {max_retries} attempts: {e}")
                    return False
                time.sleep(0.1)  # Brief delay before retry
        return False


def load_steering_config(config_path: str, device: str = "cpu") -> List[HFSteeringExperiment]:
    """
    Load steering config from .pt file and adapt for HuggingFace ActivationSteering.

    Config format:
    {
        'vectors': {
            'vector_name': {
                'vector': torch.Tensor,  # Raw steering vector (not normalized)
                'layer': int             # Target layer index
            },
            ...
        },
        'experiments': [
            {
                'id': str,               # Unique experiment identifier
                'interventions': [
                    {
                        'vector': str,   # Vector name (key from vectors dict)
                        'coeff': float   # Coefficient to multiply vector by
                    },
                    ...
                ]
            },
            ...
        ]
    }
    """
    logger.info(f"Loading steering config from {config_path}")
    payload = torch.load(config_path, map_location="cpu", weights_only=False)

    # Extract and prepare vectors
    vector_section = payload["vectors"]
    prepared_vectors = {}
    for name, vector_payload in vector_section.items():
        vector = vector_payload["vector"]
        layer = vector_payload["layer"]

        # Convert to device and dtype (will be moved to model device later)
        tensor = vector.detach().to(dtype=torch.float32).contiguous()
        prepared_vectors[name] = (layer, tensor)

    logger.info(f"Loaded {len(prepared_vectors)} vectors")

    # Build experiments for ActivationSteering
    experiments = []
    for experiment in payload["experiments"]:
        exp_id = experiment["id"]

        # Collect vectors, coefficients, and layers for this experiment
        vectors_list = []
        coefficients_list = []
        layer_indices_list = []

        for intervention in experiment["interventions"]:
            vector_name = intervention["vector"]
            coeff = float(intervention["coeff"])
            layer_idx, vector_tensor = prepared_vectors[vector_name]

            vectors_list.append(vector_tensor)
            coefficients_list.append(coeff)
            layer_indices_list.append(layer_idx)

        experiments.append(HFSteeringExperiment(
            id=exp_id,
            vectors=vectors_list,
            coefficients=coefficients_list,
            layer_indices=layer_indices_list
        ))

    logger.info(f"Loaded {len(experiments)} experiments")
    return experiments


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-vector additive steering with GPU parallelization using HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to steering config file (.pt format) containing vectors and experiments"
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
        default=2048,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Test mode: only process first 10 prompts"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        help="Specific GPU ID to use (0-indexed). If not specified, all available GPUs will be used."
    )

    parser.add_argument(
        "--company",
        type=str,
        default="Acme Corp",
        help="Company name to substitute for {company} placeholder in prompts"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="Alex",
        help="Name to substitute for {name} placeholder in prompts"
    )

    parser.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="Only use the question text, ignore any role/system prompt"
    )

    parser.add_argument(
        "--thinking",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="Enable thinking mode for chat templates (default: True). Set to False for Qwen models."
    )

    parser.add_argument(
        "--samples_per_prompt",
        type=int,
        default=1,
        help="Number of samples to generate for each unique prompt x experiment combination"
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism per worker"
    )

    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Data type for model weights (default: bfloat16)"
    )

    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.prompts_file:
        if args.questions_file or args.roles_file:
            parser.error("--prompts_file cannot be used with --questions_file or --roles_file")
    else:
        if not args.questions_file:
            parser.error("Either --prompts_file OR --questions_file (with optional --roles_file) must be provided")

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

                prompts.append(prompt_obj)

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")

    print(f"Loaded {len(prompts)} prompts")
    return prompts


def create_prompt_data(prompts_data, company_name="Acme Corp", name_value="Alex", is_combined_format=False, no_system_prompt=False):
    """Create base prompt data (without experiment duplication).

    Returns a list of dicts with formatted prompts and metadata.
    """
    base_prompts = []

    if is_combined_format:
        # prompts_data contains combined prompts
        for prompt_data in prompts_data:
            # Format company name and name in prompts
            system_prompt_text = prompt_data.get('prompt', '').format(company=company_name, name=name_value)
            user_message = prompt_data.get('question', '').format(company=company_name, name=name_value)

            if no_system_prompt:
                # In no-system-prompt mode, only use the question text
                system_prompt = ''
            else:
                # Normal mode: use system prompt
                system_prompt = system_prompt_text

            prompt_entry = prompt_data.copy()
            prompt_entry['_system_prompt'] = system_prompt
            prompt_entry['_user_message'] = user_message
            base_prompts.append(prompt_entry)
    else:
        # prompts_data contains [questions, roles]
        questions, roles = prompts_data
        for role in roles:
            for question in questions:
                # Format company name and name in prompts
                role_text = role['_role_text'].format(company=company_name, name=name_value)
                question_text = question['_question_text'].format(company=company_name, name=name_value)

                if no_system_prompt:
                    # In no-system-prompt mode, only use the question text
                    system_prompt = ''
                    user_message = question_text
                else:
                    # Normal mode: combine role and question
                    system_prompt = role_text
                    user_message = question_text

                prompt_entry = {}
                # Copy all fields from both role and question
                prompt_entry.update(role)
                prompt_entry.update(question)

                prompt_entry['_system_prompt'] = system_prompt
                prompt_entry['_user_message'] = user_message

                # Clean up temporary fields
                if '_role_text' in prompt_entry:
                    del prompt_entry['_role_text']
                if '_question_text' in prompt_entry:
                    del prompt_entry['_question_text']

                base_prompts.append(prompt_entry)

    return base_prompts


def generate_batched_responses(
    model,
    tokenizer,
    prompt_data_list: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    max_length: int = 2048,
    thinking: bool = True
) -> List[str]:
    """
    Generate responses for a batch of prompts efficiently using real batch inference.
    Always uses chat template formatting.
    """
    try:
        if not prompt_data_list:
            return []

        # Determine if this is a Gemma model (no system prompt support)
        is_gemma = 'gemma-2' in model.config.name_or_path.lower() if hasattr(model.config, 'name_or_path') else False

        # Format prompts for chat
        formatted_prompts = []
        for prompt_data in prompt_data_list:
            system_prompt = prompt_data.get('_system_prompt', '')
            user_message = prompt_data.get('_user_message', '')

            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                if is_gemma or not system_prompt:
                    # Gemma model or no system prompt: only use user message
                    content = (f"{system_prompt} {user_message}".strip() if system_prompt else user_message)
                    messages = [{"role": "user", "content": content}]
                else:
                    # Use system prompt + user message
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]

                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
                )
                formatted_prompts.append(formatted_prompt)
            else:
                # Fallback to simple concatenation
                content = (f"{system_prompt} {user_message}".strip() if system_prompt else user_message)
                formatted_prompts.append(content)

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
        return [""] * len(prompt_data_list)


def load_model_with_tensor_parallelism(model_name: str, gpu_ids: List[int], chat_model_name: Optional[str] = None, dtype_value: torch.dtype = torch.bfloat16):
    """Load model with optional tensor parallelism across multiple GPUs."""
    if len(gpu_ids) == 1:
        # Single GPU - simple loading
        device = f"cuda:{gpu_ids[0]}"
        pm = ProbingModel(model_name, device=device, chat_model_name=chat_model_name, dtype=dtype_value)
        pm.model.eval()
        return pm.model, pm.tokenizer
    else:
        # Multi-GPU tensor parallelism - use ProbingModel with device=None for auto sharding
        logger.info(f"Loading model with tensor parallelism across GPUs: {gpu_ids}")
        pm = ProbingModel(model_name, device=None, chat_model_name=chat_model_name, dtype=dtype_value)
        pm.model.eval()
        logger.info(f"Model loaded with layers distributed across available GPUs")
        return pm.model, pm.tokenizer


def worker_process(
    worker_id: int,
    gpu_ids: List[int],
    experiment_queue: mp.Queue,
    base_prompts: List[Dict[str, Any]],
    existing_combinations: set,
    config_filepath: str,
    model_name: str,
    output_jsonl: str,
    max_new_tokens: int,
    temperature: float,
    max_length: int,
    batch_size: int,
    thinking: bool,
    samples_per_prompt: int,
    dtype_value: torch.dtype,
    progress_queue: mp.Queue
):
    """
    Worker process that pulls experiments from a queue and processes them.
    For each experiment, applies steering once and generates all prompts in that context.
    """
    torch.set_float32_matmul_precision('high')

    try:
        # Set CUDA_VISIBLE_DEVICES to restrict this worker to its assigned GPUs
        # This ensures device_map='auto' only uses these GPUs
        gpu_str = ','.join(map(str, gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

        logger_worker = logging.getLogger(f"Worker-{worker_id}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - Worker-{worker_id} (GPUs {gpu_str}) - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger_worker.addHandler(handler)
        logger_worker.setLevel(logging.INFO)

        logger_worker.info(f"Starting worker on GPUs {gpu_str}")

        # Load model on assigned GPUs
        if len(gpu_ids) == 1:
            device = f"cuda:{gpu_ids[0]}"
        else:
            device = f"cuda:{gpu_ids[0]}"  # Primary device for single-GPU operations

        model, tokenizer = load_model_with_tensor_parallelism(model_name, gpu_ids, dtype_value=dtype_value)
        logger_worker.info(f"Model loaded on GPUs {gpu_str}")

        # Load experiments
        logger_worker.info(f"Loading steering config: {config_filepath}")
        all_experiments = load_steering_config(config_filepath, device=device)
        experiments_by_id = {exp.id: exp for exp in all_experiments}

        # Initialize JSONL handler
        jsonl_handler = JSONLHandler(output_jsonl, samples_per_prompt)

        # Clear GPU cache before starting
        torch.cuda.empty_cache()

        total_processed = 0

        # Process experiments from queue until empty
        while True:
            try:
                # Get experiment ID from queue with timeout
                experiment_id = experiment_queue.get(timeout=5)
                if experiment_id is None:  # Sentinel value
                    logger_worker.info("Received stop signal")
                    experiment_queue.put(None)  # Re-add for other workers
                    break
            except:
                # Queue empty or timeout
                logger_worker.info("Queue empty, exiting")
                break

            # Get experiment object
            if experiment_id not in experiments_by_id:
                logger_worker.error(f"Unknown experiment ID: {experiment_id}")
                continue

            experiment = experiments_by_id[experiment_id]

            logger_worker.info(f"\n{'='*60}")
            logger_worker.info(f"Processing experiment '{experiment_id}'")
            logger_worker.info(f"Additive steering on {len(experiment.layer_indices)} intervention(s)")

            # Filter out already completed work for this experiment
            work_items = []
            for prompt_data in base_prompts:
                for sample_id in range(samples_per_prompt):
                    # Create combination key
                    if samples_per_prompt > 1:
                        key = (prompt_data.get('id'), experiment_id, sample_id)
                    else:
                        key = (prompt_data.get('id'), experiment_id)

                    if key not in existing_combinations:
                        work_items.append((prompt_data, sample_id if samples_per_prompt > 1 else None))

            if not work_items:
                logger_worker.info(f"All work already completed for experiment '{experiment_id}', skipping")
                continue

            logger_worker.info(f"Processing {len(work_items)} work items for experiment '{experiment_id}'")

            # Apply steering ONCE for this entire experiment
            # ActivationSteering will handle device placement automatically
            try:
                with ActivationSteering(
                    model=model,
                    steering_vectors=experiment.vectors,
                    coefficients=experiment.coefficients,
                    layer_indices=experiment.layer_indices,
                    intervention_type="addition",
                    positions="all"
                ) as steerer:
                    logger_worker.info(f"[steer] Steering context active for experiment '{experiment_id}'")

                    # Generate ALL responses for this experiment in batches within steering context
                    all_responses = []
                    with tqdm(total=len(work_items), desc=f"Worker-{worker_id} exp={experiment_id}",
                             unit="prompts", leave=False, position=worker_id) as pbar:
                        for i in range(0, len(work_items), batch_size):
                            batch_work_items = work_items[i:i + batch_size]
                            batch_prompt_data = [item[0] for item in batch_work_items]

                            # Generate responses for this batch
                            batch_responses = generate_batched_responses(
                                model, tokenizer, batch_prompt_data,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                max_length=max_length,
                                thinking=thinking
                            )

                            all_responses.extend(batch_responses)
                            pbar.update(len(batch_work_items))

                    logger_worker.info(f"[steer] Generated {len(all_responses)} responses")

                logger_worker.info(f"[steer] Steering context released for experiment '{experiment_id}'")

                # Prepare output rows
                output_rows = []
                for (prompt_data, sample_id), response in zip(work_items, all_responses):
                    if response.strip():
                        row_data = prompt_data.copy()
                        row_data['response'] = response
                        row_data['experiment_id'] = experiment_id

                        if sample_id is not None:
                            row_data['sample_id'] = sample_id

                        # Remove underscore fields
                        fields_to_remove = [k for k in row_data.keys() if k.startswith('_')]
                        for field in fields_to_remove:
                            del row_data[field]

                        output_rows.append(row_data)
                    else:
                        logger_worker.warning(f"Empty response for id={prompt_data.get('id')}, experiment={experiment_id}")

                # Write all results for this experiment
                if output_rows:
                    if jsonl_handler.write_rows(output_rows):
                        logger_worker.info(f"Wrote {len(output_rows)} rows for experiment '{experiment_id}'")
                        total_processed += len(output_rows)

                        # Report progress
                        progress_queue.put({
                            'worker_id': worker_id,
                            'experiment_id': experiment_id,
                            'processed': len(output_rows)
                        })
                    else:
                        logger_worker.error(f"Failed to write results for experiment '{experiment_id}'")

            except Exception as e:
                logger_worker.error(f"Error processing experiment '{experiment_id}': {e}")
                continue
            finally:
                # Cleanup after each experiment
                torch.cuda.empty_cache()
                gc.collect()

        logger_worker.info(f"\n{'='*60}")
        logger_worker.info(f"Worker completed. Processed {total_processed} total items")

        # Signal completion
        progress_queue.put({'worker_id': worker_id, 'done': True})

    except Exception as e:
        logger.error(f"Fatal error in worker {worker_id}: {e}")
        import traceback
        traceback.print_exc()
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


def progress_monitor(progress_queue: mp.Queue, total_experiments: int, n_workers: int):
    """Monitor progress from worker processes."""
    completed_experiments = set()
    active_workers = n_workers

    with tqdm(total=total_experiments, desc="Experiments completed", unit="exp") as pbar:
        while active_workers > 0:
            try:
                result = progress_queue.get(timeout=60)

                if result.get('done'):
                    active_workers -= 1
                    logger.info(f"Worker {result['worker_id']} finished")
                    continue

                exp_id = result.get('experiment_id')
                if exp_id and exp_id not in completed_experiments:
                    completed_experiments.add(exp_id)
                    pbar.update(1)
                    pbar.set_postfix(
                        workers=active_workers,
                        last_worker=result.get('worker_id')
                    )
            except:
                # Timeout - check if we should exit
                if active_workers == 0:
                    break


def main():
    """Main function to orchestrate multi-vector GPU parallelized additive steering."""
    args = parse_arguments()

    # Map dtype string to torch dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    dtype_value = dtype_map.get(args.dtype, torch.bfloat16)

    print("="*60)
    print("Multi-Vector Additive Steering with GPU Parallelization (HuggingFace)")
    print("="*60)

    # Create output directory
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load experiments
    logger.info(f"Loading steering config: {args.config_filepath}")
    all_experiments = load_steering_config(args.config_filepath)
    logger.info(f"Loaded {len(all_experiments)} experiments")

    # Load prompts
    if args.prompts_file:
        combined_prompts = load_prompts_file(args.prompts_file)
        is_combined_format = True

        if args.test_mode and len(combined_prompts) > 10:
            combined_prompts = combined_prompts[:10]
            print(f"TEST MODE: Using only first 10 prompts")

        prompts_data = combined_prompts
    else:
        questions = load_questions(args.questions_file)

        # Load roles file if provided, otherwise create a dummy role with empty text
        if args.roles_file:
            roles = load_roles(args.roles_file)
        else:
            # Create a single dummy role with empty text (no system prompt)
            roles = [{'id': 0, '_role_text': ''}]
            print("No roles file provided - using questions only (no system prompt)")

        is_combined_format = False

        if args.test_mode and len(questions) > 10:
            questions = questions[:10]
            print(f"TEST MODE: Using only first 10 questions")

        prompts_data = [questions, roles]

    # Create base prompt data (formatted but not duplicated by experiment)
    base_prompts = create_prompt_data(
        prompts_data, args.company, args.name, is_combined_format, args.no_system_prompt
    )

    print(f"Created {len(base_prompts)} base prompts")
    print(f"Will generate {len(base_prompts) * len(all_experiments) * args.samples_per_prompt} total responses")

    # Read existing combinations
    jsonl_handler = JSONLHandler(args.output_jsonl, args.samples_per_prompt)
    existing_combinations = jsonl_handler.read_existing_combinations()
    print(f"Found {len(existing_combinations)} existing combinations")

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs available")

    # Determine GPU IDs to use
    if args.gpu_id is not None:
        if args.gpu_id < 0 or args.gpu_id >= n_gpus:
            raise ValueError(f"GPU ID {args.gpu_id} is out of range [0, {n_gpus-1}]")
        available_gpus = [args.gpu_id]
    else:
        available_gpus = list(range(n_gpus))

    print(f"Available GPUs: {available_gpus}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")

    # Calculate number of workers based on tensor parallelism
    n_available = len(available_gpus)
    if n_available % args.tensor_parallel_size != 0:
        raise ValueError(
            f"Number of available GPUs ({n_available}) must be divisible by "
            f"tensor_parallel_size ({args.tensor_parallel_size})"
        )

    n_workers = n_available // args.tensor_parallel_size
    print(f"Will spawn {n_workers} workers, each using {args.tensor_parallel_size} GPU(s)")

    # Assign GPU ranges to each worker
    worker_gpu_assignments = []
    for worker_idx in range(n_workers):
        start_idx = worker_idx * args.tensor_parallel_size
        end_idx = start_idx + args.tensor_parallel_size
        worker_gpus = available_gpus[start_idx:end_idx]
        worker_gpu_assignments.append(worker_gpus)
        print(f"  Worker {worker_idx}: GPUs {worker_gpus}")

    # Create experiment and progress queues
    experiment_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Populate experiment queue with all experiment IDs
    for experiment in all_experiments:
        experiment_queue.put(experiment.id)

    # Add sentinel values for workers to know when to stop
    for _ in range(n_workers):
        experiment_queue.put(None)

    print(f"\nPopulated queue with {len(all_experiments)} experiments")
    print(f"Workers will pull experiments dynamically for load balancing")

    # Launch worker processes
    processes = []
    for worker_id, worker_gpus in enumerate(worker_gpu_assignments):
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                worker_gpus,
                experiment_queue,
                base_prompts,
                existing_combinations,
                args.config_filepath,
                args.model_name,
                args.output_jsonl,
                args.max_new_tokens,
                args.temperature,
                args.max_length,
                args.batch_size,
                args.thinking,
                args.samples_per_prompt,
                dtype_value,
                progress_queue
            )
        )
        p.start()
        processes.append(p)

    print(f"\nLaunched {len(processes)} worker processes")

    # Start progress monitor
    monitor_process = mp.Process(
        target=progress_monitor,
        args=(progress_queue, len(all_experiments), len(processes))
    )
    monitor_process.start()

    # Wait for all workers
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            print(f"Warning: Worker process {i} exited with code {p.exitcode}")
        else:
            print(f"Worker process {i} completed successfully")

    monitor_process.join(timeout=5)
    if monitor_process.is_alive():
        monitor_process.terminate()

    print(f"\nAll workers completed!")
    print(f"Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()
