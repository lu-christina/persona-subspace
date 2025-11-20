#!/usr/bin/env python3
"""
Multi-Vector Additive Steering Script with VLLM

This script applies multiple steering vectors simultaneously with predefined coefficient configurations
using VLLM for efficient inference:
- Loads experiment configurations with complete intervention specifications
- Divides experiments across GPUs (dynamic queue-based assignment)
- Each worker processes ALL prompts for its assigned experiments in a single steering context
- VLLM handles batching internally with continuous batching
- Thread-safe output writing won't repeat work on restart

Example usage:
uv run 1_steering_vllm.py \
    --config_filepath /workspace/qwen-3-32b/evals/steering_config.pt \
    --prompts_file /root/git/persona-subspace/evals/data/default_20.jsonl \
    --output_jsonl /root/git/persona-subspace/evals/results/steering_vllm.jsonl

"""

import argparse
import json
import os
import sys
import multiprocessing as mp
import gc
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import fcntl
from tqdm import tqdm

import torch
from vllm import SamplingParams

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))
sys.path.append(str(Path.home() / 'git' / 'chatspace'))
torch.set_float32_matmul_precision('high')

# Import VLLM steering components from chatspace
from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
    SteeringSpec,
    LayerSteeringSpec,
    AddSpec
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SteeringExperiment:
    """Represents a single steering experiment with multiple vector additions."""
    id: str
    spec: SteeringSpec


class JSONLHandler:
    """Thread-safe JSONL handler with file locking for concurrent access."""

    def __init__(self, jsonl_path: str, samples_per_prompt: int = 1):
        self.jsonl_path = jsonl_path
        self.samples_per_prompt = samples_per_prompt

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


def load_steering_config(config_path: str) -> List[SteeringExperiment]:
    """
    Load steering config from .pt file.

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

    If config_path is None, returns a single baseline experiment with no steering.
    """
    # Handle baseline mode (no config file)
    if config_path is None:
        logger.info("No config file provided - running in baseline mode (no steering)")
        return [SteeringExperiment(
            id='baseline_unsteered',
            spec=SteeringSpec(layers={})
        )]

    logger.info(f"Loading steering config from {config_path}")
    payload = torch.load(config_path, map_location="cpu")

    # Extract and prepare vectors (no normalization, use as-is)
    vector_section = payload["vectors"]
    prepared_vectors = {}
    for name, vector_payload in vector_section.items():
        vector = vector_payload["vector"]
        layer = vector_payload["layer"]

        # Convert to CPU float32 and make contiguous (no normalization)
        tensor = vector.detach().to(dtype=torch.float32).contiguous()

        prepared_vectors[name] = (layer, tensor)

    logger.info(f"Loaded {len(prepared_vectors)} vectors")

    # Build experiments with AddSpec
    experiments = []
    for experiment in payload["experiments"]:
        exp_id = experiment["id"]
        layers = {}

        for intervention in experiment["interventions"]:
            vector_name = intervention["vector"]
            coeff = float(intervention["coeff"])
            layer_idx, vector_tensor = prepared_vectors[vector_name]

            # Create AddSpec for additive steering
            add_spec = AddSpec(
                vector=vector_tensor.clone(),
                scale=coeff
            )

            layers[layer_idx] = LayerSteeringSpec(
                add=add_spec,
                projection_cap=None,  # No projection capping
                ablation=None         # No ablation
            )

        experiments.append(SteeringExperiment(
            id=exp_id,
            spec=SteeringSpec(layers=layers)
        ))

    logger.info(f"Loaded {len(experiments)} experiments")
    return experiments


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-vector additive steering with GPU parallelization using VLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config_filepath",
        type=str,
        required=False,
        default=None,
        help="Path to steering config file (.pt format) containing vectors and experiments. "
             "If omitted, runs without steering (baseline mode)."
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
        "--model",
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

    # VLLM-specific parameters
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism per worker"
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization (0.0-1.0)"
    )

    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length"
    )

    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32", "auto"],
        default="auto",
        help="Model dtype"
    )

    parser.add_argument(
        "--concurrent_batch_size",
        type=int,
        default=64,
        help="Number of prompts to generate concurrently (higher = faster but more VRAM). Default: 64"
    )

    parser.add_argument(
        "--completion_mode",
        action="store_true",
        help="Use raw text completion mode instead of chat templates (for base models)"
    )

    parser.add_argument(
        "--stop_sequences",
        type=str,
        nargs="*",
        default=None,
        help="List of stop sequences where generation should stop (e.g., '\\n\\n' '###')"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (top_p). Default: 1.0 (disabled)"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling parameter. Default: -1 (disabled)"
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty. Default: 1.0 (disabled)"
    )

    parser.add_argument(
        "--experiment_ids",
        type=str,
        nargs="*",
        default=None,
        help="List of specific experiment IDs to run. If not specified, all experiments from config will be run."
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


def format_prompt_for_chat(prompt_data: Dict[str, Any], tokenizer, model_name: str = "", enable_thinking: bool = True, completion_mode: bool = False) -> str:
    """Format a single prompt using the tokenizer's chat template or raw text for completion mode."""
    system_prompt = prompt_data.get('_system_prompt', '')
    user_message = prompt_data.get('_user_message', '')

    # Completion mode: just use raw text concatenation
    if completion_mode:
        if system_prompt:
            return f"{system_prompt}\n\n{user_message}"
        else:
            return user_message

    # Chat mode: use chat template formatting
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Check if this is a Gemma model - they append system prompts to user messages
        is_gemma = 'gemma' in model_name.lower()

        if not system_prompt:
            # No system prompt: only use user message
            messages = [{"role": "user", "content": user_message}]
        elif is_gemma:
            # Gemma: append system prompt to user message
            combined_content = f"{system_prompt}\n\n{user_message}"
            messages = [{"role": "user", "content": combined_content}]
        else:
            # Other models: use system message role
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

        # Use tokenizer's apply_chat_template for proper formatting
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        return formatted_prompt
    else:
        # Fallback to simple concatenation
        content = f"{system_prompt} {user_message}".strip() if system_prompt else user_message
        return content


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
    samples_per_prompt: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    dtype: str,
    enable_thinking: bool,
    concurrent_batch_size: int,
    completion_mode: bool,
    stop_sequences: Optional[List[str]],
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    progress_queue: mp.Queue
):
    """
    Worker process that pulls experiments from a queue and processes them.
    For each experiment, applies steering once and generates all prompts in that context.
    """
    torch.set_float32_matmul_precision('high')

    try:
        # Set CUDA_VISIBLE_DEVICES for this worker's GPU subset (using physical GPU IDs)
        gpu_str = ','.join(map(str, gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

        logger = logging.getLogger(f"Worker-{worker_id}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - Worker-{worker_id} (GPUs {gpu_str}) - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info(f"Starting worker on GPUs {gpu_str} (tensor_parallel_size={tensor_parallel_size})")

        # Load experiments in the worker process (avoids passing tensors via multiprocessing)
        logger.info(f"Loading steering config: {config_filepath}")
        all_experiments = load_steering_config(config_filepath)
        experiments_by_id = {exp.id: exp for exp in all_experiments}

        # Determine bootstrap layers from ALL experiments (worker doesn't know which it will get)
        bootstrap_layers = set()
        for exp in all_experiments:
            bootstrap_layers.update(exp.spec.layers.keys())
        bootstrap_layers = tuple(sorted(bootstrap_layers))
        logger.info(f"Bootstrap layers for steering: {bootstrap_layers}")

        # Create VLLMSteeringConfig
        vllm_cfg = VLLMSteeringConfig(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            bootstrap_layers=bootstrap_layers,
        )

        # Initialize VLLMSteerModel on assigned GPUs
        vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True)
        tokenizer = vllm_model.tokenizer
        logger.info(f"VLLM model created on GPUs {gpu_str}")
        logger.info(f"Hidden size: {vllm_model.hidden_size}, Layer count: {vllm_model.layer_count}")

        # Initialize JSONL handler
        jsonl_handler = JSONLHandler(output_jsonl, samples_per_prompt)

        # ALL async operations in ONE function
        import asyncio
        import uuid

        async def worker_main_async():
            """Main async worker loop - initialize engine and process all experiments."""
            # Initialize async engine ONCE
            await vllm_model._ensure_engine_initialized()
            logger.info(f"VLLM async engine initialized")

            total_processed = 0

            # Process experiments from queue until empty
            while True:
                try:
                    # Get experiment ID from queue with timeout
                    experiment_id = experiment_queue.get(timeout=5)
                    if experiment_id is None:  # Sentinel value
                        logger.info("Received stop signal")
                        experiment_queue.put(None)  # Re-add for other workers
                        break
                except:
                    # Queue empty or timeout
                    logger.info("Queue empty, exiting")
                    break

                # Get experiment object
                if experiment_id not in experiments_by_id:
                    logger.error(f"Unknown experiment ID: {experiment_id}")
                    continue

                experiment = experiments_by_id[experiment_id]
                steering_spec = experiment.spec

                unique_layers = sorted(steering_spec.layers.keys())
                num_interventions = len(steering_spec.layers)
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing experiment '{experiment_id}'")
                logger.info(f"Additive steering on {num_interventions} layer(s): {unique_layers}")

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
                    logger.info(f"All work already completed for experiment '{experiment_id}', skipping")
                    continue

                logger.info(f"Processing {len(work_items)} work items for experiment '{experiment_id}'")

                # Format all prompts for this experiment
                formatted_prompts = []
                for prompt_data, _ in work_items:
                    formatted_prompt = format_prompt_for_chat(prompt_data, tokenizer, model_name, enable_thinking, completion_mode)
                    formatted_prompts.append(formatted_prompt)

                # Check if steering is needed
                has_steering = bool(steering_spec.layers)

                # Apply steering and generate - we're already in async context
                # Send prompts in concurrent batches to utilize vLLM's continuous batching
                try:
                    # Push steering spec only if there are layers to steer
                    if has_steering:
                        await vllm_model.push_steering_spec(steering_spec)
                        logger.info(f"[steer] Steering context active for experiment '{experiment_id}'")
                    else:
                        logger.info(f"[baseline] Running without steering for experiment '{experiment_id}'")

                    sampling_params = SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        stop=stop_sequences,
                    )

                    # Send prompts in concurrent waves for better throughput
                    # This allows vLLM's continuous batching to work effectively
                    all_responses = []

                    mode_label = "steer" if has_steering else "baseline"
                    logger.info(f"[{mode_label}] Generating {len(formatted_prompts)} responses with concurrent batching (batch_size={concurrent_batch_size})")
                    with tqdm(total=len(formatted_prompts), desc=f"Worker-{worker_id} exp={experiment_id}",
                             unit="prompts", leave=False, position=worker_id) as pbar:

                        for i in range(0, len(formatted_prompts), concurrent_batch_size):
                            batch_prompts = formatted_prompts[i:i + concurrent_batch_size]

                            # Create concurrent tasks for this batch
                            tasks = [
                                vllm_model.generate([prompt], sampling_params)
                                for prompt in batch_prompts
                            ]

                            # Wait for all concurrent requests to complete
                            batch_results = await asyncio.gather(*tasks)

                            # Extract text from results (each is a list with one item)
                            batch_texts = [result[0].strip() for result in batch_results]
                            all_responses.extend(batch_texts)
                            pbar.update(len(batch_texts))

                    logger.info(f"[{mode_label}] Generated {len(all_responses)} responses")

                finally:
                    # Pop steering spec only if we pushed one
                    if has_steering:
                        await vllm_model.pop_steering_spec()
                        logger.info(f"[steer] Steering context released for experiment '{experiment_id}'")

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
                        logger.warning(f"Empty response for id={prompt_data.get('id')}, experiment={experiment_id}")

                # Write all results for this experiment
                if output_rows:
                    if jsonl_handler.write_rows(output_rows):
                        logger.info(f"Wrote {len(output_rows)} rows for experiment '{experiment_id}'")
                        total_processed += len(output_rows)

                        # Report progress
                        progress_queue.put({
                            'worker_id': worker_id,
                            'experiment_id': experiment_id,
                            'processed': len(output_rows)
                        })
                    else:
                        logger.error(f"Failed to write results for experiment '{experiment_id}'")

                # Cleanup after each experiment
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(f"\n{'='*60}")
            logger.info(f"Worker completed. Processed {total_processed} total items")

            # Properly shutdown the vLLM engine before exiting
            try:
                logger.info("Shutting down vLLM engine...")
                if hasattr(vllm_model, 'engine') and vllm_model.engine is not None:
                    # Try to abort all requests first
                    try:
                        if hasattr(vllm_model.engine, 'abort_all'):
                            vllm_model.engine.abort_all()
                        vllm_model.engine._request_tracker.abort_all()
                        logger.info("Aborted all engine requests")
                    except:
                        pass

                    # Shutdown the background loop
                    try:
                        if hasattr(vllm_model.engine, 'shutdown_background_loop'):
                            await vllm_model.engine.shutdown_background_loop()
                            logger.info("Engine background loop shut down")
                    except:
                        pass

                    # Destroy the engine
                    try:
                        if hasattr(vllm_model.engine, 'engine'):
                            del vllm_model.engine.engine
                        del vllm_model.engine
                        logger.info("Engine object destroyed")
                    except:
                        pass

            except Exception as e:
                logger.warning(f"Error during engine shutdown: {e}")

            # Signal completion
            progress_queue.put({'worker_id': worker_id, 'done': True})
            return total_processed

        # Run the async worker main function
        asyncio.run(worker_main_async())

        logger.info("Work complete, forcefully cleaning up engine...")

    except Exception as e:
        logger.error(f"Fatal error in worker: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Final cleanup
        if 'vllm_model' in locals():
            try:
                # Properly destroy vLLM's distributed workers
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
                logger.info("vLLM model parallel destroyed")
            except Exception as e:
                logger.warning(f"Error destroying model parallel: {e}")

            # Delete model object
            try:
                del vllm_model
                logger.info("vLLM model deleted")
            except Exception as e:
                logger.warning(f"Error deleting model: {e}")

        # Properly destroy CUDA contexts before shutdown
        if torch.cuda.is_available():
            try:
                # Synchronize all CUDA streams
                for device_id in range(torch.cuda.device_count()):
                    torch.cuda.set_device(device_id)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    # Reset the device to force cleanup
                    torch.cuda.reset_peak_memory_stats(device_id)
                logger.info("CUDA contexts synchronized and cleared")
            except Exception as e:
                logger.warning(f"Error cleaning CUDA: {e}")

        # Shutdown Ray and kill all worker processes (vLLM uses Ray for distributed inference)
        try:
            import ray
            if ray.is_initialized():
                # First, try to kill all Ray actors explicitly
                try:
                    import ray._private.worker
                    worker = ray._private.worker.global_worker
                    if hasattr(worker, 'core_worker'):
                        # Kill all actors
                        core_worker = worker.core_worker
                        if hasattr(core_worker, 'disconnect'):
                            core_worker.disconnect()
                    logger.info("Disconnected Ray core worker")
                except Exception as e:
                    logger.warning(f"Error disconnecting Ray workers: {e}")

                # Now shutdown Ray
                ray.shutdown()
                logger.info("Ray shutdown complete")

                # Give Ray a moment to clean up
                import time
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Error shutting down Ray: {e}")

        # Final garbage collection
        gc.collect()

        logger.info("Worker process cleanup complete, forcing process exit...")

        # Force process termination to ensure Ray workers are killed
        # By this point, all work is done and cleanup has been attempted
        import os
        os._exit(0)


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
    """Main function to orchestrate multi-vector GPU parallelized additive steering with VLLM."""
    args = parse_arguments()

    print("="*60)
    print("Multi-Vector Additive Steering with GPU Parallelization (VLLM)")
    print("="*60)

    # Create output directory
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load experiments
    logger.info(f"Loading steering config: {args.config_filepath}")
    all_experiments = load_steering_config(args.config_filepath)
    logger.info(f"Loaded {len(all_experiments)} experiments")

    # Filter experiments if specific IDs are requested
    if args.experiment_ids is not None and len(args.experiment_ids) > 0:
        requested_ids = set(args.experiment_ids)
        available_ids = {exp.id for exp in all_experiments}

        # Validate that all requested IDs exist
        missing_ids = requested_ids - available_ids
        if missing_ids:
            raise ValueError(
                f"Requested experiment IDs not found in config: {sorted(missing_ids)}\n"
                f"Available IDs: {sorted(available_ids)}"
            )

        # Filter to only requested experiments
        all_experiments = [exp for exp in all_experiments if exp.id in requested_ids]
        logger.info(f"Filtered to {len(all_experiments)} experiments: {sorted(requested_ids)}")
    else:
        logger.info(f"Running all experiments from config")

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

    # Read existing combinations
    jsonl_handler = JSONLHandler(args.output_jsonl, args.samples_per_prompt)
    existing_combinations = jsonl_handler.read_existing_combinations()

    # Calculate work statistics
    total_possible = len(base_prompts) * len(all_experiments) * args.samples_per_prompt
    already_completed = len(existing_combinations)
    will_generate = total_possible - already_completed

    print(f"Total possible responses: {total_possible}")
    print(f"Already completed: {already_completed}")
    print(f"Will generate: {will_generate} new responses")

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Read physical GPU IDs from CUDA_VISIBLE_DEVICES if set (for task spooler compatibility)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids_str = os.environ['CUDA_VISIBLE_DEVICES']
        available_gpus = [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]
        print(f"Found {len(available_gpus)} GPUs from CUDA_VISIBLE_DEVICES: {available_gpus}")
    elif args.gpu_id is not None:
        # Explicit GPU ID specified
        n_gpus = torch.cuda.device_count()
        if args.gpu_id < 0 or args.gpu_id >= n_gpus:
            raise ValueError(f"GPU ID {args.gpu_id} is out of range [0, {n_gpus-1}]")
        available_gpus = [args.gpu_id]
        print(f"Using specified GPU: {args.gpu_id}")
    else:
        # No external constraint, use all GPUs
        n_gpus = torch.cuda.device_count()
        available_gpus = list(range(n_gpus))
        print(f"Found {n_gpus} GPUs available: {available_gpus}")

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
                args.samples_per_prompt,
                args.tensor_parallel_size,
                args.gpu_memory_utilization,
                args.max_model_len,
                args.dtype,
                args.thinking,
                args.concurrent_batch_size,
                args.completion_mode,
                args.stop_sequences,
                args.top_p,
                args.top_k,
                args.repetition_penalty,
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

    # Wait for all workers to complete
    print("\nWaiting for workers to complete...")
    for i, p in enumerate(processes):
        p.join()  # Wait indefinitely for worker to finish

        if p.exitcode != 0:
            print(f"Warning: Worker process {i} exited with code {p.exitcode}")
        else:
            print(f"Worker process {i} completed successfully")

    # Terminate monitor process
    print("Terminating monitor process...")
    monitor_process.join(timeout=5)
    if monitor_process.is_alive():
        monitor_process.terminate()
        monitor_process.join(timeout=2)
        if monitor_process.is_alive():
            monitor_process.kill()

    print(f"\nAll workers completed!")
    print(f"Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()
