#!/usr/bin/env python3
"""
Multi-Vector Capping Script with VLLM Steering

This script applies multiple steering vectors simultaneously with predefined cap configurations
using VLLM for efficient inference:
- Loads experiment configurations with complete intervention specifications
- Divides experiments across GPUs (round-robin assignment)
- Each worker processes ALL prompts for its assigned experiments in a single steering context
- VLLM handles batching internally with continuous batching
- Thread-safe output writing won't repeat work on restart

Example usage:
uv run 1_capping_vllm.py \
    --config_filepath /workspace/qwen-3-32b/evals/multi_capping_config.pt \
    --prompts_file /root/git/persona-subspace/evals/data/default_20.jsonl \
    --output_jsonl /root/git/persona-subspace/evals/results/multi_capping_vllm.jsonl

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
from chatspace.generation.vllm_steer_model import VLLMSteerModel, VLLMSteeringConfig
from chatspace.generation.compat import load_legacy_role_trait_config, LegacyExperiment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-vector capping with GPU parallelization using VLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to multi-capping config file (.pt format) containing vectors and experiments"
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
        "--vllm_batch_size",
        type=int,
        default=256,
        help="Batch size for VLLM generation (to avoid OOM)"
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
        default=0.9,
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


def format_prompt_for_chat(prompt_data: Dict[str, Any], tokenizer) -> str:
    """Format a single prompt using the tokenizer's chat template."""
    system_prompt = prompt_data.get('_system_prompt', '')
    user_message = prompt_data.get('_user_message', '')

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        if not system_prompt:
            # No system prompt: only use user message
            messages = [{"role": "user", "content": user_message}]
        else:
            # Use system prompt + user message
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

        # Use tokenizer's apply_chat_template for proper formatting
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted_prompt
    else:
        # Fallback to simple concatenation
        content = f"{system_prompt} {user_message}".strip() if system_prompt else user_message
        return content


def worker_process(
    gpu_id: int,
    experiment_queue: mp.Queue,
    base_prompts: List[Dict[str, Any]],
    existing_combinations: set,
    config_filepath: str,
    model_name: str,
    output_jsonl: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_prompt: int,
    vllm_batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    dtype: str,
    progress_queue: mp.Queue
):
    """
    Worker process that pulls experiments from a queue and processes them.
    For each experiment, applies steering once and generates all prompts in that context.
    """
    torch.set_float32_matmul_precision('high')

    try:
        logger = logging.getLogger(f"GPU-{gpu_id}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - GPU-{gpu_id} - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info(f"Starting worker, pulling experiments from queue")

        # Load experiments in the worker process (avoids passing tensors via multiprocessing)
        logger.info(f"Loading legacy config: {config_filepath}")
        all_experiments = load_legacy_role_trait_config(config_filepath)
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

        # Initialize VLLMSteerModel on assigned GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True)
        tokenizer = vllm_model.llm.get_tokenizer()
        logger.info(f"VLLM model loaded on GPU {gpu_id}")
        logger.info(f"Hidden size: {vllm_model.hidden_size}, Layer count: {vllm_model.layer_count}")

        # Initialize JSONL handler
        jsonl_handler = JSONLHandler(output_jsonl, samples_per_prompt)

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
            logger.info(f"Projection caps on {num_interventions} layer(s): {unique_layers}")

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
                formatted_prompt = format_prompt_for_chat(prompt_data, tokenizer)
                formatted_prompts.append(formatted_prompt)

            # Apply steering ONCE for this entire experiment
            with vllm_model.steering(steering_spec):
                logger.info(f"[steer] Steering context active for experiment '{experiment_id}'")

                # Generate ALL responses for this experiment in one steering context
                # Process in batches with progress bar
                all_responses = []
                with tqdm(total=len(formatted_prompts), desc=f"GPU-{gpu_id} exp={experiment_id}",
                         unit="prompts", leave=False, position=gpu_id) as pbar:
                    for i in range(0, len(formatted_prompts), vllm_batch_size):
                        batch_prompts = formatted_prompts[i:i + vllm_batch_size]

                        sampling_params = SamplingParams(
                            max_tokens=max_new_tokens,
                            temperature=temperature,
                        )

                        outputs = vllm_model.llm.generate(
                            prompts=batch_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False
                        )

                        batch_responses = [output.outputs[0].text.strip() for output in outputs]
                        all_responses.extend(batch_responses)
                        pbar.update(len(batch_prompts))

                logger.info(f"[steer] Generated {len(all_responses)} responses")

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
                        'gpu_id': gpu_id,
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

        # Signal completion
        progress_queue.put({'gpu_id': gpu_id, 'done': True})

    except Exception as e:
        logger.error(f"Fatal error in worker: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Final cleanup
        if 'vllm_model' in locals():
            del vllm_model
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
                    logger.info(f"Worker GPU-{result['gpu_id']} finished")
                    continue

                exp_id = result.get('experiment_id')
                if exp_id and exp_id not in completed_experiments:
                    completed_experiments.add(exp_id)
                    pbar.update(1)
                    pbar.set_postfix(
                        workers=active_workers,
                        last_gpu=result.get('gpu_id')
                    )
            except:
                # Timeout - check if we should exit
                if active_workers == 0:
                    break


def main():
    """Main function to orchestrate multi-vector GPU parallelized steering with VLLM."""
    args = parse_arguments()

    print("="*60)
    print("Multi-Vector Capping with GPU Parallelization (VLLM)")
    print("="*60)

    # Create output directory
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load experiments
    logger.info(f"Loading legacy config: {args.config_filepath}")
    all_experiments = load_legacy_role_trait_config(args.config_filepath)
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
        roles = load_roles(args.roles_file)
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
        gpu_ids = [args.gpu_id]
    else:
        gpu_ids = list(range(n_gpus))

    print(f"Using GPUs: {gpu_ids}")

    # Create experiment and progress queues
    experiment_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Populate experiment queue with all experiment IDs
    for experiment in all_experiments:
        experiment_queue.put(experiment.id)

    # Add sentinel values for workers to know when to stop
    for _ in gpu_ids:
        experiment_queue.put(None)

    print(f"\nPopulated queue with {len(all_experiments)} experiments")
    print(f"Workers will pull experiments dynamically for load balancing")

    # Launch worker processes
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                experiment_queue,
                base_prompts,
                existing_combinations,
                args.config_filepath,
                args.model_name,
                args.output_jsonl,
                args.max_new_tokens,
                args.temperature,
                args.samples_per_prompt,
                args.vllm_batch_size,
                args.tensor_parallel_size,
                args.gpu_memory_utilization,
                args.max_model_len,
                args.dtype,
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
