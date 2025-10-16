#!/usr/bin/env python3
"""
Benchmark Evaluation with Steering

This script evaluates steered models on common benchmarks like MMLU-Pro, GSM8K, etc.
using lm-evaluation-harness. It supports multi-vector projection capping interventions.

Example usage:
uv run capped/2_benchmark_eval.py \
    --config_filepath /workspace/qwen-3-32b/evals/multi_capping_config.pt \
    --experiment_ids baseline cap_1.0 \
    --model_name google/gemma-2-27b-it \
    --tasks mmlu_pro,gsm8k \
    --output_dir /root/git/persona-subspace/evals/benchmark_results \
    --batch_size 8

"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))

from utils.steering_utils import create_projection_cap_steerer
from utils.probing_utils import load_model

# Import lm-eval
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.api.model import LM

os.environ['HF_ALLOW_CODE_EVAL'] = '1'


class SteeredHFLM(HFLM):
    """
    Wrapper around HuggingFace LM that applies steering interventions.
    """

    def __init__(
        self,
        pretrained: str,
        steering_vectors: Optional[List[torch.Tensor]] = None,
        cap_thresholds: Optional[List[float]] = None,
        layer_indices: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Args:
            pretrained: Model name or path
            steering_vectors: List of steering vectors to apply
            cap_thresholds: List of cap thresholds (one per vector)
            layer_indices: List of layer indices (one per vector)
            **kwargs: Additional arguments passed to HFLM
        """
        # Initialize parent class
        super().__init__(pretrained=pretrained, **kwargs)

        # Store steering configuration
        self.steering_vectors = steering_vectors
        self.cap_thresholds = cap_thresholds
        self.layer_indices = layer_indices
        self._steerer = None

        # If steering is configured, create the steerer
        if steering_vectors is not None and cap_thresholds is not None and layer_indices is not None:
            self._setup_steering()

    def _setup_steering(self):
        """Set up steering hooks on the model."""
        if self.steering_vectors is None:
            return

        print(f"Setting up steering with {len(self.steering_vectors)} vectors across layers {sorted(set(self.layer_indices))}")

        # Note: The steerer will handle moving vectors to the correct device for each layer
        # This is important for multi-GPU setups where layers may be on different devices
        self._steerer = create_projection_cap_steerer(
            model=self.model,
            feature_directions=self.steering_vectors,
            cap_thresholds=self.cap_thresholds,
            layer_indices=self.layer_indices,
            positions="all"
        )
        self._steerer.__enter__()
        print("Steering hooks registered successfully")

    def _cleanup_steering(self):
        """Remove steering hooks from the model."""
        if self._steerer is not None:
            self._steerer.__exit__(None, None, None)
            self._steerer = None
            print("Steering hooks removed")


def load_multi_config(config_filepath: str) -> tuple[Dict[str, Dict], List[Dict]]:
    """Load and validate multi-capping configuration file.

    Returns:
        Tuple of (vectors_dict, experiments_list)
    """
    print(f"Loading multi-capping config from {config_filepath}")

    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Config file not found: {config_filepath}")

    config = torch.load(config_filepath, weights_only=False)

    # Validate config structure
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a dictionary")

    if 'vectors' not in config or 'experiments' not in config:
        raise ValueError("Config must contain 'vectors' and 'experiments' keys")

    vectors = config['vectors']
    experiments = config['experiments']

    print(f"Loaded {len(vectors)} vectors and {len(experiments)} experiments")
    return vectors, experiments


def get_experiment_config(
    experiment_id: str,
    vectors_dict: Dict[str, Dict],
    experiments_list: List[Dict]
) -> tuple[Optional[List[torch.Tensor]], Optional[List[float]], Optional[List[int]]]:
    """Get steering configuration for a specific experiment.

    Args:
        experiment_id: ID of the experiment to load
        vectors_dict: Dictionary of all available vectors
        experiments_list: List of all experiments

    Returns:
        Tuple of (steering_vectors, cap_thresholds, layer_indices) or (None, None, None) for baseline
    """
    # Check if this is the baseline (no steering)
    if experiment_id.lower() in ['baseline', 'unsteered', 'control']:
        print(f"Experiment '{experiment_id}': No steering (baseline)")
        return None, None, None

    # Find the experiment
    experiment = None
    for exp in experiments_list:
        if exp['id'] == experiment_id:
            experiment = exp
            break

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in config")

    # Extract interventions
    interventions = experiment['interventions']
    steering_vectors = []
    cap_thresholds = []
    layer_indices = []

    for interv in interventions:
        vec_name = interv['vector']
        cap_value = interv['cap']

        if vec_name not in vectors_dict:
            raise ValueError(f"Vector '{vec_name}' not found in config")

        vec_data = vectors_dict[vec_name]
        steering_vectors.append(torch.tensor(vec_data['vector']))
        cap_thresholds.append(float(cap_value))
        layer_indices.append(vec_data['layer'])

    print(f"Experiment '{experiment_id}': {len(interventions)} interventions across layers {sorted(set(layer_indices))}")
    return steering_vectors, cap_thresholds, layer_indices


def run_evaluation(
    model_name: str,
    tasks: List[str],
    experiment_id: str,
    steering_vectors: Optional[List[torch.Tensor]],
    cap_thresholds: Optional[List[float]],
    layer_indices: Optional[List[int]],
    batch_size: int = 8,
    num_fewshot: int = 0,
    limit: Optional[int] = 1000,
    random_seed: int = 42,
    numpy_random_seed: int = 42,
    torch_random_seed: int = 42,
    fewshot_random_seed: int = 42,
    model_parallel: bool = False
) -> Dict[str, Any]:
    """Run evaluation on specified tasks with optional steering.

    Args:
        model_name: HuggingFace model name or path
        tasks: List of task names to evaluate on
        experiment_id: ID of the experiment being run
        steering_vectors: List of steering vectors (None for baseline)
        cap_thresholds: List of cap thresholds (None for baseline)
        layer_indices: List of layer indices (None for baseline)
        batch_size: Batch size for evaluation
        num_fewshot: Number of few-shot examples
        limit: Limit number of examples per task (for testing, uses all if limit > task size)
        random_seed: Random seed for python's random module
        numpy_random_seed: Random seed for numpy
        torch_random_seed: Random seed for torch
        fewshot_random_seed: Random seed for fewshot example sampling
        model_parallel: Use model parallelism (default is data parallelism)

    Returns:
        Dictionary of evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Running evaluation for experiment: {experiment_id}")
    print(f"Tasks: {tasks}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Create model wrapper with multi-GPU support
    model_kwargs = {
        'pretrained': model_name,
        'steering_vectors': steering_vectors,
        'cap_thresholds': cap_thresholds,
        'layer_indices': layer_indices,
        'batch_size': batch_size
    }

    if model_parallel:
        # Use model parallelism (split model layers across GPUs)
        print(f"Using model parallelism (device_map='auto') across {torch.cuda.device_count()} GPUs")
        model_kwargs['device_map'] = 'auto'
    else:
        # Use data parallelism (split batches across GPUs) - default
        print(f"Using data parallelism across {torch.cuda.device_count()} GPUs")
        model_kwargs['parallelize'] = True

    lm = SteeredHFLM(**model_kwargs)

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
        confirm_run_unsafe_code=True
    )

    # Cleanup steering
    lm._cleanup_steering()

    # Add experiment metadata
    results['experiment_id'] = experiment_id
    results['model_name'] = model_name

    return results


def load_existing_experiment_results(output_dir: str, experiment_id: str) -> Optional[Dict[str, Any]]:
    """Load existing results for a specific experiment if file exists."""
    output_file = os.path.join(output_dir, f"{experiment_id}_results.json")
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing results for {experiment_id}: {e}")
    return None


def merge_results(existing_results: Dict[str, Any], new_results: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new task results into existing results."""
    # Start with existing results
    merged = existing_results.copy()

    # Merge task results
    if 'results' in existing_results and 'results' in new_results:
        merged['results'].update(new_results['results'])
    elif 'results' in new_results:
        merged['results'] = new_results['results']

    # Update metadata fields (use new values)
    for key in ['experiment_id', 'model_name', 'configs', 'versions']:
        if key in new_results:
            merged[key] = new_results[key]

    return merged


def load_existing_all_results(output_dir: str) -> Dict[str, Any]:
    """Load existing all_results.json if it exists."""
    combined_output = os.path.join(output_dir, "all_results.json")
    if os.path.exists(combined_output):
        try:
            with open(combined_output, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing all_results.json: {e}")
    return {}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate steered models on benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to multi-capping config file (.pt format)"
    )

    parser.add_argument(
        "--experiment_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of experiment IDs to evaluate (use 'baseline' for unsteered, 'all' for all experiments in config)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-27b-it",
        help="HuggingFace model name or path"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu_pro,gsm8k",
        help="Comma-separated list of tasks to evaluate on"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples"
    )

    parser.add_argument(
        "--model_parallel",
        action="store_true",
        help="Use model parallelism (split model across GPUs). Default is data parallelism (split batches)."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Limit number of examples per task (for testing, uses all if limit > task size)"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for python's random module"
    )

    parser.add_argument(
        "--numpy_random_seed",
        type=int,
        default=42,
        help="Random seed for numpy"
    )

    parser.add_argument(
        "--torch_random_seed",
        type=int,
        default=42,
        help="Random seed for torch"
    )

    parser.add_argument(
        "--fewshot_random_seed",
        type=int,
        default=42,
        help="Random seed for fewshot example sampling"
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(',')]

    # Load config
    vectors_dict, experiments_list = load_multi_config(args.config_filepath)

    # Expand 'all' to all experiment IDs from config
    experiment_ids = []
    for exp_id in args.experiment_ids:
        if exp_id.lower() == 'all':
            # Add all experiment IDs from config
            config_exp_ids = [exp['id'] for exp in experiments_list]
            experiment_ids.extend(config_exp_ids)
            print(f"Expanded 'all' to {len(config_exp_ids)} experiments from config")
        else:
            experiment_ids.append(exp_id)

    print(f"\nTotal experiments to run: {len(experiment_ids)}")
    print(f"Experiments: {experiment_ids}\n")

    # Run evaluations for each experiment
    all_results = {}

    for experiment_id in experiment_ids:
        try:
            # Get experiment configuration
            steering_vectors, cap_thresholds, layer_indices = get_experiment_config(
                experiment_id, vectors_dict, experiments_list
            )

            # Run evaluation
            results = run_evaluation(
                model_name=args.model_name,
                tasks=tasks,
                experiment_id=experiment_id,
                steering_vectors=steering_vectors,
                cap_thresholds=cap_thresholds,
                layer_indices=layer_indices,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
                random_seed=args.random_seed,
                numpy_random_seed=args.numpy_random_seed,
                torch_random_seed=args.torch_random_seed,
                fewshot_random_seed=args.fewshot_random_seed,
                model_parallel=args.model_parallel
            )

            # Load existing results for this experiment
            existing_exp_results = load_existing_experiment_results(args.output_dir, experiment_id)

            # Merge if existing results found
            if existing_exp_results:
                print(f"Found existing results for {experiment_id}, merging...")

                # Track what's being added
                existing_tasks = set(existing_exp_results.get('results', {}).keys())
                new_tasks = set(results.get('results', {}).keys())
                added_tasks = new_tasks - existing_tasks
                updated_tasks = new_tasks & existing_tasks

                # Merge results
                results = merge_results(existing_exp_results, results)

                # Report changes
                if added_tasks:
                    print(f"  Added benchmarks: {', '.join(sorted(added_tasks))}")
                if updated_tasks:
                    print(f"  Updated benchmarks: {', '.join(sorted(updated_tasks))}")
                print(f"  Total benchmarks: {len(results.get('results', {}))}")

            all_results[experiment_id] = results

            # Save individual results (now merged)
            output_file = os.path.join(args.output_dir, f"{experiment_id}_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results for {experiment_id} to {output_file}")

            # Print summary
            print(f"\nResults for {experiment_id}:")
            if 'results' in results:
                for task, task_results in results['results'].items():
                    print(f"  {task}:")
                    for metric, value in task_results.items():
                        if isinstance(value, (int, float)):
                            print(f"    {metric}: {value:.4f}")

        except Exception as e:
            print(f"Error evaluating experiment {experiment_id}: {e}")
            continue

    # Load existing combined results and merge
    existing_all_results = load_existing_all_results(args.output_dir)

    # Merge current run results into existing
    for experiment_id, new_exp_results in all_results.items():
        if experiment_id in existing_all_results:
            # Merge task results for existing experiment
            existing_all_results[experiment_id] = merge_results(
                existing_all_results[experiment_id],
                new_exp_results
            )
        else:
            # Add new experiment
            existing_all_results[experiment_id] = new_exp_results

    # Save merged combined results
    combined_output = os.path.join(args.output_dir, "all_results.json")
    with open(combined_output, 'w') as f:
        json.dump(existing_all_results, f, indent=2)

    print(f"\nSaved combined results to {combined_output}")
    print(f"  Total experiments in file: {len(existing_all_results)}")
    print(f"  Updated in this run: {len(all_results)}")

    # Create summary comparison (show all experiments from merged results)
    print(f"\n{'='*60}")
    print("Summary Comparison (All Experiments)")
    print(f"{'='*60}")

    # Get all available tasks across all experiments
    all_available_tasks = set()
    for exp_results in existing_all_results.values():
        if 'results' in exp_results:
            all_available_tasks.update(exp_results['results'].keys())

    for task in sorted(all_available_tasks):
        print(f"\n{task}:")
        for experiment_id in sorted(existing_all_results.keys()):
            if 'results' in existing_all_results[experiment_id]:
                if task in existing_all_results[experiment_id]['results']:
                    task_results = existing_all_results[experiment_id]['results'][task]
                    # Try to find the main metric
                    main_metrics = ['acc', 'exact_match', 'acc_norm']
                    metric_value = None
                    for metric in main_metrics:
                        if metric in task_results:
                            metric_value = task_results[metric]
                            break
                    if metric_value is not None:
                        # Mark experiments from current run with *
                        marker = "*" if experiment_id in all_results else " "
                        print(f"  {marker} {experiment_id}: {metric_value:.4f}")

    print(f"\n{'='*60}")
    print("* = Evaluated in this run")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
