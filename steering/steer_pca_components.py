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
from pathlib import Path
from typing import List, Dict, Any

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
    
    parser.add_argument(
        "--questions_filepath", 
        type=str,
        required=True,
        help="Path to questions JSONL file"
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
        default=[-4000.0, -3000.0, -2000.0, -1500.0, 0.0, 1500.0, 2000.0, 3000.0, 4000.0],
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
        default=5,
        help="Number of questions to use for testing (0 = all questions)"
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


def load_questions(questions_filepath: str, test_questions: int = 0, question_range: List[int] = None) -> List[str]:
    """Load questions from JSONL file."""
    print(f"Loading questions from {questions_filepath}")
    
    if not os.path.exists(questions_filepath):
        raise FileNotFoundError(f"Questions file not found: {questions_filepath}")
    
    questions = []
    with open(questions_filepath, 'r') as f:
        for line in f:
            try:
                question_obj = json.loads(line.strip())
                if 'question' in question_obj:
                    questions.append(question_obj['question'])
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
    
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


def worker_process(
    gpu_id: int,
    assigned_components: List[int], 
    pca_filepath: str,
    questions: List[str],
    magnitudes: List[float],
    layer: int,
    model_name: str,
    output_dir: str
):
    """
    Worker process that handles steering for assigned PC components on a single GPU.
    
    Args:
        gpu_id: CUDA device ID to use
        assigned_components: List of PC component indices to process
        pca_filepath: Path to PCA results file
        questions: List of question strings
        magnitudes: List of steering magnitudes
        layer: Layer index for steering
        model_name: Model name to load
        output_dir: Output directory for results
    """
    try:
        print(f"Worker GPU {gpu_id}: Starting with components {assigned_components}")
        
        # Load model on assigned GPU
        device = f"cuda:{gpu_id}"
        model, tokenizer = load_model(model_name, device=device)
        print(f"Worker GPU {gpu_id}: Model loaded on {device}")
        
        # Load PCA results
        pca_results = torch.load(pca_filepath, weights_only=False)
        
        # Process each assigned component
        for pc_idx in assigned_components:
            print(f"Worker GPU {gpu_id}: Processing PC{pc_idx+1}")
            
            output_file = os.path.join(output_dir, f"pc{pc_idx+1}.json")
            
            # Load existing results if file exists
            steered_results = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        steered_results = json.load(f)
                    print(f"Worker GPU {gpu_id}: Loaded existing results for PC{pc_idx+1}")
                except Exception as e:
                    print(f"Worker GPU {gpu_id}: Warning - couldn't load existing results: {e}")
                    steered_results = {}
            
            # Get steering vector and ensure correct device/dtype
            steering_vector = torch.from_numpy(pca_results['pca'].components_[pc_idx])
            steering_vector = steering_vector.to(device=device, dtype=model.dtype)
            
            print(f"Worker GPU {gpu_id}: Steering vector shape: {steering_vector.shape}")
            
            # Process each magnitude
            total_combinations = len(magnitudes) * len(questions)
            completed = 0
            
            for magnitude in magnitudes:
                print(f"Worker GPU {gpu_id}: Processing magnitude {magnitude}")
                
                try:
                    with ActivationSteering(
                        model=model,
                        steering_vectors=[steering_vector],
                        coefficients=magnitude,
                        layer_indices=layer,
                        intervention_type="addition",
                        positions="all"
                    ) as steerer:
                        for question in questions:
                            # Initialize structure if needed
                            if question not in steered_results:
                                steered_results[question] = {}
                            if magnitude not in steered_results[question]:
                                steered_results[question][magnitude] = []
                            
                            # Generate response (always append to existing list)
                            response = generate_text(model, tokenizer, question, chat_format=True)
                            steered_results[question][magnitude].append(response)
                            
                            completed += 1
                            if completed % 5 == 0:
                                print(f"Worker GPU {gpu_id}: Progress {completed}/{total_combinations}")
                
                except Exception as e:
                    print(f"Worker GPU {gpu_id}: Error with magnitude {magnitude}: {e}")
                    continue
            
            # Save results
            try:
                with open(output_file, 'w') as f:
                    json.dump(steered_results, f, indent=2)
                print(f"Worker GPU {gpu_id}: Saved results to {output_file}")
            except Exception as e:
                print(f"Worker GPU {gpu_id}: Error saving results: {e}")
        
        print(f"Worker GPU {gpu_id}: Completed all assigned components")
        
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
    questions = load_questions(args.questions_filepath, args.test_questions, args.question_range)
    
    # Validate component indices
    n_components = pca_results['pca'].components_.shape[0]
    for comp in args.components:
        if comp < 0 or comp >= n_components:
            raise ValueError(f"Component index {comp} out of range [0, {n_components-1}]")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs available")
    
    # Distribute components across GPUs
    component_assignments = [[] for _ in range(n_gpus)]
    for i, comp in enumerate(args.components):
        gpu_id = i % n_gpus
        component_assignments[gpu_id].append(comp)
    
    print("\nGPU assignments:")
    for gpu_id, components in enumerate(component_assignments):
        if components:
            pc_names = [f"PC{c+1}" for c in components]
            print(f"  GPU {gpu_id}: {pc_names}")
    
    # Launch worker processes
    processes = []
    for gpu_id, components in enumerate(component_assignments):
        if not components:  # Skip GPUs with no assigned components
            continue
            
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                components,
                args.pca_filepath,
                questions,
                args.magnitudes,
                args.layer,
                args.model_name,
                args.output_dir
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
    
    print("\nAll workers completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    main()