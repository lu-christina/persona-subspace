#!/usr/bin/env python3
"""
Activation extraction script for different character traits.

This script takes the output from 1_inference_traits.py and extracts activations
for each trait prompt, computing contrast vectors relative to control activations.

Usage:
    uv run roleplay/2_activations_traits.py \
        --traits-file roleplay/results/gemma-2-27b/conversations/0_traits.json \
        --control-file /workspace/roleplay/gemma-2-27b/activations_65.pt \
        --output-file /workspace/roleplay/gemma-2-27b/0_traits_activations.pt \
        --model-name google/gemma-2-27b-it
"""

import argparse
import json
import sys
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.internals import ProbingModel, ActivationExtractor, ConversationEncoder


def load_traits_data(traits_file: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load and validate traits data file from 1_inference_traits.py output.
    
    Args:
        traits_file: Path to traits JSON file
        
    Returns:
        (samples, prompts) where samples is the samples array and prompts is ordered list of prompts
    """
    try:
        with open(traits_file, 'r', encoding='utf-8') as f:
            traits_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Traits file not found: {traits_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {traits_file}: {e}")
    
    if "samples" not in traits_data:
        raise ValueError("Traits file must contain a 'samples' key")
    
    samples = traits_data["samples"]
    
    # Validate sample structure
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {i} must be a dictionary")
        required_fields = ["id", "trait_name", "format_id", "prompt"]
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Sample {i} must have a '{field}' field")
    
    # Sort samples by ID to ensure correct order
    samples_sorted = sorted(samples, key=lambda x: x["id"])
    
    # Extract prompts in ID order
    prompts = [sample["prompt"] for sample in samples_sorted]
    
    return samples_sorted, prompts


def load_control_activations(control_file: str, verbose: bool = False) -> torch.Tensor:
    """
    Load control activations from .pt file.
    
    Args:
        control_file: Path to control activations .pt file
        verbose: Enable verbose logging
        
    Returns:
        torch.Tensor of shape (num_layers, hidden_dim)
    """
    try:
        control_data = torch.load(control_file, map_location='cpu')
    except FileNotFoundError:
        raise FileNotFoundError(f"Control file not found: {control_file}")
    except Exception as e:
        raise ValueError(f"Error loading control file {control_file}: {e}")
    
    # Extract control activations from the loaded data
    # The control file should have activations at index 0 (control persona)
    if "activations" in control_data:
        control_activations = control_data["activations"][0]  # Shape: (num_layers, hidden_dim)
    else:
        raise ValueError(f"Control file {control_file} must contain 'activations' key")
    
    if verbose:
        print(f"Loaded control activations shape: {control_activations.shape}")
    
    return control_activations


def extract_all_activations(model, tokenizer, prompts: List[str], verbose: bool = False) -> torch.Tensor:
    """
    Extract activations from all layers for all prompts.
    
    Args:
        model: Loaded model
        tokenizer: Model tokenizer  
        prompts: List of system prompts
        verbose: Enable verbose logging
        
    Returns:
        torch.Tensor of shape (num_prompts, num_layers, hidden_dim)
    """
    num_hidden_layers = model.config.num_hidden_layers
    
    if verbose:
        print(f"Extracting activations from all {num_hidden_layers} layers...")
    
    # Extract from all layers
    layer_range = list(range(num_hidden_layers))
    pm = ProbingModel.from_existing(model, tokenizer)
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    activations_dict = extractor.for_prompts(prompts, layer=layer_range, swap=False)
    
    if verbose:
        print(f"Successfully extracted activations for {len(prompts)} prompts")
        for layer_idx, activation_tensor in activations_dict.items():
            print(f"  Layer {layer_idx}: {activation_tensor.shape}")
    
    # Stack into tensor of shape (num_layers, num_prompts, hidden_dim)
    stacked_activations = torch.stack([activations_dict[layer_idx] for layer_idx in layer_range])
    
    # Transpose to (num_prompts, num_layers, hidden_dim) 
    activations = stacked_activations.transpose(0, 1)
    
    if verbose:
        print(f"Final activations shape: {activations.shape}")
    
    return activations


def compute_contrast_vectors(activations: torch.Tensor, control_activations: torch.Tensor,
                           verbose: bool = False) -> torch.Tensor:
    """
    Compute contrast vectors for all samples relative to control activations.
    
    Args:
        activations: Shape (num_samples, num_layers, hidden_dim)
        control_activations: Shape (num_layers, hidden_dim)
        verbose: Enable verbose logging
        
    Returns:
        torch.Tensor of shape (num_samples, num_layers, hidden_dim)
    """
    num_samples, num_layers, hidden_dim = activations.shape
    
    if verbose:
        print(f"Computing contrast vectors for {num_samples} samples relative to control")
    
    # Broadcast control activations to all samples and compute difference
    # control_activations: (num_layers, hidden_dim) -> (1, num_layers, hidden_dim)
    # activations: (num_samples, num_layers, hidden_dim)
    # Result: (num_samples, num_layers, hidden_dim)
    contrast_vectors = activations - control_activations.unsqueeze(0)
    
    if verbose:
        for i in range(min(5, num_samples)):  # Show first 5 samples
            contrast_norm = torch.norm(contrast_vectors[i]).item()
            print(f"  Sample {i}: contrast norm = {contrast_norm:.3f}")
        if num_samples > 5:
            print(f"  ... and {num_samples - 5} more samples")
    
    return contrast_vectors


def save_results(activations: torch.Tensor, contrast_vectors: torch.Tensor,
                control_activations: torch.Tensor, samples: List[Dict[str, Any]],
                output_file: str, model_name: str, control_file: str,
                verbose: bool = False) -> None:
    """
    Save results to .pt file.
    
    Args:
        activations: Activation tensors (num_samples, num_layers, hidden_dim)
        contrast_vectors: Contrast vectors (num_samples, num_layers, hidden_dim)
        control_activations: Control activations (num_layers, hidden_dim)
        samples: List of sample metadata dictionaries
        output_file: Output file path
        model_name: Model name used
        control_file: Path to control file
        verbose: Enable verbose logging
    """
    # Prepare results dictionary
    results = {
        "metadata": {
            "model_name": model_name,
            "control_file": control_file,
            "num_layers": activations.shape[1],
            "num_samples": activations.shape[0],
            "timestamp": datetime.now().isoformat()
        },
        "samples": samples,
        "activations": activations,
        "control_activations": control_activations,
        "contrast_vectors": contrast_vectors
    }
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to .pt file
    torch.save(results, output_file)
    
    if verbose:
        print(f"Saved results to: {output_file}")
        print(f"  Activations shape: {activations.shape}")
        print(f"  Control activations shape: {control_activations.shape}")
        print(f"  Contrast vectors shape: {contrast_vectors.shape}")
        print(f"  Number of samples: {len(samples)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract activations and compute contrast vectors for role-playing personas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    uv run roleplay/2_activations_traits.py \\
        --traits-file results/0_traits.json \\
        --control-file /workspace/roleplay/gemma-2-27b/activations_65.pt \\
        --output-file results/traits_activations.pt

    # Custom model
    uv run roleplay/2_activations_traits.py \\
        --traits-file results/0_traits.json \\
        --control-file /workspace/roleplay/gemma-2-27b/activations_65.pt \\
        --output-file results/traits_activations.pt \\
        --model-name google/gemma-2-9b-it
        """
    )
    
    # Required arguments
    parser.add_argument('--traits-file', type=str, required=True,
                       help='Path to JSON file containing traits data from 1_inference_traits.py')
    parser.add_argument('--control-file', type=str, required=True,
                       help='Path to .pt file containing control activations')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Path to output .pt file for results')
    
    # Model configuration  
    parser.add_argument('--model-name', type=str, default='google/gemma-2-27b-it',
                       help='HuggingFace model name (default: google/gemma-2-27b-it)')
    
    # Optional flags
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Print configuration if verbose
    if args.verbose:
        print("Configuration:")
        print(f"  Traits file: {args.traits_file}")
        print(f"  Control file: {args.control_file}")
        print(f"  Output file: {args.output_file}")
        print(f"  Model: {args.model_name}")
        print()
    
    try:
        # Load traits file
        print("Loading traits file...")
        samples, prompts = load_traits_data(args.traits_file)
        print(f"Loaded {len(samples)} samples")
        
        if args.verbose:
            for i, sample in enumerate(samples[:3]):
                print(f"  Sample {sample['id']}: {sample['trait_name']} (format {sample['format_id']})")
                print(f"    Prompt: {sample['prompt'][:50]}...")
            if len(samples) > 3:
                print(f"  ... and {len(samples) - 3} more samples")
        
        # Load control activations
        print("Loading control activations...")
        control_activations = load_control_activations(args.control_file, verbose=args.verbose)
        
        # Load model with multi-GPU support
        print(f"Loading model: {args.model_name}")
        pm = ProbingModel(args.model_name)
        model = pm.model
        tokenizer = pm.tokenizer
        print(f"Model loaded successfully on device: {model.device}")
        
        # Extract activations from all layers
        print("Extracting activations from all layers...")
        activations = extract_all_activations(
            model, tokenizer, prompts, verbose=args.verbose
        )
        
        # Compute contrast vectors
        print("Computing contrast vectors...")
        contrast_vectors = compute_contrast_vectors(
            activations, control_activations, verbose=args.verbose
        )
        
        # Save results
        print("Saving results...")
        save_results(
            activations=activations,
            contrast_vectors=contrast_vectors,
            control_activations=control_activations,
            samples=samples,
            output_file=args.output_file,
            model_name=args.model_name,
            control_file=args.control_file,
            verbose=args.verbose
        )
        
        print("✓ Activation extraction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())