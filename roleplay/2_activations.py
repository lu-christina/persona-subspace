#!/usr/bin/env python3
"""
Activation extraction script for role-playing personas.

This script extracts the activation extraction logic from 2_activations.ipynb,
allowing for batch processing of personas to extract activations from all layers
and compute contrast vectors relative to a control persona.

Usage:
    uv run roleplay/2_activations.py \
        --personas-file prompts/personas_short.json \
        --output-file results/activations.pt \
        --model-name google/gemma-2-9b-it
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


def load_personas(personas_file: str, control_key: str = "control") -> Tuple[Dict[str, Any], List[str]]:
    """
    Load and validate personas file.
    
    Args:
        personas_file: Path to personas JSON file
        control_key: Key for control persona
        
    Returns:
        (personas_data, persona_prompts) where persona_prompts is ordered list of system prompts
    """
    try:
        with open(personas_file, 'r', encoding='utf-8') as f:
            personas_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Personas file not found: {personas_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {personas_file}: {e}")
    
    if "personas" not in personas_data:
        raise ValueError("Personas file must contain a 'personas' key")
    
    if control_key not in personas_data["personas"]:
        raise ValueError(f"Control persona '{control_key}' not found in personas file")
    
    # Validate persona structure
    for persona_name, persona_info in personas_data["personas"].items():
        if not isinstance(persona_info, dict):
            raise ValueError(f"Persona '{persona_name}' must be a dictionary")
        if "system_prompt" not in persona_info:
            raise ValueError(f"Persona '{persona_name}' must have a 'system_prompt' field")
    
    # Extract system prompts in consistent order (control first)
    persona_names = list(personas_data["personas"].keys())
    if control_key in persona_names:
        # Move control to front
        persona_names.remove(control_key)
        persona_names = [control_key] + persona_names
    
    persona_prompts = [personas_data["personas"][name]["system_prompt"] for name in persona_names]
    
    return personas_data, persona_prompts, persona_names


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
    if model.config.model_type == "gemma3":
        num_hidden_layers = model.config.text_config.num_hidden_layers
    else:
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


def compute_contrast_vectors(activations: torch.Tensor, control_idx: int = 0, 
                           verbose: bool = False) -> torch.Tensor:
    """
    Compute contrast vectors for all personas relative to control.
    
    Args:
        activations: Shape (num_personas, num_layers, hidden_dim)
        control_idx: Index of control persona (usually 0)
        verbose: Enable verbose logging
        
    Returns:
        torch.Tensor of shape (num_personas, num_layers, hidden_dim) where 
        control persona has zero contrast vector
    """
    num_personas, num_layers, hidden_dim = activations.shape
    contrast_vectors = torch.zeros_like(activations)
    
    control_activation = activations[control_idx]  # Shape: (num_layers, hidden_dim)
    
    if verbose:
        print(f"Computing contrast vectors relative to control persona (index {control_idx})")
    
    for i in range(num_personas):
        if i == control_idx:
            # Control persona gets zero contrast vector (already initialized to zeros)
            if verbose:
                print(f"  Persona {i} (control): zero vector")
        else:
            # Compute contrast: persona - control for all layers
            contrast_vectors[i] = activations[i] - control_activation
            if verbose:
                contrast_norm = torch.norm(contrast_vectors[i]).item()
                print(f"  Persona {i}: contrast norm = {contrast_norm:.3f}")
    
    return contrast_vectors


def save_results(activations: torch.Tensor, contrast_vectors: torch.Tensor,
                personas_data: Dict[str, Any], persona_names: List[str],
                output_file: str, model_name: str, personas_file: str,
                control_key: str, verbose: bool = False) -> None:
    """
    Save results to .pt file.
    
    Args:
        activations: Activation tensors
        contrast_vectors: Contrast vectors
        personas_data: Original personas data
        persona_names: Ordered list of persona names
        output_file: Output file path
        model_name: Model name used
        personas_file: Path to personas file
        control_key: Control persona key
        verbose: Enable verbose logging
    """
    # Prepare results dictionary
    results = {
        "metadata": {
            "model_name": model_name,
            "personas_file": personas_file,
            "control_key": control_key,
            "num_layers": activations.shape[1],
            "timestamp": datetime.now().isoformat(),
            "num_personas": len(persona_names)
        },
        "personas": personas_data,
        "activations": activations,
        "contrast_vectors": contrast_vectors, 
        "persona_names": persona_names
    }
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to .pt file
    torch.save(results, output_file)
    
    if verbose:
        print(f"Saved results to: {output_file}")
        print(f"  Activations shape: {activations.shape}")
        print(f"  Contrast vectors shape: {contrast_vectors.shape}")
        print(f"  Number of personas: {len(persona_names)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract activations and compute contrast vectors for role-playing personas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    uv run roleplay/2_activations.py \\
        --personas-file prompts/personas_short.json \\
        --output-file results/activations.pt

    # Custom model and control key
    uv run roleplay/2_activations.py \\
        --personas-file prompts/personas_short.json \\
        --output-file results/gemma_activations.pt \\
        --model-name google/gemma-2-9b-it \\
        --control-key control
        """
    )
    
    # Required arguments
    parser.add_argument('--personas-file', type=str, required=True,
                       help='Path to JSON file containing personas')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Path to output .pt file for results')
    
    # Model configuration  
    parser.add_argument('--model-name', type=str, default='google/gemma-2-9b-it',
                       help='HuggingFace model name (default: google/gemma-2-9b-it)')
    parser.add_argument('--control-key', type=str, default='control',
                       help='Key for control persona in personas file (default: control)')
    
    # Optional flags
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Print configuration if verbose
    if args.verbose:
        print("Configuration:")
        print(f"  Personas file: {args.personas_file}")
        print(f"  Output file: {args.output_file}")
        print(f"  Model: {args.model_name}")
        print(f"  Control key: {args.control_key}")
        print()
    
    try:
        # Load personas file
        print("Loading personas file...")
        personas_data, persona_prompts, persona_names = load_personas(
            args.personas_file, args.control_key
        )
        print(f"Loaded {len(persona_names)} personas")
        
        if args.verbose:
            for i, name in enumerate(persona_names[:3]):
                prompt = persona_prompts[i]
                print(f"  {name}: {prompt[:50]}...")
            if len(persona_names) > 3:
                print(f"  ... and {len(persona_names) - 3} more")
        
        # Load model with multi-GPU support
        print(f"Loading model: {args.model_name}")
        pm = ProbingModel(args.model_name)
        model = pm.model
        tokenizer = pm.tokenizer
        print(f"Model loaded successfully on device: {model.device}")
        
        # Extract activations from all layers
        print("Extracting activations from all layers...")
        activations = extract_all_activations(
            model, tokenizer, persona_prompts, verbose=args.verbose
        )
        
        # Compute contrast vectors
        print("Computing contrast vectors...")
        contrast_vectors = compute_contrast_vectors(
            activations, control_idx=0, verbose=args.verbose
        )
        
        # Save results
        print("Saving results...")
        save_results(
            activations=activations,
            contrast_vectors=contrast_vectors,
            personas_data=personas_data,
            persona_names=persona_names,
            output_file=args.output_file,
            model_name=args.model_name,
            personas_file=args.personas_file,
            control_key=args.control_key,
            verbose=args.verbose
        )
        
        print("✓ Activation extraction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())