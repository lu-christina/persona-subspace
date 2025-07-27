#!/usr/bin/env python3
"""
Batch inference script for role-playing personas using vLLM.

This script extracts the inference logic from probing/6_direct_role.ipynb,
allowing for batch processing of personas and prompts with configurable parameters.

Usage:
    uv run roleplay/1_inference.py \
        --personas-file prompts/6_direct_role/personas_short.json \
        --prompts-file prompts/6_direct_role/questions.json \
        --output-file results/inference_results.json \
        --model-name google/gemma-2-9b-it
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.inference_utils import load_vllm_model, batch_chat, close_vllm_model, cleanup_all_models


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def validate_personas_file(personas_data: Dict[str, Any]) -> None:
    """Validate the structure of the personas file."""
    if "personas" not in personas_data:
        raise ValueError("Personas file must contain a 'personas' key")
    
    for persona_name, persona_info in personas_data["personas"].items():
        if not isinstance(persona_info, dict):
            raise ValueError(f"Persona '{persona_name}' must be a dictionary")
        if "system_prompt" not in persona_info:
            raise ValueError(f"Persona '{persona_name}' must have a 'system_prompt' field")


def validate_prompts_file(prompts_data: Dict[str, Any]) -> None:
    """Validate the structure of the prompts file."""
    # For this implementation, we'll be flexible about prompts file structure
    # The notebook shows it's not directly used in the current inference loop
    pass


def extract_system_prompts(personas_data: Dict[str, Any]) -> List[str]:
    """Extract system prompts from personas data for batch inference."""
    system_prompts = []
    for persona_name in personas_data["personas"]:
        system_prompt = personas_data["personas"][persona_name]["system_prompt"]
        system_prompts.append(system_prompt)
    return system_prompts


def format_results(personas_data: Dict[str, Any], responses: List[str], 
                  model_name: str, model_config: Dict[str, Any], 
                  generation_params: Dict[str, Any]) -> Dict[str, Any]:
    """Format the results into the desired JSON structure."""
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "model_config": model_config,
        "generation_params": generation_params,
        "timestamp": datetime.now().isoformat(),
        "total_personas": len(personas_data["personas"])
    }
    
    # Create results mapping
    results = {}
    persona_names = list(personas_data["personas"].keys())
    
    for i, persona_name in enumerate(persona_names):
        if i < len(responses):
            system_prompt = personas_data["personas"][persona_name]["system_prompt"]
            response = responses[i]
            
            # Format as conversation turns
            results[persona_name] = [
                {"user": system_prompt},
                {"assistant": response}
            ]
    
    return {
        "metadata": metadata,
        "results": results
    }


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description='Run batch inference on personas and prompts using vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with system prompts only (no prompts file)
    python roleplay/1_inference.py \\
        --personas-file probing/prompts/6_direct_role/personas_short.json \\
        --output-file results/inference_results.json

    # Usage with external prompts file
    python roleplay/1_inference.py \\
        --personas-file probing/prompts/6_direct_role/personas_short.json \\
        --prompts-file probing/prompts/6_direct_role/questions.json \\
        --output-file results/inference_results.json

    # Custom model and parameters
    python roleplay/1_inference.py \\
        --personas-file probing/prompts/6_direct_role/personas_short.json \\
        --output-file results/custom_inference.json \\
        --model-name meta-llama/Llama-3.1-8B-Instruct \\
        --temperature 0.8 \\
        --max-tokens 1024
        """
    )
    
    # Required arguments
    parser.add_argument('--personas-file', type=str, required=True,
                       help='Path to JSON file containing personas (e.g., personas_short.json)')
    parser.add_argument('--prompts-file', type=str, required=False, 
                       help='Path to JSON file containing prompts/questions (e.g., questions.json). If not provided, uses system_prompt from personas file.')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Path to output JSON file for results')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='google/gemma-2-9b-it',
                       help='HuggingFace model name (default: google/gemma-2-9b-it)')
    parser.add_argument('--max-model-len', type=int, default=4096,
                       help='Maximum model context length (default: 4096)')
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                       help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization ratio (default: 0.9)')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter (default: 0.9)')
    
    # Optional flags
    parser.add_argument('--progress', action='store_true', default=True,
                       help='Show progress during batch inference (default: True)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Print configuration if verbose
    if args.verbose:
        print("Configuration:")
        print(f"  Personas file: {args.personas_file}")
        print(f"  Prompts file: {args.prompts_file or 'None (using system prompts)'}")
        print(f"  Output file: {args.output_file}")
        print(f"  Model: {args.model_name}")
        print(f"  Max model length: {args.max_model_len}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Max tokens: {args.max_tokens}")
        print()
    
    try:
        # Load input files
        print("Loading input files...")
        personas_data = load_json_file(args.personas_file)
        
        prompts_data = None
        if args.prompts_file:
            prompts_data = load_json_file(args.prompts_file)
        
        # Validate input files
        print("Validating input files...")
        validate_personas_file(personas_data)
        if prompts_data:
            validate_prompts_file(prompts_data)
        
        # Extract system prompts for batch inference
        print("Extracting system prompts...")
        system_prompts = extract_system_prompts(personas_data)
        print(f"Found {len(system_prompts)} personas for inference")
        
        if args.verbose:
            for i, prompt in enumerate(system_prompts[:3]):  # Show first 3
                print(f"  Persona {i+1}: {prompt[:100]}...")
            if len(system_prompts) > 3:
                print(f"  ... and {len(system_prompts) - 3} more")
        
        # Load vLLM model
        print(f"Loading vLLM model: {args.model_name}")
        model_wrapper = load_vllm_model(
            model_name=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        
        # Run batch inference
        print("Running batch inference...")
        responses = batch_chat(
            model_wrapper=model_wrapper,
            messages=system_prompts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            progress=args.progress
        )
        
        print(f"Generated {len(responses)} responses")
        
        # Format results
        print("Formatting results...")
        model_config = {
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization
        }
        
        generation_params = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p
        }
        
        results = format_results(
            personas_data=personas_data,
            responses=responses,
            model_name=args.model_name,
            model_config=model_config,
            generation_params=generation_params
        )
        
        # Save results
        print(f"Saving results to: {args.output_file}")
        save_results(results, args.output_file)
        
        print(" Inference completed successfully!")
        
        if args.verbose:
            print(f"\nSample results:")
            for persona_name, conversation in list(results["results"].items())[:2]:
                print(f"  {persona_name}: {conversation[1]['assistant'][:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        # Clean up model resources
        if 'model_wrapper' in locals():
            print("Cleaning up model resources...")
            close_vllm_model(model_wrapper)
        
        # Ensure all resources are cleaned up
        print("Performing final cleanup...")
        cleanup_all_models()
    
    return 0


if __name__ == "__main__":
    exit(main())