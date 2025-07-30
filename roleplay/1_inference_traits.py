#!/usr/bin/env python3
"""
Batch inference script for role-playing personas using vLLM.

This script extracts the inference logic from probing/6_direct_role.ipynb,
allowing for batch processing of personas and prompts with configurable parameters.

Usage:
    uv run roleplay/1_inference_traits.py \
        --traits-file prompts/6_direct_role/personas_short.json \
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

def validate_prompts_file(prompts_data: Dict[str, Any]) -> None:
    """Validate the structure of the prompts file."""
    # For this implementation, we'll be flexible about prompts file structure
    # The notebook shows it's not directly used in the current inference loop
    pass

def generate_prompts(traits_data: Dict[str, Any], format_data: Dict[str, Any]) -> tuple[List[str], List[Dict[str, Any]]]:
    """Generate prompts from traits data and return messages + metadata."""
    messages = []
    metadata = []
    
    sample_id = 0
    for trait_name in traits_data.keys():
        for format_id, format_template in enumerate(format_data["format"]):
            formatted_prompt = format_template.format(trait=trait_name)
            messages.append(formatted_prompt)
            
            metadata.append({
                "id": sample_id,
                "trait_name": trait_name,
                "format_id": format_id,
                "prompt": formatted_prompt
            })
            sample_id += 1
    
    return messages, metadata


def format_results(sample_metadata: List[Dict[str, Any]], responses: List[str], 
                  model_name: str, model_config: Dict[str, Any], 
                  generation_params: Dict[str, Any]) -> Dict[str, Any]:
    """Format the results into the desired JSON structure."""
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "model_config": model_config,
        "generation_params": generation_params,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(sample_metadata)
    }
    
    # Create samples array
    samples = []
    for i, sample_meta in enumerate(sample_metadata):
        if i < len(responses):
            sample = {
                "id": sample_meta["id"],
                "trait_name": sample_meta["trait_name"],
                "format_id": sample_meta["format_id"],
                "prompt": sample_meta["prompt"],
                "response": responses[i],
                "response_type": "roleplay"
            }
            samples.append(sample)
    
    return {
        "metadata": metadata,
        "samples": samples
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
    python roleplay/1_inference_traits.py \\
        --personas-file probing/prompts/6_direct_role/personas_short.json \\
        --output-file results/inference_results.json

    # Usage with external prompts file
    python roleplay/1_inference_traits.py \\
        --personas-file probing/prompts/6_direct_role/personas_short.json \\
        --prompts-file probing/prompts/6_direct_role/questions.json \\
        --output-file results/inference_results.json

    # Custom model and parameters
    python roleplay/1_inference_traits.py \\
        --personas-file probing/prompts/6_direct_role/personas_short.json \\
        --output-file results/custom_inference.json \\
        --model-name meta-llama/Llama-3.1-8B-Instruct \\
        --temperature 0.8 \\
        --max-tokens 1024
        """
    )
    
    # Required arguments
    parser.add_argument('--traits-file', type=str, required=True,
                       help='Path to JSON file containing traits (e.g., traits.json)')
    parser.add_argument('--format-file', type=str, required=True,
                       help='Path to JSON file containing traits format (e.g., traits_format.json)')
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
        print(f"  Traits file: {args.traits_file}")
        print(f"  Format file: {args.format_file}")
        print(f"  Output file: {args.output_file}")
        print(f"  Model: {args.model_name}")
        print(f"  Max model length: {args.max_model_len}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Max tokens: {args.max_tokens}")
        print()
    
    try:
        # Load input files
        print("Loading input files...")
        traits_data = load_json_file(args.traits_file)
        format_data = load_json_file(args.format_file)
        
        # Load vLLM model
        print(f"Loading vLLM model: {args.model_name}")
        model_wrapper = load_vllm_model(
            model_name=args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )

        # Create prompts from format and traits
        messages, sample_metadata = generate_prompts(traits_data, format_data)
        
        # Run batch inference
        print("Running batch inference...")
        responses = batch_chat(
            model_wrapper=model_wrapper,
            messages=messages,
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
            sample_metadata=sample_metadata,
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
            print("\nSample results:")
            for sample in results["samples"][:2]:
                print(f"  {sample['trait_name']} (format {sample['format_id']}): {sample['response'][:100]}...")
        
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