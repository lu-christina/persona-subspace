"""
Main pipeline for SAE misalignment feature analysis.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio

# Add paths for imports
sys.path.append('.')
sys.path.append('..')

from utils.activation_utils import extract_dataset_activations
from utils.steering_utils import ActivationSteering
from dictionary_learning.utils import load_dictionary

from .constants import DEFAULT_CONFIG, DEFAULT_SAE_PATH_TEMPLATE, DEFAULT_FEATURE_MINING_PATH_TEMPLATE, DEFAULT_CLAUDE_MODEL
from .autointerp import analyze_features_with_claude
from .steering_eval import evaluate_feature_steering, analyze_steering_results


def load_models_and_tokenizer(baseline_model_path: str, target_model_path: str) -> tuple:
    """Load baseline and target models with tokenizer."""
    print(f"🤖 Loading baseline model: {baseline_model_path}")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        baseline_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    baseline_model.eval()
    print(f"✅ Baseline model loaded on device: {baseline_model.device}")
    
    print(f"🎯 Loading target model: {target_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=torch.bfloat16, 
        device_map="cuda:1"
    )
    target_model.eval()
    print(f"✅ Target model loaded on device: {target_model.device}")
    
    print(f"🔤 Loading tokenizer: {baseline_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    return baseline_model, target_model, tokenizer


def load_sae_for_layer(layer: int, sae_path_template: str = None) -> tuple:
    """Load SAE for a specific layer."""
    if sae_path_template is None:
        sae_path_template = DEFAULT_SAE_PATH_TEMPLATE
    
    sae_path = sae_path_template.format(layer=layer)
    print(f"🧠 Loading SAE: {sae_path}")
    
    sae, _ = load_dictionary(sae_path, device='cpu')
    sae.eval()
    print(f"✅ SAE loaded (feature count: {sae.decoder.weight.shape[1]})")
    
    return sae, sae_path


def extract_feature_differences(
    baseline_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    layer: int,
    sae: Any,
    activation_strategy: str,
    max_prompts: int = 2048,
    batch_size: int = 8,
    ctx_len: int = 512
) -> tuple[torch.Tensor, float]:
    """
    Extract activation differences between baseline and target models.
    
    Args:
        baseline_model: Baseline model
        target_model: Target model (post-SFT)
        tokenizer: Tokenizer
        dataset_path: Path to dataset file
        layer: Layer to analyze
        sae: SAE model for projecting to feature space
        activation_strategy: Strategy for pooling activations
        max_prompts: Maximum number of prompts to process
        batch_size: Batch size for processing
        ctx_len: Context length
        
    Returns:
        Tuple of (feature_difference_tensor, mean_diff_norm)
    """
    print(f"📊 Extracting activations from dataset: {dataset_path}")
    print(f"   Layer: {layer}, Strategy: {activation_strategy}")
    print(f"   Max prompts: {max_prompts}, Batch size: {batch_size}")
    
    # Extract activations from baseline model
    print("🤖 Processing baseline model...")
    baseline_activations = extract_dataset_activations(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        model=baseline_model,
        layer=layer,
        variant=activation_strategy,
        batch_size=batch_size,
        n_limit=max_prompts,
        device=baseline_model.device
    )
    print(f"✅ Baseline activations: {baseline_activations.shape}")
    
    # Extract activations from target model  
    print("🎯 Processing target model...")
    target_activations = extract_dataset_activations(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        model=target_model,
        layer=layer,
        variant=activation_strategy,
        batch_size=batch_size,
        n_limit=max_prompts,
        device=target_model.device
    )
    print(f"✅ Target activations: {target_activations.shape}")
    
    # Calculate mean difference in residual space
    print("📈 Computing activation differences...")
    baseline_mean = baseline_activations.mean(dim=0)  # (d_model,)
    target_mean = target_activations.mean(dim=0)      # (d_model,)
    residual_diff = target_mean - baseline_mean       # (d_model,)
    
    # Store mean difference norm for steering magnitude scaling
    mean_diff_norm = residual_diff.norm().item()
    
    print(f"✅ Residual difference computed:")
    print(f"   Shape: {residual_diff.shape}")
    print(f"   Norm: {residual_diff.norm():.4f}")
    print(f"   📏 Mean diff norm (for steering): {mean_diff_norm:.4f}")
    
    # Project to SAE feature space
    print("🔍 Projecting to SAE feature space...")
    sae_cpu = sae.to('cpu')
    feature_diff = torch.einsum('d,df->f', residual_diff.to(torch.float32).cpu(), sae_cpu.decoder.weight.data.to(torch.float32))
    
    print(f"✅ Feature differences computed:")
    print(f"   Shape: {feature_diff.shape}")
    print(f"   Range: [{feature_diff.min():.4f}, {feature_diff.max():.4f}]")
    
    return feature_diff, mean_diff_norm


def identify_top_features(
    feature_differences: torch.Tensor,
    top_k: int,
    change_direction: str = "positive_diff"
) -> tuple:
    """
    Identify top features based on difference values.
    
    Args:
        feature_differences: Feature difference tensor
        top_k: Number of top features to return
        change_direction: "positive_diff", "negative_diff", or "absolute_diff"
        
    Returns:
        Tuple of (top_feature_ids, top_values)
    """
    if change_direction == "positive_diff":
        top_values, top_indices = torch.topk(feature_differences, k=top_k)
    elif change_direction == "negative_diff":
        top_values, top_indices = torch.topk(feature_differences, k=top_k, largest=False)
    elif change_direction == "absolute_diff":
        abs_diff = torch.abs(feature_differences)
        top_values, top_indices = torch.topk(abs_diff, k=top_k)
        # Get original values for the selected indices
        top_values = feature_differences[top_indices]
    else:
        raise ValueError(f"Unknown change_direction: {change_direction}")
    
    return top_indices, top_values


async def run_misalignment_pipeline(
    baseline_model_path: str,
    target_model_path: str,
    dataset_path: str,
    layers: Union[int, List[int]],
    activation_strategy: str,
    output_dir: str,
    output_label: str,
    top_k_features: int = 100,
    change_direction: str = "positive_diff",
    run_autointerp: bool = True,
    run_steering: bool = True,
    max_concurrent_claude: int = 8,
    steering_coefficients: List[float] = None,
    safe_threshold: float = 0.90,
    max_prompts: int = 2048,
    batch_size: int = 8,
    ctx_len: int = 512,
    sae_path_template: str = None,
    feature_mining_path_template: str = None,
    claude_model: str = None,
    cache_dir: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main pipeline for SAE misalignment feature analysis.
    
    Args:
        baseline_model_path: Path to baseline model
        target_model_path: Path to target model (post-SFT)
        dataset_path: Path to dataset file
        layers: Layer(s) to analyze (int or list of ints)
        activation_strategy: Activation extraction strategy
        output_dir: Output directory path
        output_label: Output file label
        top_k_features: Number of top features to analyze
        change_direction: Direction of change to focus on
        run_autointerp: Whether to run Claude analysis
        run_steering: Whether to run steering evaluation
        max_concurrent_claude: Max concurrent Claude requests
        steering_coefficients: Steering coefficients to test
        safe_threshold: Safe steering threshold
        max_prompts: Max prompts to process
        batch_size: Batch size
        ctx_len: Context length
        sae_path_template: SAE path template (with {layer} placeholder)
        feature_mining_path_template: Feature mining path template (with {layer} and {trainer} placeholders)
        claude_model: Claude model ID for autointerp analysis
        cache_dir: Directory for caching Claude responses (defaults to output_dir/.cache)
        **kwargs: Additional arguments
        
    Returns:
        Dict with analysis results
    """
    # Convert single layer to list
    if isinstance(layers, int):
        layers = [layers]
    
    # Set defaults
    if steering_coefficients is None:
        steering_coefficients = DEFAULT_CONFIG["steering_coefficients"]
    if claude_model is None:
        claude_model = DEFAULT_CLAUDE_MODEL
    if cache_dir is None:
        cache_dir = str(Path(output_dir) / ".cache")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    baseline_model, target_model, tokenizer = load_models_and_tokenizer(
        baseline_model_path, target_model_path
    )
    
    all_results = {}
    
    # Process each layer
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer}")
        print(f"{'='*60}")
        
        # Load SAE for this layer
        sae, sae_path = load_sae_for_layer(layer, sae_path_template)
        
        # Extract feature differences
        feature_differences, mean_diff_norm = extract_feature_differences(
            baseline_model=baseline_model,
            target_model=target_model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            layer=layer,
            sae=sae,
            activation_strategy=activation_strategy,
            max_prompts=max_prompts,
            batch_size=batch_size,
            ctx_len=ctx_len
        )
        
        # Identify top features
        print(f"🏆 Identifying top {top_k_features} features ({change_direction})...")
        top_feature_ids, top_values = identify_top_features(
            feature_differences, top_k_features, change_direction
        )
        
        print(f"✅ Top features identified:")
        print(f"   Feature IDs: {top_feature_ids[:5].tolist()}... (showing first 5)")
        print(f"   Value range: [{top_values.min():.4f}, {top_values.max():.4f}]")
        
        # Initialize layer results
        layer_results = {
            'layer': layer,
            'activation_strategy': activation_strategy,
            'top_feature_ids': top_feature_ids.tolist(),
            'top_feature_values': top_values.tolist(),
            'features': {}
        }
        
        # Run autointerp analysis
        if run_autointerp:
            print(f"\n📝 Running autointerp analysis...")
            autointerp_results = await analyze_features_with_claude(
                feature_ids=top_feature_ids.tolist(),
                layer=layer,
                sae_path=sae_path,
                max_concurrent=max_concurrent_claude,
                output_dir=cache_dir,
                feature_mining_path_template=feature_mining_path_template,
                claude_model=claude_model,
                baseline_model_path=baseline_model_path
            )
            
            # Add autointerp results to layer results
            print(f"📋 Adding autointerp results to layer data...")
            for feature_id in top_feature_ids.tolist():
                layer_results['features'][feature_id] = {
                    'feature_id': feature_id,
                    'difference_value': feature_differences[feature_id].item(),
                    'autointerp': autointerp_results.get(feature_id, {})
                }
        
        # Run steering evaluation
        if run_steering:
            print(f"\n🎮 Running steering evaluation...")
            for i, feature_id in enumerate(top_feature_ids.tolist()):
                print(f"🎯 Evaluating feature {i+1}/{len(top_feature_ids)}: {feature_id}")
                
                # Get SAE direction for this feature
                sae_direction = sae.decoder.weight.data[:, feature_id].clone()
                
                # Evaluate steering
                steering_results = evaluate_feature_steering(
                    feature_id=feature_id,
                    layer=layer,
                    steering_coefficients=steering_coefficients,
                    baseline_model=baseline_model,
                    target_model=target_model,
                    tokenizer=tokenizer,
                    sae_direction=sae_direction,
                    global_steering_magnitude=mean_diff_norm,
                    safe_threshold=safe_threshold,
                    device=str(baseline_model.device)
                )
                
                # Analyze steering results
                steering_analysis = analyze_steering_results(steering_results)
                
                # Add to results
                if feature_id not in layer_results['features']:
                    layer_results['features'][feature_id] = {
                        'feature_id': feature_id,
                        'difference_value': feature_differences[feature_id].item()
                    }
                
                layer_results['features'][feature_id]['steering'] = steering_results
                layer_results['features'][feature_id]['steering_analysis'] = steering_analysis
                
                print(f"   ✅ Feature {feature_id} steering complete")
        
        # Save layer results
        output_filename = f"{output_label}_layer{layer}_{activation_strategy}.json"
        output_path = output_dir / output_filename
        
        print(f"\n💾 Saving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(layer_results, f, indent=2)
        print(f"✅ Layer {layer} results saved")
        
        all_results[f"layer_{layer}"] = layer_results
    
    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return all_results 