#!/usr/bin/env python3
"""
Test script to verify multi-layer support and backward compatibility
"""

import torch
import sys
sys.path.append('.')

from probing_utils import (
    load_model, 
    extract_activation_at_newline,
    extract_activations_for_prompts,
    compute_contrast_vector
)

def test_backward_compatibility():
    """Test that single layer usage still works as before"""
    print("Testing backward compatibility...")
    
    # Use a small model for testing
    model_name = "microsoft/DialoGPT-small"  # Small model for quick testing
    
    try:
        model, tokenizer = load_model(model_name)
        
        # Test single layer extraction (original behavior)
        test_prompt = "Hello, how are you?"
        
        # Single layer mode
        activation_single = extract_activation_at_newline(
            model, tokenizer, test_prompt, layer=0  # Use layer 0 for small model
        )
        print(f"✓ Single layer activation shape: {activation_single.shape}")
        
        # Test with list of prompts (original behavior)
        prompts = ["Hello", "Hi there"]
        activations_batch = extract_activations_for_prompts(
            model, tokenizer, prompts, layer=0
        )
        print(f"✓ Batch activations shape: {activations_batch.shape}")
        
        # Test contrast vector (original behavior)
        pos_acts = torch.randn(3, activation_single.shape[0])  # 3 examples
        neg_acts = torch.randn(2, activation_single.shape[0])  # 2 examples
        
        contrast, pos_mean, neg_mean = compute_contrast_vector(pos_acts, neg_acts)
        print(f"✓ Contrast vector shape: {contrast.shape}")
        
        print("✓ All backward compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

def test_multi_layer_functionality():
    """Test new multi-layer functionality"""
    print("\nTesting multi-layer functionality...")
    
    model_name = "microsoft/DialoGPT-small"
    
    try:
        model, tokenizer = load_model(model_name)
        
        # Test multi-layer extraction
        test_prompt = "Hello, how are you?"
        layers = [0, 1, 2]  # Test first 3 layers
        
        activation_dict = extract_activation_at_newline(
            model, tokenizer, test_prompt, layer=layers
        )
        
        print(f"✓ Multi-layer activation keys: {list(activation_dict.keys())}")
        for layer_idx in layers:
            print(f"  Layer {layer_idx} shape: {activation_dict[layer_idx].shape}")
        
        # Test multi-layer batch extraction
        prompts = ["Hello", "Hi there"]
        activations_multi = extract_activations_for_prompts(
            model, tokenizer, prompts, layer=layers
        )
        
        print(f"✓ Multi-layer batch keys: {list(activations_multi.keys())}")
        for layer_idx in layers:
            if activations_multi[layer_idx] is not None:
                print(f"  Layer {layer_idx} batch shape: {activations_multi[layer_idx].shape}")
        
        # Test multi-layer contrast vector
        # Create mock data for testing
        multi_pos = {}
        multi_neg = {}
        for layer_idx in layers:
            multi_pos[layer_idx] = torch.randn(3, activation_dict[layer_idx].shape[0])
            multi_neg[layer_idx] = torch.randn(2, activation_dict[layer_idx].shape[0])
        
        contrast_results = compute_contrast_vector(multi_pos, multi_neg)
        
        print(f"✓ Multi-layer contrast keys: {list(contrast_results.keys())}")
        for layer_idx in layers:
            if contrast_results[layer_idx] is not None:
                contrast_vec, _, _ = contrast_results[layer_idx]
                print(f"  Layer {layer_idx} contrast shape: {contrast_vec.shape}")
        
        print("✓ All multi-layer tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Multi-layer test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running multi-layer support tests...\n")
    
    # Run tests
    backward_ok = test_backward_compatibility()
    multi_ok = test_multi_layer_functionality()
    
    if backward_ok and multi_ok:
        print("\n🎉 All tests passed! Multi-layer support is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
        sys.exit(1)