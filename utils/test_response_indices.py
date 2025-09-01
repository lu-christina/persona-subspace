#!/usr/bin/env python3
"""
Standalone test functions for model-specific response token extraction.

Usage:
    from test_response_indices import test_gemma, test_qwen
    
    test_gemma()  # Test Gemma-2-27b-it
    test_qwen()   # Test Qwen3-32B
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

from transformers import AutoTokenizer
from probing_utils import get_response_indices


def get_gemma_test_conversation():
    """Get test conversation for Gemma (no system prompt, simple 2-turn)."""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "3+3 equals 6."}
    ]


def get_qwen_test_conversation_no_system():
    """Get test conversation for Qwen (no system prompt, 2-turn)."""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "3+3 equals 6."}
    ]


def get_qwen_test_conversation_with_system():
    """Get test conversation for Qwen (with system prompt, 2-turn)."""
    return [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "3+3 equals 6."}
    ]


def get_qwen_long_test_conversation():
    """Get longer test conversation for Qwen with multi-line responses."""
    return [
        {"role": "system", "content": "You are a helpful assistant that explains concepts clearly."},
        {"role": "user", "content": "Can you explain what photosynthesis is?"},
        {"role": "assistant", "content": """Photosynthesis is a vital biological process that occurs in plants, algae, and some bacteria.

Here's how it works:
- Plants absorb sunlight through their leaves
- They take in carbon dioxide from the air
- Water is absorbed through the roots
- These combine to produce glucose and oxygen

The basic equation is:
6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂

This process is essential for life on Earth!"""},
        {"role": "user", "content": "What about cellular respiration?"},
        {"role": "assistant", "content": """Cellular respiration is essentially the opposite of photosynthesis.

Key points:
- It breaks down glucose to release energy
- Occurs in the mitochondria of cells
- Uses oxygen and produces carbon dioxide
- Releases ATP (energy currency of cells)

The equation is:
C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP

Both processes work together in the carbon-oxygen cycle."""}
    ]


def get_qwen_empty_system_test_conversation():
    """Get test conversation for Qwen with empty system prompt."""
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "3+3 equals 6."}
    ]


def get_llama_test_conversation_no_system():
    """Get test conversation for Llama (no system prompt)."""
    return [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."}
    ]


def get_llama_test_conversation_with_system():
    """Get test conversation for Llama (with system prompt)."""
    return [
        {"role": "system", "content": "You are a helpful geography assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."}
    ]


def get_llama_empty_system_test_conversation():
    """Get test conversation for Llama with empty system prompt."""
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."}
    ]


def get_llama_long_test_conversation():
    """Get longer test conversation for Llama with multi-line responses."""
    return [
        {"role": "system", "content": "You are an expert in composing functions. You provide clear explanations."},
        {"role": "user", "content": "Can you explain how function composition works in mathematics?"},
        {"role": "assistant", "content": """Function composition is a fundamental concept in mathematics where you combine two or more functions to create a new function.

Here's how it works:
- Given functions f(x) and g(x), the composition (f ∘ g)(x) = f(g(x))
- You apply the inner function first, then the outer function
- The output of g becomes the input to f

Example:
If f(x) = 2x and g(x) = x + 3, then:
(f ∘ g)(x) = f(g(x)) = f(x + 3) = 2(x + 3) = 2x + 6

This creates powerful mathematical tools for building complex functions!"""},
        {"role": "user", "content": "What are some practical applications?"},
        {"role": "assistant", "content": """Function composition has many practical applications:

Programming:
- Chain operations together (map, filter, reduce)
- Build complex transformations from simple ones
- Create pipelines for data processing

Mathematics:
- Calculus: Chain rule is based on function composition
- Algebra: Solving nested equations
- Geometry: Transformations (rotation ∘ translation)

Real-world examples:
- Converting units: celsius_to_kelvin(fahrenheit_to_celsius(temp))
- Image processing: blur(sharpen(resize(image)))
- Data analysis: analyze(clean(load(data)))

It's a powerful way to break complex problems into manageable pieces!"""}
    ]


def validate_extraction(extracted_text, expected_text, model_name):
    """Validate extracted text against expected text."""
    # Clean up whitespace for comparison
    extracted_clean = extracted_text.strip()
    expected_clean = expected_text.strip()
    
    print(f"\nEXTRACTED RESPONSE TEXT ({model_name}):")
    print("-" * 40)
    print(repr(extracted_text))
    print("-" * 40)
    
    print(f"\nEXPECTED RESPONSE TEXT ({model_name}):")
    print("-" * 40)
    print(repr(expected_text))
    print("-" * 40)
    
    # Check if extraction contains expected content
    if expected_clean in extracted_clean or extracted_clean in expected_clean:
        print("✅ PASS: Extracted text contains expected content")
        return True
    else:
        print("❌ FAIL: Extracted text doesn't match expected content")
        print(f"Expected: {repr(expected_clean)}")
        print(f"Extracted: {repr(extracted_clean)}")
        return False


def test_gemma():
    """Test Gemma-2-27b-it model response extraction (no system prompt, no thinking)."""
    model_name = "google/gemma-2-27b-it"
    
    print("=" * 60)
    print(f"TESTING {model_name}")
    print("=" * 60)
    
    try:
        # Load tokenizer
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get test conversation (no system prompt)
        conversation = get_gemma_test_conversation()
        
        print(f"\nTest conversation ({len(conversation)} turns):")
        for i, turn in enumerate(conversation):
            role = turn.get('role', 'unknown')
            content_preview = turn.get('content', '')[:40] + ('...' if len(turn.get('content', '')) > 40 else '')
            print(f"  Turn {i}: {role} - {content_preview}")
        
        # Extract response indices (no thinking for Gemma)
        print("\nExtracting response indices...")
        response_indices = get_response_indices(conversation, tokenizer, model_name)
        
        if not response_indices:
            print("❌ FAIL: No response indices found!")
            return False
        
        print(f"Found {len(response_indices)} response token indices")
        
        # Get full conversation tokens for verification
        full_formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        full_tokens = tokenizer(full_formatted, add_special_tokens=False)
        all_token_ids = full_tokens['input_ids']
        
        print(f"Full conversation has {len(all_token_ids)} tokens")
        print(f"Response tokens span indices: {min(response_indices)}-{max(response_indices)}")
        
        # Extract and decode the response tokens
        response_token_ids = [all_token_ids[i] for i in response_indices if i < len(all_token_ids)]
        response_text = tokenizer.decode(response_token_ids)
        
        # Expected response text (ground truth)
        expected_responses = []
        for turn in conversation:
            if turn['role'] == 'assistant':
                expected_responses.append(turn['content'])
        expected_text = "".join(expected_responses)
        
        # Validate extraction
        return validate_extraction(response_text, expected_text, model_name)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_qwen_single(conversation, test_name, model_name):
    """Test a single Qwen conversation."""
    print(f"\n--- {test_name} ---")
    
    try:
        # Load tokenizer
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"\nTest conversation ({len(conversation)} turns):")
        for i, turn in enumerate(conversation):
            role = turn.get('role', 'unknown')
            content_preview = turn.get('content', '')[:40] + ('...' if len(turn.get('content', '')) > 40 else '')
            print(f"  Turn {i}: {role} - {content_preview}")
        
        # Extract response indices (no thinking for Qwen)
        print("\nExtracting response indices...")
        chat_kwargs = {'enable_thinking': False}
        response_indices = get_response_indices(conversation, tokenizer, model_name, **chat_kwargs)
        
        if not response_indices:
            print("❌ FAIL: No response indices found!")
            return False
        
        print(f"Found {len(response_indices)} response token indices")
        
        # Get full conversation tokens for verification
        full_formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
        )
        full_tokens = tokenizer(full_formatted, add_special_tokens=False)
        all_token_ids = full_tokens['input_ids']
        
        print(f"Full conversation has {len(all_token_ids)} tokens")
        print(f"Response tokens span indices: {min(response_indices)}-{max(response_indices)}")
        
        # Extract and decode the response tokens
        response_token_ids = [all_token_ids[i] for i in response_indices if i < len(all_token_ids)]
        response_text = tokenizer.decode(response_token_ids)
        
        # Expected response text (ground truth)
        expected_responses = []
        for turn in conversation:
            if turn['role'] == 'assistant':
                expected_responses.append(turn['content'])
        expected_text = "".join(expected_responses)
        
        # Validate extraction
        return validate_extraction(response_text, expected_text, f"{model_name} ({test_name})")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_qwen():
    """Test Qwen3-32B model response extraction (multiple scenarios, no thinking)."""
    model_name = "Qwen/Qwen3-32B"
    
    print("=" * 60)
    print(f"TESTING {model_name}")
    print("=" * 60)
    
    # Test without system prompt
    conv_no_system = get_qwen_test_conversation_no_system()
    result1 = test_qwen_single(conv_no_system, "NO SYSTEM PROMPT", model_name)
    
    # Test with system prompt (short responses)
    conv_with_system = get_qwen_test_conversation_with_system()
    result2 = test_qwen_single(conv_with_system, "WITH SYSTEM PROMPT (SHORT)", model_name)
    
    # Test with empty system prompt
    conv_empty_system = get_qwen_empty_system_test_conversation()
    result3 = test_qwen_single(conv_empty_system, "EMPTY SYSTEM PROMPT", model_name)
    
    # Test with longer multi-line responses
    conv_long = get_qwen_long_test_conversation()
    result4 = test_qwen_single(conv_long, "LONG MULTI-LINE RESPONSES", model_name)
    
    # All tests must pass
    return result1 and result2 and result3 and result4


def test_llama_single(conversation, test_name, model_name):
    """Test a single Llama conversation."""
    print(f"\n--- {test_name} ---")
    
    try:
        # Load tokenizer
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"\nTest conversation ({len(conversation)} turns):")
        for i, turn in enumerate(conversation):
            role = turn.get('role', 'unknown')
            content_preview = turn.get('content', '')[:40] + ('...' if len(turn.get('content', '')) > 40 else '')
            print(f"  Turn {i}: {role} - {content_preview}")
        
        # Extract response indices
        print("\nExtracting response indices...")
        response_indices = get_response_indices(conversation, tokenizer, model_name)
        
        if not response_indices:
            print("❌ FAIL: No response indices found!")
            return False
        
        print(f"Found {len(response_indices)} response token indices")
        
        # Get full conversation tokens for verification
        full_formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        full_tokens = tokenizer(full_formatted, add_special_tokens=False)
        all_token_ids = full_tokens['input_ids']
        
        print(f"Full conversation has {len(all_token_ids)} tokens")
        print(f"Response tokens span indices: {min(response_indices)}-{max(response_indices)}")
        
        # Extract and decode the response tokens
        response_token_ids = [all_token_ids[i] for i in response_indices if i < len(all_token_ids)]
        response_text = tokenizer.decode(response_token_ids)
        
        # Expected response text (ground truth)
        expected_responses = []
        for turn in conversation:
            if turn['role'] == 'assistant':
                expected_responses.append(turn['content'])
        expected_text = "".join(expected_responses)
        
        # Validate extraction
        return validate_extraction(response_text, expected_text, f"{model_name} ({test_name})")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_llama():
    """Test Meta-Llama-3.3-70B-Instruct model response extraction (multiple scenarios)."""
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    
    print("=" * 60)
    print(f"TESTING {model_name}")
    print("=" * 60)
    
    # Test without system prompt
    conv_no_system = get_llama_test_conversation_no_system()
    result1 = test_llama_single(conv_no_system, "NO SYSTEM PROMPT", model_name)
    
    # Test with system prompt
    conv_with_system = get_llama_test_conversation_with_system()
    result2 = test_llama_single(conv_with_system, "WITH SYSTEM PROMPT", model_name)
    
    # Test with empty system prompt
    conv_empty_system = get_llama_empty_system_test_conversation()
    result3 = test_llama_single(conv_empty_system, "EMPTY SYSTEM PROMPT", model_name)
    
    # Test with longer multi-line responses
    conv_long = get_llama_long_test_conversation()
    result4 = test_llama_single(conv_long, "LONG MULTI-LINE RESPONSES", model_name)
    
    # All tests must pass
    return result1 and result2 and result3 and result4


def run_all_tests():
    """Run all available model tests."""
    print("🧪 Running all model-specific response extraction tests...\n")
    
    results = {}
    
    # Test Gemma
    results['gemma'] = test_gemma()
    print()
    
    # Test Qwen  
    results['qwen'] = test_qwen()
    print()
    
    # Test Llama (placeholder)
    results['llama'] = test_llama()
    print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for model, result in results.items():
        if result is True:
            print(f"✅ {model.upper()}: PASS")
        elif result is False:
            print(f"❌ {model.upper()}: FAIL") 
        else:
            print(f"🚧 {model.upper()}: NOT IMPLEMENTED")
    
    return results


if __name__ == "__main__":
    # If run as script, run all tests
    run_all_tests()