#!/usr/bin/env python3
"""
Minimum working example for persona subspace analysis.
Processes assistant_prompts.jsonl and extracts top-activating SAE features.
Includes feature interpretation and qualitative analysis.
"""

import json
import os
import torch
import h5py
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from dictionary_learning.utils import load_dictionary
from sae_utils.feature_mining import FeatureMiningConfig, FeatureMiner
from sae_utils.custom_generator import format_chat_prompt
import random

def create_persona_prompts_generator(jsonl_path, tokenizer, model_name):
    """Create generator for persona prompts from JSONL file."""
    prompts = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['content'])
    
    # Convert to chat format and repeat for more samples
    def gen():
        while True:  # Infinite generator
            for prompt in prompts:
                # Create a simple conversation format
                conversation = [
                    {"role": "user", "content": prompt}
                ]
                # Format as chat prompt
                text = format_chat_prompt(conversation, tokenizer, model_name, 
                                        remove_system_prompt=False, include_bos=False)
                yield text
    
    return gen()

def main():
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Using Qwen as specified in your CLAUDE.md
    LAYER_INDEX = 15  # Middle layer as specified
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SAE path - you'll need to update this to your actual SAE path
    # For now using a placeholder path
    SAE_PATH = "/workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_15/trainer_0"  # UPDATE THIS
    
    # Output directory
    OUTPUT_DIR = "./results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Input file
    JSONL_PATH = "./assistant_prompts.jsonl"
    
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading SAE from: {SAE_PATH}")
    try:
        sae, sae_cfg = load_dictionary(SAE_PATH, device=DEVICE)
        print(f"SAE loaded successfully. Features: {sae.decoder.weight.shape[0]}")
    except Exception as e:
        print(f"Error loading SAE: {e}")
        print("Please update SAE_PATH to point to your actual SAE files")
        return
    
    # Create data generator
    print("Creating data generator...")
    data_generator = create_persona_prompts_generator(JSONL_PATH, tokenizer, MODEL_NAME)
    
    # Feature mining configuration
    config = FeatureMiningConfig(
        model_name=MODEL_NAME,
        sae_path=SAE_PATH,
        layer_index=LAYER_INDEX,
        out_dir=OUTPUT_DIR,
        ctx_len=512,
        batch_size=4,  # Small batch size for testing
        top_k=20,      # Top 20 activating examples per feature
        num_samples=1000,  # Small number for testing
        device=DEVICE,
        dtype=torch.bfloat16
    )
    
    print("Starting feature mining...")
    miner = FeatureMiner(config)
    
    try:
        miner.run(data_generator)
        print(f"Feature mining completed! Results saved to: {OUTPUT_DIR}")
        
        # Basic analysis - show top features
        print("\nAnalyzing results...")
        analyze_results(OUTPUT_DIR)
        
    except Exception as e:
        print(f"Error during feature mining: {e}")
        print("This might be due to SAE path issues or model compatibility")

def decode_tokens_with_activations(tokenizer, tokens, sae_acts, top_n=5):
    """Decode tokens and highlight those with highest SAE activations."""
    if len(tokens.shape) > 1:
        tokens = tokens[0]  # Take first sequence if batched
        sae_acts = sae_acts[0]
    
    # Remove padding tokens
    valid_mask = tokens != tokenizer.pad_token_id
    tokens = tokens[valid_mask]
    sae_acts = sae_acts[valid_mask]
    
    # Get top activating token positions
    top_indices = np.argsort(sae_acts)[-top_n:][::-1]
    
    # Decode tokens with highlighting
    decoded_text = ""
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if i in top_indices:
            decoded_text += f"<<{token_text}>>"
        else:
            decoded_text += token_text
    
    return decoded_text, sae_acts[top_indices]

def analyze_feature_examples(h5_path, tokenizer, feature_idx, max_examples=5):
    """Analyze and display examples for a specific feature."""
    with h5py.File(h5_path, 'r') as f:
        scores = f['scores'][feature_idx]
        tokens = f['tokens'][feature_idx] 
        sae_acts = f['sae_acts'][feature_idx]
        
        # Get examples sorted by activation score
        sorted_indices = np.argsort(scores)[::-1]
        
        print(f"\nTop {max_examples} activating examples for feature {feature_idx}:")
        print("-" * 80)
        
        examples = []
        for i, idx in enumerate(sorted_indices[:max_examples]):
            if scores[idx] == 0:  # Skip zero activations
                continue
                
            decoded_text, top_activations = decode_tokens_with_activations(
                tokenizer, tokens[idx], sae_acts[idx]
            )
            
            print(f"\nExample {i+1} (Score: {scores[idx]:.3f}):")
            print(f"Text: {decoded_text}")
            print(f"Top activations: {top_activations}")
            
            examples.append({
                'text': decoded_text,
                'score': scores[idx].item(),
                'top_activations': top_activations.tolist()
            })
        
        return examples

def interpret_feature_with_claude(examples, feature_idx):
    """Generate interpretation using Claude (requires API key)."""
    try:
        import anthropic
        
        # This requires ANTHROPIC_API_KEY environment variable
        client = anthropic.Anthropic()
        
        # Prepare examples text
        examples_text = ""
        for i, ex in enumerate(examples[:5]):  # Top 5 examples
            examples_text += f"\n**Example {i+1}** (Score: {ex['score']:.3f})\n"
            examples_text += f"{ex['text']}\n"
        
        prompt = f"""You are an AI safety researcher analyzing features from a Sparse Autoencoder (SAE) trained on a language model. This feature was discovered by analyzing persona-related prompts asking about AI self-perception and internal states.

## Top Activating Examples

Below are text sequences where feature {feature_idx} activated most strongly:
{examples_text}

## Your Task

Based on these examples, provide:
1. **Feature Description**: What pattern or concept does this feature detect?
2. **Persona Relevance**: How does this relate to AI self-perception, internal states, or persona concepts?
3. **Key Observations**: What do you notice about the contexts where this feature activates?
4. **Potential Significance**: Why might this feature be important for understanding AI self-representation?

Keep your analysis concise but insightful."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    except ImportError:
        return "Claude interpretation unavailable (install anthropic package and set ANTHROPIC_API_KEY)"
    except Exception as e:
        return f"Error generating interpretation: {e}"

def simple_feature_interpretation(examples, feature_idx):
    """Simple rule-based interpretation when Claude API is not available."""
    if not examples:
        return "No examples available for interpretation"
    
    # Extract text patterns
    all_text = " ".join([ex['text'].replace("<<", "").replace(">>", "") for ex in examples])
    
    # Simple pattern detection
    patterns = []
    if any(word in all_text.lower() for word in ['i feel', 'i think', 'i am', 'my experience']):
        patterns.append("Self-referential language")
    if any(word in all_text.lower() for word in ['you', 'your', 'yourself']):
        patterns.append("Second-person references")
    if any(word in all_text.lower() for word in ['what', 'how', 'why', 'where']):
        patterns.append("Question patterns")
    if any(word in all_text.lower() for word in ['assistant', 'ai', 'model', 'system']):
        patterns.append("AI/System references")
    
    interpretation = f"""
Feature {feature_idx} Analysis:

**Detected Patterns**: {', '.join(patterns) if patterns else 'No clear patterns detected'}

**Activation Context**: This feature activates {len(examples)} times in persona-related prompts.
Average activation score: {np.mean([ex['score'] for ex in examples]):.3f}

**Key Observations**: 
- Most active on: {examples[0]['text'][:100]}...
- Pattern suggests: {'Self-perception related' if 'Self-referential' in patterns else 'Context-dependent activation'}

**Note**: This is a simple rule-based analysis. For deeper insights, consider using Claude API interpretation.
"""
    return interpretation

def analyze_results(output_dir):
    """Comprehensive analysis of the mined features with interpretation."""
    try:
        # Look for HDF5 files
        h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]
        if not h5_files:
            print("No HDF5 result files found")
            return
            
        h5_path = os.path.join(output_dir, h5_files[0])
        print(f"Found results file: {h5_files[0]}")
        
        # Load tokenizer for decoding
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Qwen2.5-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        with h5py.File(h5_path, 'r') as f:
            scores = f['scores'][:]
            freq = f['freq'][:]
            
            print(f"Total features: {len(freq)}")
            print(f"Features with activations: {(freq > 0).sum()}")
            
            # Show top features by frequency
            top_freq_idx = freq.argsort()[-10:][::-1]
            print("\nTop 10 most frequently activated features:")
            for i, idx in enumerate(top_freq_idx):
                print(f"{i+1}. Feature {idx}: {freq[idx]} activations, max score: {scores[idx].max():.3f}")
        
        # Detailed analysis of top 3 features
        print("\n" + "="*80)
        print("DETAILED FEATURE ANALYSIS")
        print("="*80)
        
        for i, feature_idx in enumerate(top_freq_idx[:3]):  # Top 3 features
            print(f"\n{'='*60}")
            print(f"FEATURE {feature_idx} - DETAILED ANALYSIS")
            print(f"{'='*60}")
            
            # Get examples
            examples = analyze_feature_examples(h5_path, tokenizer, feature_idx)
            
            # Generate interpretation
            print("\n" + "-"*40)
            print("INTERPRETATION")
            print("-"*40)
            
            # Try Claude interpretation first, fallback to simple
            interpretation = interpret_feature_with_claude(examples, feature_idx)
            if "unavailable" in interpretation or "Error" in interpretation:
                interpretation = simple_feature_interpretation(examples, feature_idx)
            
            print(interpretation)
            
            # Ask user if they want to continue
            if i < 2:  # Don't ask after the last one
                user_input = input("\nPress Enter to see next feature, or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
                    
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    main()