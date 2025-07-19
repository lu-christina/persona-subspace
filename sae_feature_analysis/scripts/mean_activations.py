#!/usr/bin/env python3
"""
Extract Feature Mean Activations Script

This script extracts feature directions from an SAE, loads prompts from LMSYS,
gets activations for all prompts and tokens, projects into feature directions,
and saves mean activations for later mean ablation experiments.

Based on the 4_diffing.py pipeline.
"""

import argparse
import torch
import os
import pandas as pd
from typing import List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from tqdm.auto import tqdm
from dataclasses import dataclass

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# MODEL CONFIGURATIONS (copied from 4_diffing.py)
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model-specific settings"""
    base_model_name: str
    chat_model_name: str
    hf_release: str
    assistant_header: str
    token_offsets: Dict[str, int]
    sae_base_path: str
    saelens_release: str
    sae_id_template: str
    
    def get_sae_params(self, sae_layer: int, sae_trainer: str) -> Tuple[str, str]:
        """Generate SAE lens release and sae_id parameters."""
        if self.saelens_release == "llama_scope_lxr_{trainer}":
            release = self.saelens_release.format(trainer=sae_trainer)
            sae_id = self.sae_id_template.format(layer=sae_layer, trainer=sae_trainer)
        elif self.saelens_release == "gemma-scope-9b-pt-res":
            parts = sae_trainer.split("-")
            width = parts[0]
            l0_value = parts[2]
            release = self.saelens_release
            sae_id = self.sae_id_template.format(layer=sae_layer, width=width, l0=l0_value)
        elif self.saelens_release == "gemma-scope-9b-pt-res-canonical":
            parts = sae_trainer.split("-")
            width = parts[0]
            release = self.saelens_release
            sae_id = self.sae_id_template.format(layer=sae_layer, width=width)
        else:
            raise ValueError(f"Unknown SAE lens release template: {self.saelens_release}")
        
        return release, sae_id

# Model configurations
MODEL_CONFIGS = {
    "llama": ModelConfig(
        base_model_name="meta-llama/Llama-3.1-8B",
        chat_model_name="meta-llama/Llama-3.1-8B-Instruct",
        hf_release="fnlp/Llama3_1-8B-Base-LXR-32x",
        assistant_header="<|start_header_id|>assistant<|end_header_id|>",
        token_offsets={"asst": -2, "endheader": -1, "newline": 0},
        sae_base_path="/workspace/sae/llama-3.1-8b/saes",
        saelens_release="llama_scope_lxr_{trainer}",
        sae_id_template="l{layer}r_{trainer}"
    ),
    "gemma": ModelConfig(
        base_model_name="google/gemma-2-9b",
        chat_model_name="google/gemma-2-9b-it",
        hf_release="google/gemma-scope-9b-pt-res/layer_{layer}/width_{width}/average_l0_{l0}",
        assistant_header="<start_of_turn>model",
        token_offsets={"model": -1, "newline": 0},
        sae_base_path="/workspace/sae/gemma-2-9b/saes",
        saelens_release="gemma-scope-9b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_{width}/canonical"
    )
}

# =============================================================================
# UTILITY FUNCTIONS (copied from 4_diffing.py)
# =============================================================================

def load_lmsys_prompts(prompts_path: str, prompts_hf: str, n_prompts: int, seed: int) -> pd.DataFrame:
    """Load LMSYS prompts, either from cache or download fresh."""
    if os.path.exists(prompts_path):
        print(f"Prompts already exist at {prompts_path}")
        return pd.read_json(prompts_path, lines=True)
    else:
        print(f"Prompts do not exist at {prompts_path}. Loading from {prompts_hf}...")
        dataset = load_dataset(prompts_hf)
        dataset = dataset['train'].shuffle(seed=seed).select(range(n_prompts))
        df = dataset.to_pandas()
        
        # Extract the prompt from the first conversation item
        df['prompt'] = df['conversation'].apply(lambda x: x[0]['content'])
        
        # Only keep some columns
        df = df[['conversation_id', 'prompt', 'redacted', 'language']]
        
        # Save to .jsonl file
        os.makedirs(os.path.dirname(prompts_path), exist_ok=True)
        df.to_json(prompts_path, orient='records', lines=True)
        return df

def load_sae(config: ModelConfig, sae_path: str, sae_layer: int, sae_trainer: str) -> SAE:
    """Load SAE from disk or download from HuggingFace."""
    ae_file_path = os.path.join(sae_path, "sae_weights.safetensors")
    
    if os.path.exists(ae_file_path):
        print(f"✓ Found SAE files at: {os.path.dirname(ae_file_path)}")
        sae = SAE.load_from_disk(sae_path)
        return sae
    
    print(f"SAE not found locally, downloading from HF via sae_lens...")
    os.makedirs(os.path.dirname(sae_path), exist_ok=True)
    
    # Get SAE parameters from config
    release, sae_id = config.get_sae_params(sae_layer, sae_trainer)
    print(f"Loading SAE with release='{release}', sae_id='{sae_id}'")
    
    # Load the SAE using sae_lens
    sae, _, _ = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device="cuda"
    )
    
    # Save the SAE locally for future use
    sae.save_model(sae_path)
    return sae

# =============================================================================
# ACTIVATION EXTRACTION FUNCTIONS
# =============================================================================

class StopForward(Exception):
    """Exception to stop forward pass after target layer."""
    pass

@torch.no_grad()
def extract_and_project_activations_chunked(prompts: List[str], layer_idx: int, model, tokenizer,
                                           sae, feature_ids: List[int], feature_chunk_size: int = 4096,
                                           batch_size: int = 32, max_length: int = 512) -> Tuple[torch.Tensor, int]:
    """
    Extract activations and project onto feature directions with chunked processing.
    More memory efficient for processing all features in large SAEs.
    
    Args:
        prompts: List of prompt strings
        layer_idx: Layer index to extract activations from
        model: The transformer model
        tokenizer: Tokenizer for the model
        sae: SAE model to get decoder weights from
        feature_ids: List of feature IDs to process
        feature_chunk_size: Number of features to process in each chunk
        batch_size: Batch size for processing
        max_length: Maximum token length
        
    Returns:
        Tuple of (mean_activations [num_features], total_tokens_processed)
    """
    target_layer = model.model.layers[layer_idx]
    
    # Running statistics
    running_sum = torch.zeros(len(feature_ids), device=device, dtype=torch.float32)
    total_tokens = 0
    
    print(f"Processing {len(feature_ids)} features in chunks of {feature_chunk_size}")
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing prompts"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Format prompts as chat messages
        formatted_prompts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize batch
        batch_inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        
        # Hook to capture activations and process in chunks
        def hook_fn(module, input, output):
            nonlocal running_sum, total_tokens
            activations = output[0] if isinstance(output, tuple) else output
            # activations shape: [batch_size, seq_len, hidden_dim]
            
            # Collect all valid activations from this batch
            batch_activations = []
            batch_token_count = 0
            
            for j in range(activations.shape[0]):
                attention_mask = batch_inputs["attention_mask"][j]
                valid_length = attention_mask.sum().item()
                
                # Extract activations for valid tokens only
                valid_activations = activations[j, :valid_length, :].float()  # [valid_seq_len, hidden_dim]
                batch_activations.append(valid_activations)
                batch_token_count += valid_length
            
            # Concatenate all valid activations in this batch
            if batch_activations:
                all_batch_activations = torch.cat(batch_activations, dim=0)  # [total_batch_tokens, hidden_dim]
                
                # Process features in chunks to avoid large intermediate tensors
                for chunk_start in range(0, len(feature_ids), feature_chunk_size):
                    chunk_end = min(chunk_start + feature_chunk_size, len(feature_ids))
                    chunk_feature_ids = feature_ids[chunk_start:chunk_end]
                    
                    # Get feature directions for this chunk
                    chunk_directions = sae.W_dec[chunk_feature_ids, :].T.to(device).float()  # [hidden_dim, chunk_size]
                    
                    # Project onto chunk directions
                    chunk_projections = torch.mm(all_batch_activations, chunk_directions)  # [total_batch_tokens, chunk_size]
                    
                    # Add to running sum
                    running_sum[chunk_start:chunk_end] += chunk_projections.sum(dim=0)
                
                total_tokens += batch_token_count
            
            raise StopForward()
        
        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            _ = model(**batch_inputs)
        except StopForward:
            pass
        finally:
            handle.remove()
    
    # Compute final mean
    mean_activations = running_sum / total_tokens
    return mean_activations.cpu(), total_tokens

@torch.no_grad()
def extract_and_project_activations_direct(prompts: List[str], layer_idx: int, model, tokenizer,
                                          feature_directions: torch.Tensor,
                                          batch_size: int = 32, max_length: int = 512) -> Tuple[torch.Tensor, int]:
    """
    Extract activations and project onto feature directions directly, computing running mean.
    Much more memory efficient than storing all activations first.
    
    Args:
        prompts: List of prompt strings
        layer_idx: Layer index to extract activations from
        model: The transformer model
        tokenizer: Tokenizer for the model
        feature_directions: Feature directions tensor [hidden_dim, num_features]
        batch_size: Batch size for processing
        max_length: Maximum token length
        
    Returns:
        Tuple of (mean_activations [num_features], total_tokens_processed)
    """
    target_layer = model.model.layers[layer_idx]
    feature_directions = feature_directions.to(device)
    
    # Running statistics
    running_sum = torch.zeros(feature_directions.shape[1], device=device, dtype=torch.float32)
    total_tokens = 0
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting and projecting activations"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Format prompts as chat messages
        formatted_prompts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize batch
        batch_inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        
        # Hook to capture activations and project immediately
        def hook_fn(module, input, output):
            nonlocal running_sum, total_tokens
            activations = output[0] if isinstance(output, tuple) else output
            # activations shape: [batch_size, seq_len, hidden_dim]
            
            # Process each sequence in the batch
            for j in range(activations.shape[0]):
                attention_mask = batch_inputs["attention_mask"][j]
                valid_length = attention_mask.sum().item()
                
                # Extract activations for valid tokens only
                valid_activations = activations[j, :valid_length, :]  # [valid_seq_len, hidden_dim]
                
                # Convert to float32 to match feature_directions dtype
                valid_activations = valid_activations.float()
                
                # Project onto feature directions immediately
                # [valid_seq_len, hidden_dim] @ [hidden_dim, num_features] = [valid_seq_len, num_features]
                projections = torch.mm(valid_activations, feature_directions)
                
                # Add to running sum (sum over tokens in this sequence)
                running_sum += projections.sum(dim=0)  # [num_features]
                total_tokens += valid_length
            
            raise StopForward()
        
        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            _ = model(**batch_inputs)
        except StopForward:
            pass
        finally:
            handle.remove()
    
    # Compute final mean
    mean_activations = running_sum / total_tokens
    return mean_activations.cpu(), total_tokens

# =============================================================================
# FEATURE PROCESSING FUNCTIONS
# =============================================================================

def load_feature_ids_from_csv(csv_path: str) -> List[int]:
    """Load feature IDs from CSV file. Expects a 'feature_id' column."""
    print(f"Loading feature IDs from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'feature_id' not in df.columns:
        raise ValueError(f"CSV must have 'feature_id' column. Found columns: {df.columns.tolist()}")
    
    feature_ids = df['feature_id'].unique().tolist()
    print(f"Loaded {len(feature_ids)} unique feature IDs")
    return feature_ids

def load_existing_results(output_path: str) -> Tuple[Dict, List[int]]:
    """
    Load existing results file if it exists.
    
    Returns:
        Tuple of (existing_results_dict, existing_feature_ids)
    """
    if not os.path.exists(output_path):
        print("No existing results file found")
        return None, []
    
    print(f"Loading existing results from {output_path}")
    existing_results = torch.load(output_path, map_location='cpu')
    
    if 'feature_ids' not in existing_results:
        print("Warning: existing results missing feature_ids, treating as empty")
        return existing_results, []
    
    existing_feature_ids = existing_results['feature_ids']
    print(f"Found {len(existing_feature_ids)} features in existing results")
    return existing_results, existing_feature_ids

def merge_results(existing_results: Dict, new_results: Dict, 
                 existing_feature_ids: List[int], new_feature_ids: List[int]) -> Dict:
    """
    Merge existing and new results.
    
    Args:
        existing_results: Previously saved results dict
        new_results: Newly computed results dict  
        existing_feature_ids: Feature IDs from existing results
        new_feature_ids: Feature IDs from new computation
        
    Returns:
        Dict: Merged results
    """
    print("Merging existing and new results")
    
    # Combine feature IDs (existing first, then new)
    all_feature_ids = existing_feature_ids + new_feature_ids
    
    # Combine mean activations
    if existing_results is not None and 'mean_activations' in existing_results:
        existing_activations = existing_results['mean_activations']
        new_activations = new_results['mean_activations']
        
        # Concatenate: [existing_features] + [new_features]
        combined_activations = torch.cat([existing_activations, new_activations], dim=0)
    else:
        # No existing results, use new results only
        combined_activations = new_results['mean_activations']
    
    # Update metadata
    merged_metadata = new_results['metadata'].copy()
    merged_metadata['num_features'] = len(all_feature_ids)
    
    merged_results = {
        'mean_activations': combined_activations,
        'feature_ids': all_feature_ids,
        'metadata': merged_metadata
    }
    
    print(f"Merged results: {len(all_feature_ids)} total features")
    return merged_results

@torch.no_grad()
def extract_feature_directions(sae, feature_ids: List[int]) -> torch.Tensor:
    """
    Extract decoder weight vectors (feature directions) for specified features.
    
    Args:
        sae: SAE model
        feature_ids: List of feature IDs to extract
        
    Returns:
        torch.Tensor: Feature directions of shape [hidden_dim, num_features]
    """
    print(f"Extracting feature directions for {len(feature_ids)} features")
    
    # Get the decoder weights: [num_features, hidden_dim]
    decoder_weights = sae.W_dec.detach()  # [num_sae_features, hidden_dim]
    
    # Extract the specific feature directions we want
    selected_directions = decoder_weights[feature_ids, :]  # [num_target_features, hidden_dim]
    
    # Transpose to get [hidden_dim, num_features] for easier matrix multiplication
    feature_directions = selected_directions.T  # [hidden_dim, num_target_features]
    
    print(f"Feature directions shape: {feature_directions.shape}")
    return feature_directions


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract feature mean activations for mean ablation')
    parser.add_argument('--feature_csv', type=str, required=False,
                       help='Path to CSV file containing feature_id column')
    parser.add_argument('--all_features', action='store_true',
                       help='Process all features in the SAE (ignores --feature_csv)')
    parser.add_argument('--model_type', type=str, default='gemma', choices=['gemma', 'llama'],
                       help='Model type to use')
    parser.add_argument('--target_layer', type=int, required=True,
                       help='Layer number for SAE')
    parser.add_argument('--sae_trainer', type=str, required=True,
                       help='SAE trainer identifier (e.g., "131k-l0-114" for Gemma)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the output .pt file')
    parser.add_argument('--n_prompts', type=int, default=10000,
                       help='Number of LMSYS prompts to use')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum token length for prompts')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for prompt selection')
    parser.add_argument('--feature_chunk_size', type=int, default=16384,
                       help='Process features in chunks of this size (for memory efficiency)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_features and not args.feature_csv:
        parser.error("Must specify either --feature_csv or --all_features")
    
    print(f"{'='*60}")
    print("FEATURE MEAN ACTIVATION EXTRACTION")
    print(f"{'='*60}")
    print(f"Model type: {args.model_type}")
    print(f"Target layer: {args.target_layer}")
    print(f"SAE trainer: {args.sae_trainer}")
    print(f"Mode: {'All features' if args.all_features else 'CSV features'}")
    if not args.all_features:
        print(f"Feature CSV: {args.feature_csv}")
    if args.all_features:
        print(f"Feature chunk size: {args.feature_chunk_size}")
    print(f"Output path: {args.output_path}")
    print(f"Number of prompts: {args.n_prompts}")
    
    # Get model config
    if args.model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    config = MODEL_CONFIGS[args.model_type]
    
    # Set up paths
    prompts_hf = "lmsys/lmsys-chat-1m"
    prompts_path = f"/workspace/data/{prompts_hf.split('/')[-1]}/chat_{args.n_prompts}.jsonl"
    sae_path = f"{config.sae_base_path}/resid_post_layer_{args.target_layer}/trainer_{args.sae_trainer}"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Step 1: Load or generate feature IDs and check existing results
    print(f"\n{'='*60}")
    print("STEP 1: Loading feature IDs and checking existing results")
    print(f"{'='*60}")
    
    if args.all_features:
        # We'll get the number of features after loading the SAE
        requested_feature_ids = None
    else:
        requested_feature_ids = load_feature_ids_from_csv(args.feature_csv)
    
    # Step 2: Load SAE first (needed to get total features for all_features mode)
    print(f"\n{'='*60}")
    print("STEP 2: Loading SAE")
    print(f"{'='*60}")
    sae = load_sae(config, sae_path, args.target_layer, args.sae_trainer)
    sae = sae.to(device)
    print(f"✓ SAE loaded with {sae.cfg.d_sae} features")
    
    # Now handle feature ID logic
    if args.all_features:
        requested_feature_ids = list(range(sae.cfg.d_sae))
        print(f"Processing all {len(requested_feature_ids)} features in SAE")
    
    # Load existing results if they exist
    existing_results, existing_feature_ids = load_existing_results(args.output_path)
    
    # Filter out features that already exist
    new_feature_ids = [fid for fid in requested_feature_ids if fid not in existing_feature_ids]
    
    if len(new_feature_ids) == 0:
        print("✓ All requested features already exist in output file!")
        print("Nothing to do. Exiting.")
        return
    
    print(f"Features already processed: {len(existing_feature_ids)}")
    print(f"New features to process: {len(new_feature_ids)}")
    feature_ids = new_feature_ids  # Only process the new ones
    
    # Step 3: Load model and tokenizer
    print(f"\n{'='*60}")
    print("STEP 3: Loading model and tokenizer")
    print(f"{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        config.chat_model_name,  # Use chat model
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    model.eval()
    print(f"✓ Model loaded: {model.__class__.__name__}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.chat_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Step 4: Load prompts
    print(f"\n{'='*60}")
    print("STEP 4: Loading prompts")
    print(f"{'='*60}")
    prompts_df = load_lmsys_prompts(prompts_path, prompts_hf, args.n_prompts, args.seed)
    print(f"✓ Loaded {len(prompts_df)} prompts")
    
    # Step 5: Extract activations and compute mean projections
    print(f"\n{'='*60}")
    print("STEP 5: Extracting activations and computing mean projections")
    print(f"{'='*60}")
    
    # Choose processing method based on number of features and available memory
    use_chunked = args.all_features or len(feature_ids) > 10000
    
    if use_chunked:
        print("Using chunked processing for memory efficiency")
        # H100 can handle much larger batches
        if args.all_features and args.batch_size < 32:
            print(f"Note: With H100 GPU, you can use larger --batch_size (32-64) for faster processing")
        mean_activations, total_tokens = extract_and_project_activations_chunked(
            prompts_df['prompt'].tolist(),
            args.target_layer,
            model,
            tokenizer,
            sae,
            feature_ids,
            args.feature_chunk_size,
            args.batch_size,
            args.max_length
        )
    else:
        print("Using direct processing")
        feature_directions = extract_feature_directions(sae, feature_ids)
        mean_activations, total_tokens = extract_and_project_activations_direct(
            prompts_df['prompt'].tolist(),
            args.target_layer,
            model,
            tokenizer,
            feature_directions,
            args.batch_size,
            args.max_length
        )
    
    print(f"✓ Processed {total_tokens:,} tokens and computed mean activations")
    print(f"✓ Mean activations shape: {mean_activations.shape}")
    
    # Step 8: Merge with existing results and save
    print(f"\n{'='*60}")
    print("STEP 8: Merging with existing results and saving")
    print(f"{'='*60}")
    
    new_results = {
        'mean_activations': mean_activations,
        'feature_ids': feature_ids,
        'metadata': {
            'model_type': args.model_type,
            'model_name': config.chat_model_name,
            'sae_layer': args.target_layer,
            'sae_trainer': args.sae_trainer,
            'num_features': len(feature_ids)
        }
    }
    
    # Merge with existing results
    final_results = merge_results(existing_results, new_results, 
                                existing_feature_ids, feature_ids)
    
    torch.save(final_results, args.output_path)
    print(f"✓ Results saved to: {args.output_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens processed this run: {total_tokens:,}")
    print(f"New features processed this run: {len(feature_ids)}")
    print(f"Total features in final output: {len(final_results['feature_ids'])}")
    print(f"Previously existing features: {len(existing_feature_ids)}")
    
    final_activations = final_results['mean_activations']
    print(f"Final mean activation stats (all features):")
    print(f"  Min: {final_activations.min().item():.6f}")
    print(f"  Max: {final_activations.max().item():.6f}")
    print(f"  Mean: {final_activations.mean().item():.6f}")
    print(f"  Std: {final_activations.std().item():.6f}")
    
    if len(feature_ids) > 0:
        print(f"New features mean activation stats:")
        print(f"  Min: {mean_activations.min().item():.6f}")
        print(f"  Max: {mean_activations.max().item():.6f}")
        print(f"  Mean: {mean_activations.mean().item():.6f}")
        print(f"  Std: {mean_activations.std().item():.6f}")
    
    # Clean up
    del model, sae
    torch.cuda.empty_cache()
    print("\n✓ Memory cleaned up")
    print("Done!")

if __name__ == "__main__":
    main()