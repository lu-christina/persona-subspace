"""
feature_mining.py
-----------------
Mine the top-K (highest-activation) text examples for every feature in a
Sparse Autoencoder (SAE) dictionary. Adapted from model-diffing-em.

Usage:
    python feature_mining.py --num_samples 10000 --top_k 10
"""

import os
import gc
import math
import pathlib
import time
import argparse
import dataclasses
from typing import Iterator, List

import torch
import einops
import tqdm
import h5py
from transformers import AutoTokenizer, AutoModelForCausalLM
from dictionary_learning.trainers import BatchTopKSAE


@dataclasses.dataclass
class FeatureMiningConfig:
    """Configuration for feature mining."""
    # Model and SAE paths - using /workspace for big files
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    sae_path: str = "/workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_15/trainer_0"
    layer_index: int = 15
    out_dir: pathlib.Path = pathlib.Path("/workspace/feature_mining")
    
    # Processing parameters
    ctx_len: int = 512
    batch_size: int = 4  # Smaller batch size for memory efficiency
    num_samples: int = 10_000  # Smaller default for testing
    top_k: int = 10  # Fewer examples per feature
    
    # Device and dtype settings
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_dtype: torch.dtype = torch.bfloat16
    score_dtype: torch.dtype = torch.bfloat16
    token_dtype: torch.dtype = torch.int32

    def __post_init__(self):
        if isinstance(self.out_dir, str):
            self.out_dir = pathlib.Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Mining Configuration")
    parser.add_argument("--num_samples", type=int, default=10_000,
                       help="Number of samples to process")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top examples to keep per feature")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--ctx_len", type=int, default=512,
                       help="Maximum context length")
    parser.add_argument("--feature_indices", type=str, default=None,
                       help="Comma-separated list of specific feature indices to analyze")
    return parser.parse_args()


def simple_text_generator(num_samples: int = 10_000):
    """Simple text generator using a small dataset for testing."""
    from datasets import load_dataset
    
    # Use a small, fast dataset for testing
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    count = 0
    for example in dataset:
        if count >= num_samples:
            break
        text = example["text"].strip()
        if len(text) > 50:  # Skip very short texts
            yield text
            count += 1


def tokens_from_generator(
    text_gen: Iterator[str],
    tokenizer: AutoTokenizer,
    batch_size: int,
    ctx_len: int
):
    """Convert text generator to tokenized batches."""
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(text_gen))
            except StopIteration:
                if batch:
                    # Yield partial batch if we have some texts
                    break
                else:
                    return
        
        if not batch:
            return
            
        yield tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=ctx_len,
            add_special_tokens=True,
        )


@torch.no_grad()
def collect_activations(model, layer_module, inputs, config):
    """Collect activations from specified layer."""
    acts = None

    def hook(_, __, output):
        nonlocal acts
        acts = output[0] if isinstance(output, tuple) else output
        raise StopForward

    class StopForward(Exception):
        pass

    handle = layer_module.register_forward_hook(hook)
    try:
        _ = model(**inputs)
    except StopForward:
        pass
    finally:
        handle.remove()

    return acts.to(config.act_dtype)


@torch.no_grad()
def mine_top_features(
    text_gen: Iterator[str],
    out_path: pathlib.Path,
    config: FeatureMiningConfig,
    specific_features: List[int] = None,
) -> None:
    """Mine top-K examples for SAE features."""
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    
    # Set up cache directory
    cache_dir = "/workspace/model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if model is already cached
    model_cache_path = os.path.join(cache_dir, f"models--{config.model_name.replace('/', '--')}")
    if os.path.exists(model_cache_path):
        print(f"✓ Found cached model at: {model_cache_path}")
    else:
        print(f"Model not cached, will download to: {cache_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.act_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "70GB"},
        cache_dir=cache_dir
    )
    model.eval()

    # Check if SAE exists locally first
    if os.path.exists(config.sae_path):
        print(f"✓ Found SAE at: {config.sae_path}")
        sae = BatchTopKSAE.from_pretrained(config.sae_path, device="cpu")
    else:
        print(f"SAE not found at {config.sae_path}")
        print("Please ensure SAE is downloaded to /workspace/sae/ directory")
        raise FileNotFoundError(f"SAE not found at {config.sae_path}")
    
    sae = sae.to(config.device)
    sae.eval()

    layer_mod = model.model.layers[config.layer_index]
    dict_size = sae.dict_size
    
    # If specific features provided, only analyze those
    if specific_features:
        features_to_analyze = specific_features
        print(f"Analyzing {len(features_to_analyze)} specific features: {features_to_analyze}")
    else:
        features_to_analyze = list(range(dict_size))
        print(f"Analyzing all {dict_size} features")

    # Pre-allocate buffers only for features we're analyzing
    num_features = len(features_to_analyze)
    top_k_scores = torch.full(
        (num_features, config.top_k), -float("inf"),
        device=config.device, dtype=config.score_dtype
    )
    top_k_tokens = torch.full(
        (num_features, config.top_k, config.ctx_len), tokenizer.pad_token_id,
        device=config.device, dtype=config.token_dtype
    )
    top_k_sae_acts = torch.zeros_like(top_k_tokens, dtype=config.score_dtype)
    
    # Feature index mapping
    feature_idx_map = {feat_id: i for i, feat_id in enumerate(features_to_analyze)}

    # Process data
    token_iter = tokens_from_generator(text_gen, tokenizer, config.batch_size, config.ctx_len)
    n_batches = math.ceil(config.num_samples / config.batch_size)
    processed = 0

    for step, batch in enumerate(tqdm.tqdm(token_iter, total=n_batches, desc="Mining features")):
        if processed >= config.num_samples:
            break

        # Move batch to GPU
        batch_gpu = {k: v.to(config.device) for k, v in batch.items()}

        # Forward pass and SAE encode
        model_acts = collect_activations(model, layer_mod, batch_gpu, config)  # [B,L,D]
        sae_acts = sae.encode(model_acts)  # [B,L,F]

        # Mask padding tokens
        mask = (batch_gpu["attention_mask"] == 1) & \
               (batch_gpu["input_ids"] != tokenizer.bos_token_id)
        sae_acts_masked = sae_acts * mask.unsqueeze(-1)

        # Get peak scores per sequence for our specific features
        sae_acts_subset = sae_acts_masked[:, :, features_to_analyze]  # [B,L,num_features]
        peak_scores, _ = sae_acts_subset.to(config.score_dtype).max(dim=1)  # [B,num_features]
        current_scores = peak_scores.T.contiguous()  # [num_features,B]

        # Prepare tokens and acts
        tokens_BL = batch_gpu["input_ids"].to(config.token_dtype)
        current_tokens = tokens_BL.unsqueeze(0).expand(num_features, -1, -1)  # [num_features,B,L]
        
        # Get SAE acts for our subset of features
        current_acts = einops.rearrange(sae_acts_subset, "b l f -> f b l")  # [num_features,B,L]

        # Combine with existing top-K and update
        comb_scores = torch.cat([top_k_scores, current_scores], dim=1)
        comb_tokens = torch.cat([top_k_tokens, current_tokens], dim=1)
        comb_acts = torch.cat([top_k_sae_acts, current_acts], dim=1)

        # Keep top-K
        new_scores, idx = torch.topk(comb_scores, config.top_k, dim=1)
        top_k_scores = new_scores

        # Gather corresponding tokens and acts
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, config.ctx_len)
        top_k_tokens = torch.gather(comb_tokens, dim=1, index=idx_expanded)
        top_k_sae_acts = torch.gather(comb_acts, dim=1, index=idx_expanded)

        processed += batch_gpu["input_ids"].shape[0]
        
        # Clean up GPU memory
        del batch_gpu, model_acts, sae_acts, sae_acts_masked, sae_acts_subset
        del peak_scores, current_scores, current_tokens, current_acts
        del comb_scores, comb_tokens, comb_acts, new_scores, idx, idx_expanded
        torch.cuda.empty_cache()

    # Save results
    print(f"Saving results to {out_path}...")
    scores_np = top_k_scores.cpu().to(torch.float16).numpy()
    tokens_np = top_k_tokens.cpu().numpy()
    acts_np = top_k_sae_acts.cpu().to(torch.float16).numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w", libver="latest") as f:
        f.attrs.update({
            "MODEL_NAME": config.model_name,
            "SAE_PATH": config.sae_path,
            "LAYER_INDEX": config.layer_index,
            "CTX_LEN": config.ctx_len,
            "TOP_K": config.top_k,
            "processed_samples": processed,
            "num_features_analyzed": num_features,
            "feature_indices": features_to_analyze,
        })
        
        f.create_dataset("scores", data=scores_np, chunks=(1, config.top_k), compression="lzf")
        f.create_dataset("tokens", data=tokens_np, chunks=(1, config.top_k, config.ctx_len), compression="lzf")
        f.create_dataset("sae_acts", data=acts_np, chunks=(1, config.top_k, config.ctx_len), compression="lzf")
        f.create_dataset("feature_indices", data=features_to_analyze, compression="lzf")

    print(f"Done! Processed {processed:,d} examples for {num_features} features.")


def load_and_display_results(h5_path: pathlib.Path, tokenizer: AutoTokenizer, feature_idx: int = 0, top_n: int = 5):
    """Load and display top activating examples for a specific feature."""
    with h5py.File(h5_path, "r") as f:
        scores = f["scores"][feature_idx]  # [top_k]
        tokens = f["tokens"][feature_idx]  # [top_k, ctx_len]
        feature_indices = f["feature_indices"][:]
        
        # Get the actual feature ID
        actual_feature_id = feature_indices[feature_idx]
        
        print(f"\nTop {top_n} examples for feature {actual_feature_id}:")
        print("=" * 80)
        
        for i in range(min(top_n, len(scores))):
            score = scores[i]
            token_ids = tokens[i]
            
            # Decode tokens
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            text = text.strip()
            
            print(f"Rank {i+1} (score: {score:.4f}):")
            print(f"  {text[:200]}{'...' if len(text) > 200 else ''}")
            print()


if __name__ == "__main__":
    args = parse_args()
    
    # Create config with command line overrides
    config = FeatureMiningConfig(
        num_samples=args.num_samples,
        top_k=args.top_k,
        batch_size=args.batch_size,
        ctx_len=args.ctx_len,
    )
    
    # Parse specific feature indices if provided
    specific_features = None
    if args.feature_indices:
        specific_features = [int(x.strip()) for x in args.feature_indices.split(",")]
    
    # Create text generator
    text_gen = simple_text_generator(config.num_samples)
    
    # Mine features
    out_path = config.out_dir / "topk_examples.h5"
    mine_top_features(text_gen, out_path, config, specific_features)
    
    # Display some results
    print("\nExample results:")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if out_path.exists():
        load_and_display_results(out_path, tokenizer, feature_idx=0, top_n=3)