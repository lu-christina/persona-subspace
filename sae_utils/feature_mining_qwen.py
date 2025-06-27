"""
feature_mining.py
-----------------
Mine the *top-K* (highest-activation) text examples for every feature in a
Sparse-Autoencoder (SAE) dictionary trained on a transformer's residual stream.

Key points
~~~~~~~~~~
* Works on a single GPU.
* Saves checkpoints as chunked HDF5 files so downstream analysis can read a
  single feature without slurping the entire tensor.

The final HDF5 file contains:
    scores[F,K]      - best activation value
    tokens[F,K,L]    - token sequences
    sae_acts[F,K,L]  - SAE activations for those sequences
    freq[F]          - raw counts (int64)
with two attributes:
    processed_samples - how many *examples* seen
    tokens_seen       - how many *tokens* considered (denominator for freq)
"""

"""
CUDA_VISIBLE_DEVICES=0 python sae_utils/feature_mining.py \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --sae_path "/workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_19/trainer_1" \
  --layer_index 19 \
  --out_dir "/workspace/sae/llama-3-8b-instruct/feature_mining/resid_post_layer_19/trainer_1" \
  --ctx_len 512 \
  --num_samples 500000
"""


# ------------------------------------------------------------------------ #
#  Standard imports
# ------------------------------------------------------------------------ #
from json import decoder
import os, gc, math, pathlib, pickle, time, argparse
import dataclasses
from typing import Iterator

import torch, einops, tqdm, h5py
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, Qwen2ForCausalLM

from dictionary_learning.utils import load_dictionary              # SAE loader

# ------------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------------ #
BOS_OFFSET = 8  # Number of tokens to cut off at the beginning

# ------------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------------ #
@dataclasses.dataclass
class FeatureMiningConfig:
    """Configuration for feature mining."""
    # Model and SAE paths
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    sae_path: str = "/workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_19/trainer_1"
    layer_index: int = 19
    out_dir: pathlib.Path = pathlib.Path("/workspace/sae/llama-3-8b-instruct/feature_mining/resid_post_layer_19/trainer_1")
    
    # Processing parameters
    ctx_len: int = 512
    batch_size: int = 8
    num_samples: int = 500_000
    top_k: int = 20
    top_k_embed_unembed: int = 10
    
    # Device and dtype settings
    device: torch.device = torch.device("cuda:0")
    act_dtype: torch.dtype = torch.bfloat16
    score_dtype: torch.dtype = torch.bfloat16
    token_dtype: torch.dtype = torch.int32

    def __post_init__(self):
        # Convert string path to Path object if needed
        if isinstance(self.out_dir, str):
            self.out_dir = pathlib.Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description="Feature Mining Configuration")
    
    # Model and SAE paths
    parser.add_argument("--model_name", type=str, 
                       help="HuggingFace model name")
    parser.add_argument("--sae_path", type=str,
                       help="Path to the SAE dictionary")
    parser.add_argument("--layer_index", type=int,
                       help="Layer index to extract features from")
    parser.add_argument("--out_dir", type=str,
                       help="Output directory for mined features")
    
    # Processing parameters
    parser.add_argument("--ctx_len", type=int,
                       help="Maximum context length")
    parser.add_argument("--batch_size", type=int,
                       help="Batch size for processing")
    parser.add_argument("--num_samples", type=int,
                       help="Number of samples to process")
    parser.add_argument("--top_k", type=int,
                       help="Number of top examples to keep per feature")
    parser.add_argument("--top_k_embed_unembed", type=int,
                       help="Number of top embedding/unembedding similarities to keep")
    
    # Device and dtype settings
    parser.add_argument("--device", type=str,
                       help="Device to use (e.g., 'cuda:0')")
    parser.add_argument("--act_dtype", type=str, choices=["float16", "float32", "bfloat16"],
                       help="Activation dtype")
    parser.add_argument("--score_dtype", type=str, choices=["float16", "float32", "bfloat16"],
                       help="Score dtype")
    parser.add_argument("--token_dtype", type=str, choices=["int32", "int64"],
                       help="Token dtype")
    
    return parser.parse_args()

def get_config(args=None):
    """Get configuration, optionally overriding with command line arguments."""
    if args is None:
        args = parse_args()
    
    # Start with default config
    config_dict = dataclasses.asdict(FeatureMiningConfig())
    
    # Override with any provided arguments
    for key, value in vars(args).items():
        if value is not None:
            if key == "act_dtype":
                config_dict[key] = getattr(torch, value)
            elif key == "score_dtype":
                config_dict[key] = getattr(torch, value)
            elif key == "token_dtype":
                config_dict[key] = getattr(torch, value)
            elif key == "device":
                config_dict[key] = torch.device(value)
            elif key == "out_dir":
                config_dict[key] = pathlib.Path(value)
            else:
                config_dict[key] = value
    
    return FeatureMiningConfig(**config_dict)

# ------------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------------ #
def tokens_from_generator(
    text_gen: Iterator[str],
    tokenizer: AutoTokenizer,
    batch_size: int,
    ctx_len: int
):
    """
    Pull `batch_size` raw strings from an *infinite* iterator `text_gen`,
    tokenize with padding-to-max-length, and yield a dict of batched tensors.
    """
    while True:
        batch = [next(text_gen) for _ in range(batch_size)]
        yield tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=ctx_len,
            add_special_tokens=True,
        )


@torch.no_grad()
def collect_activations(model, submodule, inputs, config):
    """
    Run the transformer **until** `submodule` (layer `layer_index`),
    capture its *output* (residual stream), and abort the forward pass early
    to save compute + memory.
    """
    acts = None

    def hook(_, __, output):
        nonlocal acts
        acts = output[0] if isinstance(output, tuple) else output  # [B,L,D]
        raise StopForward

    class StopForward(Exception):
        pass

    handle = submodule.register_forward_hook(hook)
    try:
        _ = model(**inputs)           # the hook raises StopForward at layer L
    except StopForward:
        pass
    finally:
        handle.remove()

    return acts.to(config.act_dtype)   # ensure dtype is correct


# ------------------------------------------------------------------------ #
#  Main routine
# ------------------------------------------------------------------------ #
@torch.no_grad()
def mine_topk(
    text_gen: Iterator[str],
    out_path: pathlib.Path,
    config: FeatureMiningConfig,
    ) -> None:
    """Vectorised Top-K mining loop."""
    # ------------------- tokenizer & model --------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    assert tokenizer.bos_token_id != tokenizer.eos_token_id, \
        "BOS and EOS must differ or BOS masking will blank out everything."

    # NOTE: `device_map` needs a **dict**, not a torch.device
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.act_dtype,
        device_map={"": config.device.index},
    )
    model.eval()

    # SAE (dictionary) – weights live on the same GPU
    sae, _ = load_dictionary(config.sae_path, device=config.device)
    sae.eval()

    layer_mod = model.model.layers[config.layer_index]   # hook point
    dict_size = sae.dict_size                     # number of learned features
    print(f"[info] SAE dictionary size: {dict_size:,d} features")

    # ------------------- pre-allocate running Top-K buffers ---------------
    # Shapes: scores  (F,K)
    #         tokens  (F,K,L)
    #         acts    (F,K,L)
    top_k_scores = torch.full(
        (dict_size, config.top_k), -float("inf"),
        device=config.device, dtype=config.score_dtype
    )
    top_k_tokens = torch.full(
        (dict_size, config.top_k, config.ctx_len), tokenizer.pad_token_id,
        device=config.device, dtype=config.token_dtype
    )
    top_k_sae_acts = torch.zeros_like(top_k_tokens, dtype=config.score_dtype)
    freq_counts = torch.zeros(dict_size, dtype=torch.int64, device=config.device)
    tokens_seen = 0


    # ------------------- iterate over data --------------------------------
    token_iter = tokens_from_generator(text_gen, tokenizer, config.batch_size, config.ctx_len)
    n_batches  = math.ceil(config.num_samples / config.batch_size)
    processed  = 0

    for step, batch in enumerate(tqdm.tqdm(token_iter, total=n_batches,
                                           desc="mining")):
        # Stop after num_samples examples even if generator is infinite
        if processed >= config.num_samples:
            break

        # ---- 1. ship batch to GPU ----------------------------------------
        batch_gpu = {k: v.to(config.device) for k, v in batch.items()}

        # ---- 2. forward to layer L, then SAE encode ----------------------
        model_acts_BLD = collect_activations(model, layer_mod, batch_gpu, config)
        sae_acts_BLF   = sae.encode(model_acts_BLD)        # [B,L,F]

        # ---- 3. mask padding + BOS tokens + BOS_OFFSET ----------------
        # Create position mask for BOS_OFFSET
        seq_len = batch_gpu["input_ids"].shape[1]
        pos_mask = torch.arange(seq_len, device=config.device).unsqueeze(0) >= BOS_OFFSET
        
        mask_BL = (batch_gpu["attention_mask"] == 1) & \
                  pos_mask
        
        # ---- 3.5. filter out large activations ------------------------
        # Compute median norm across batch and sequence dimensions
        act_norms = sae_acts_BLF.norm(dim=-1)  # [B,L]
        median_norm = act_norms[mask_BL].median()
        large_act_mask = act_norms <= (8.0 * median_norm)
        
        # Apply both masks
        combined_mask_BL = mask_BL & large_act_mask
        sae_acts_masked_BLF = sae_acts_BLF * combined_mask_BL.unsqueeze(-1)

        # ---- frequency update (use combined mask) --------------------
        # Count tokens where feature activation > 0
        freq_counts += (sae_acts_masked_BLF > 0).sum(dim=(0, 1)).to(freq_counts.dtype)
        tokens_seen += combined_mask_BL.sum().item()

        # ---- 4. per-feature peak score over the sequence -----------------
        # peak_scores_BF: [B,F]
        peak_scores_BF, _ = sae_acts_masked_BLF.to(config.score_dtype).max(dim=1)
        current_scores_FB = peak_scores_BF.T.contiguous()  # [F,B]

        # ---- 5. prepare tokens & full-trace acts -------------------------
        # tokens: zero-copy expand instead of costly repeat  (F,B,L) view
        tokens_BL = batch_gpu["input_ids"].to(config.token_dtype)  # cast once
        current_tokens_FBL = tokens_BL.unsqueeze(0).expand(dict_size, -1, -1)

        # acts: need real tensor, so we rearrange (F,B,L)
        current_acts_FBL = einops.rearrange(sae_acts_masked_BLF,
                                            "b l f -> f b l")

        # ---- 6. cat with running buffers & keep global Top-K -------------
        # shapes after cat: (F, K+B)   /  (F, K+B, L)
        comb_scores = torch.cat([top_k_scores,  current_scores_FB], dim=1)
        comb_tokens = torch.cat([top_k_tokens,  current_tokens_FBL], dim=1)
        comb_acts   = torch.cat([top_k_sae_acts, current_acts_FBL], dim=1)

        # topk over dimension 1 (K+B) -> keep K best per feature
        new_scores, idx = torch.topk(comb_scores, config.top_k, dim=1)
        top_k_scores = new_scores

        # gather tokens & acts with the same indices
        idx_FKL = idx.unsqueeze(-1).expand(-1, -1, config.ctx_len)
        top_k_tokens = torch.gather(comb_tokens, dim=1, index=idx_FKL)
        top_k_sae_acts = torch.gather(comb_acts, dim=1, index=idx_FKL)

        # ---- 7. book-keeping --------------------------------------------
        processed += batch_gpu["input_ids"].shape[0]
        last_batch = (processed >= config.num_samples) or (step + 1 >= n_batches)

        # ---- 8. checkpoint to disk ---------------------------------------
        if last_batch:
            print(f"[save] {out_path}  ({processed:,d} samples)")

            # move to CPU for HDF5
            scores_np = top_k_scores.cpu().to(torch.float16).numpy()
            tokens_np = top_k_tokens.cpu().numpy()
            acts_np   = top_k_sae_acts.cpu().to(torch.float16).numpy()
            freq_np   = freq_counts.cpu().numpy()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w", libver="latest") as f:
                # metadata
                f.attrs.update({
                    "MODEL_NAME": config.model_name,
                    "SAE_PATH": config.sae_path,
                    "LAYER_INDEX": config.layer_index,
                    "CTX_LEN": config.ctx_len,
                    "TOP_K": config.top_k,
                    "processed_samples": processed,
                    "num_features": dict_size,
                    "tokens_seen": tokens_seen,
                })
                # chunk on first dim so reading a single feature is cheap
                # todo: compression="lzf"?
                f.create_dataset("scores",   data=scores_np, chunks=(1, config.top_k), compression="lzf")
                f.create_dataset("tokens",   data=tokens_np, chunks=(1, config.top_k, config.ctx_len), compression="lzf")
                f.create_dataset("sae_acts", data=acts_np, chunks=(1, config.top_k, config.ctx_len), compression="lzf")
                f.create_dataset("frequency", data=freq_np, chunks=(1,), compression="lzf")

            # free temp tensors before next loop
            del comb_scores, comb_tokens, comb_acts, new_scores, idx, idx_FKL
            del model_acts_BLD, sae_acts_BLF, sae_acts_masked_BLF
            del peak_scores_BF, current_scores_FB
            del current_tokens_FBL, current_acts_FBL, batch_gpu
            torch.cuda.empty_cache(); gc.collect()

    print(f"[done] {processed:,d} examples, {tokens_seen:,d} tokens processed.")

# ------------------------------------------------------------------------ #
#  Embedding / Un-embedding similarity DB
# ------------------------------------------------------------------------ #
@torch.no_grad()
def dump_embed_unembed_similarity(
    out_path: pathlib.Path,
    config: FeatureMiningConfig,
    eps: float = 1e-8,
):
    """Write top/bottom-K cosine-similarity rankings into an HDF-5 file."""

    def get_embed_matrix(m):
        if isinstance(m, LlamaForCausalLM):
            return m.model.embed_tokens.weight.data  # (V,D)
        elif isinstance(m, Qwen2ForCausalLM):
            return m.model.embed_tokens.weight.data  # (V,D)
        raise ValueError("Unsupported model type")

    def get_unembed_matrix(m):
        if isinstance(m, LlamaForCausalLM):
            return m.lm_head.weight.data.T           # (D,V)
        elif isinstance(m, Qwen2ForCausalLM):
            return m.lm_head.weight.data.T           # (D,V)
        raise ValueError("Unsupported model type")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.act_dtype,
        device_map={"": config.device.index},
    )
    model.eval()
    sae, _ = load_dictionary(config.sae_path, device=config.device)
    sae.eval()

    sae_enc = sae.encoder.weight.data.to(torch.float32)           # (F,D)
    sae_dec = sae.decoder.weight.data.T.to(torch.float32)         # (F,D)
    sae_enc = sae_enc / (sae_enc.norm(dim=-1, keepdim=True) + eps)
    sae_dec = sae_dec / (sae_dec.norm(dim=-1, keepdim=True) + eps)

    embed = get_embed_matrix(model).to(dtype=torch.float32, device=config.device)     # (V,D)
    unemb = get_unembed_matrix(model).T.to(dtype=torch.float32, device=config.device)   # (V,D)
    embed = embed / (embed.norm(dim=-1, keepdim=True) + eps)
    unemb = unemb / (unemb.norm(dim=-1, keepdim=True) + eps)

    vocab_size   = embed.size(0)
    num_features = sae_enc.size(0)
    k = config.top_k_embed_unembed

    # allocate output buffers
    shape = (num_features, k)
    in_top_val   = torch.empty(shape, device=config.device, dtype=torch.float32)
    in_top_idx   = torch.empty(shape, device=config.device, dtype=torch.int32)
    in_bot_val   = torch.empty_like(in_top_val)
    in_bot_idx   = torch.empty_like(in_top_idx)

    out_top_val  = torch.empty_like(in_top_val)
    out_top_idx  = torch.empty_like(in_top_idx)
    out_bot_val  = torch.empty_like(in_top_val)
    out_bot_idx  = torch.empty_like(in_top_idx)

    for i in tqdm.tqdm(range(num_features), desc="sim-DB"):
        enc_dir = sae_enc[i]
        dec_dir = sae_dec[i]

        sims_enc = torch.einsum("vd,d->v", embed, enc_dir)
        sims_dec = torch.einsum("vd,d->v", unemb, dec_dir)

        in_top_val[i],  in_top_idx[i]  = torch.topk(sims_enc, k, largest=True)
        in_bot_val[i],  in_bot_idx[i]  = torch.topk(sims_enc, k, largest=False)
        out_top_val[i], out_top_idx[i] = torch.topk(sims_dec, k, largest=True)
        out_bot_val[i], out_bot_idx[i] = torch.topk(sims_dec, k, largest=False)

    # ------------- write HDF-5 (tokens as int32, scores as float16) -------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w", libver="latest") as f:
        f.attrs.update({
            "MODEL_NAME":   config.model_name,
            "SAE_PATH":     config.sae_path,
            "k":            k,
            "num_features": num_features,
            "vocab_size":   vocab_size,
        })

        def ds_f16(name, tensor):
            f.create_dataset(name,
                             data=tensor.cpu().to(torch.float16).numpy(),
                             chunks=(1, k), compression="lzf")

        def ds_i32(name, tensor):
            f.create_dataset(name,
                             data=tensor.cpu().to(torch.int32).numpy(),
                             chunks=(1, k), compression="lzf")

        # embedding direction
        ds_f16("in_top_scores",     in_top_val)
        ds_i32("in_top_tokens",     in_top_idx)
        ds_f16("in_bottom_scores",  in_bot_val)
        ds_i32("in_bottom_tokens",  in_bot_idx)

        # un-embedding direction
        ds_f16("out_top_scores",    out_top_val)
        ds_i32("out_top_tokens",    out_top_idx)
        ds_f16("out_bottom_scores", out_bot_val)
        ds_i32("out_bottom_tokens", out_bot_idx)

    print(f"[sim-DB] saved → {out_path}")


# ----------------------------------------------------------------------- #
#  Example entry-point
# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    from custom_generator import hf_chat_dataset_to_generator, hf_dataset_to_generator
    
    # Parse arguments and get configuration
    config = get_config()

    chat_gen = hf_chat_dataset_to_generator(
        dataset_name="lmsys/lmsys-chat-1m",
        tokenizer=AutoTokenizer.from_pretrained(config.model_name),
        model_name=config.model_name,
        split="train",
        streaming=True,
        remove_system_prompt_p=0.9,
    )
    chat_out_path = config.out_dir / "chat_topk.h5"
    mine_topk(chat_gen, chat_out_path, config)

    pt_gen = hf_dataset_to_generator(
        dataset_name="monology/pile-uncopyrighted",
        split="train",
        streaming=True
    )
    pt_out_path = config.out_dir / "pt_topk.h5"
    mine_topk(pt_gen, pt_out_path, config)

    dump_embed_unembed_similarity(
        config.out_dir / "embed_unembed_similarity.h5", config
    )
