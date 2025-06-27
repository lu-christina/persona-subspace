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

# ------------------------------------------------------------------------ #
#  Standard imports
# ------------------------------------------------------------------------ #
from json import decoder
import os, gc, math, pathlib, pickle, time
from typing import Iterator

import torch, einops, tqdm, h5py
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

from dictionary_learning.utils import load_dictionary              # SAE loader

# ------------------------------------------------------------------------ #
#  Configuration – tweak as needed
# ------------------------------------------------------------------------ #
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_PATH    = "/workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_19/trainer_1"
LAYER_INDEX = 19                      # grab activations after this layer
CTX_LEN     = 512                     # max tokens per example
BATCH_SIZE  = 8                       # sequences per forward pass
NUM_SAMPLES = 500_000                 # how many examples to mine
TOP_K       = 20                      # keep K best examples per feature
TOP_K_EMBED_UNEMBED = 10

OUT_DIR = pathlib.Path("/workspace/sae/llama-3-8b-instruct/feature_mining/resid_post_layer_19/trainer_1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device       = torch.device("cuda:0")
act_dtype    = torch.bfloat16         # activations / SAE
score_dtype  = torch.bfloat16         # scores stored in Top-K buffers
token_dtype  = torch.int32            # tokens – int32 for vocab up to 2^31-1

torch.backends.cuda.matmul.allow_tf32 = True

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
def collect_activations(model, submodule, inputs):
    """
    Run the transformer **until** `submodule` (layer `LAYER_INDEX`),
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

    return acts.to(act_dtype)          # ensure dtype is bfloat16


# ------------------------------------------------------------------------ #
#  Main routine
# ------------------------------------------------------------------------ #
@torch.no_grad()
def mine_topk(
    text_gen: Iterator[str],
    out_path: pathlib.Path,
    top_k: int = TOP_K,
    ) -> None:
    """Vectorised Top-K mining loop."""
    # ------------------- tokenizer & model --------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    assert tokenizer.bos_token_id != tokenizer.eos_token_id, \
        "BOS and EOS must differ or BOS masking will blank out everything."

    # NOTE: `device_map` needs a **dict**, not a torch.device
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=act_dtype,
        device_map={"": device.index},        # <-- fixed
    )
    model.eval()

    # SAE (dictionary) – weights live on the same GPU
    sae, _ = load_dictionary(SAE_PATH, device=device)
    sae.eval()

    layer_mod = model.model.layers[LAYER_INDEX]   # hook point
    dict_size = sae.dict_size                     # number of learned features
    print(f"[info] SAE dictionary size: {dict_size:,d} features")

    # ------------------- pre-allocate running Top-K buffers ---------------
    # Shapes: scores  (F,K)
    #         tokens  (F,K,L)
    #         acts    (F,K,L)
    top_k_scores = torch.full(
        (dict_size, top_k), -float("inf"),
        device=device, dtype=score_dtype
    )
    top_k_tokens = torch.full(
        (dict_size, top_k, CTX_LEN), tokenizer.pad_token_id,
        device=device, dtype=token_dtype
    )
    top_k_sae_acts = torch.zeros_like(top_k_tokens, dtype=score_dtype)
    freq_counts = torch.zeros(dict_size, dtype=torch.int64, device=device)
    tokens_seen = 0


    # ------------------- iterate over data --------------------------------
    token_iter = tokens_from_generator(text_gen, tokenizer, BATCH_SIZE, CTX_LEN)
    n_batches  = math.ceil(NUM_SAMPLES / BATCH_SIZE)
    processed  = 0

    for step, batch in enumerate(tqdm.tqdm(token_iter, total=n_batches,
                                           desc="mining")):
        # Stop after NUM_SAMPLES examples even if generator is infinite
        if processed >= NUM_SAMPLES:
            break

        # ---- 1. ship batch to GPU ----------------------------------------
        batch_gpu = {k: v.to(device) for k, v in batch.items()}

        # ---- 2. forward to layer L, then SAE encode ----------------------
        model_acts_BLD = collect_activations(model, layer_mod, batch_gpu)
        sae_acts_BLF   = sae.encode(model_acts_BLD)        # [B,L,F]

        # ---- 3. mask padding + BOS tokens --------------------------------
        mask_BL = (batch_gpu["attention_mask"] == 1) & \
                  (batch_gpu["input_ids"] != tokenizer.bos_token_id)
        sae_acts_masked_BLF = sae_acts_BLF * mask_BL.unsqueeze(-1)

        # ---- frequency update ------------------------------------------- #
        # Count tokens where feature activation > 0
        freq_counts += (sae_acts_masked_BLF > 0).sum(dim=(0, 1)).to(freq_counts.dtype)
        tokens_seen += mask_BL.sum().item()

        # ---- 4. per-feature peak score over the sequence -----------------
        # peak_scores_BF: [B,F]
        peak_scores_BF, _ = sae_acts_masked_BLF.to(score_dtype).max(dim=1)
        current_scores_FB = peak_scores_BF.T.contiguous()  # [F,B]

        # ---- 5. prepare tokens & full-trace acts -------------------------
        # tokens: zero-copy expand instead of costly repeat  (F,B,L) view
        tokens_BL = batch_gpu["input_ids"].to(token_dtype)  # cast once
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
        new_scores, idx = torch.topk(comb_scores, top_k, dim=1)
        top_k_scores = new_scores

        # gather tokens & acts with the same indices
        idx_FKL = idx.unsqueeze(-1).expand(-1, -1, CTX_LEN)
        top_k_tokens = torch.gather(comb_tokens, dim=1, index=idx_FKL)
        top_k_sae_acts = torch.gather(comb_acts, dim=1, index=idx_FKL)

        # ---- 7. book-keeping --------------------------------------------
        processed += batch_gpu["input_ids"].shape[0]
        last_batch = (processed >= NUM_SAMPLES) or (step + 1 >= n_batches)

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
                    "MODEL_NAME": MODEL_NAME,
                    "SAE_PATH": SAE_PATH,
                    "LAYER_INDEX": LAYER_INDEX,
                    "CTX_LEN": CTX_LEN,
                    "TOP_K": top_k,
                    "processed_samples": processed,
                    "num_features": dict_size,
                    "tokens_seen": tokens_seen,
                })
                # chunk on first dim so reading a single feature is cheap
                # todo: compression="lzf"?
                f.create_dataset("scores",   data=scores_np, chunks=(1, top_k), compression="lzf")
                f.create_dataset("tokens",   data=tokens_np, chunks=(1, top_k, CTX_LEN), compression="lzf")
                f.create_dataset("sae_acts", data=acts_np, chunks=(1, top_k, CTX_LEN), compression="lzf")
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
    k: int = 10,
    eps: float = 1e-8,
):
    """Write top/bottom-K cosine-similarity rankings into an HDF-5 file."""

    def get_embed_matrix(m):
        if isinstance(m, LlamaForCausalLM):
            return m.model.embed_tokens.weight.data  # (V,D)
        raise ValueError("Unsupported model type")

    def get_unembed_matrix(m):
        if isinstance(m, LlamaForCausalLM):
            return m.lm_head.weight.data.T           # (D,V)
        raise ValueError("Unsupported model type")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=act_dtype,
        device_map={"": device.index},
    )
    model.eval()
    sae, _ = load_dictionary(SAE_PATH, device=device)
    sae.eval()

    sae_enc = sae.encoder.weight.data.to(torch.float32)           # (F,D)
    sae_dec = sae.decoder.weight.data.T.to(torch.float32)         # (F,D)
    sae_enc = sae_enc / (sae_enc.norm(dim=-1, keepdim=True) + eps)
    sae_dec = sae_dec / (sae_dec.norm(dim=-1, keepdim=True) + eps)

    embed = get_embed_matrix(model).to(dtype=torch.float32, device=device)     # (V,D)
    unemb = get_unembed_matrix(model).T.to(dtype=torch.float32, device=device)   # (V,D)
    embed = embed / (embed.norm(dim=-1, keepdim=True) + eps)
    unemb = unemb / (unemb.norm(dim=-1, keepdim=True) + eps)

    vocab_size   = embed.size(0)
    num_features = sae_enc.size(0)

    # allocate output buffers
    shape = (num_features, k)
    in_top_val   = torch.empty(shape, device=device, dtype=torch.float32)
    in_top_idx   = torch.empty(shape, device=device, dtype=torch.int32)
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
            "MODEL_NAME":   MODEL_NAME,
            "SAE_PATH":     SAE_PATH,
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

    chat_gen = hf_chat_dataset_to_generator(
        dataset_name="lmsys/lmsys-chat-1m",
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        model_name=MODEL_NAME,
        split="train",
        streaming=True,
        remove_system_prompt_p=0.9,
    )
    chat_out_path = pathlib.Path(OUT_DIR / "chat_topk.h5")
    mine_topk(chat_gen, chat_out_path)

    pt_gen = hf_dataset_to_generator(
        dataset_name="monology/pile-uncopyrighted",
        split="train",
        streaming=True
    )
    pt_out_path = pathlib.Path(OUT_DIR / "pt_topk.h5")
    mine_topk(pt_gen, pt_out_path)

    dump_embed_unembed_similarity(
        OUT_DIR / "embed_unembed_similarity.h5", k=TOP_K_EMBED_UNEMBED
    )
