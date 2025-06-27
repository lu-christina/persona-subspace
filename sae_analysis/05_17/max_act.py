#%%
from IPython import get_ipython

# Initialize IPython shell
ipython = get_ipython()
if ipython is None:  # If not running in IPython environment
    from IPython import embed
    ipython = embed()

# Now you can run magic commands
ipython.run_line_magic('load_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')

# %%
import sys
sys.path.append('.')

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# %%

# run_feature_mining.py
import os, gc, json, math, torch, einops, tqdm, pickle, pathlib, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from dictionary_learning.utils import load_dictionary
from typing import Iterator
import h5py # Import HDF5 library

# ----------------------------------------
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_PATH    = "/workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_19/trainer_1"
LAYER_INDEX = 19                         # the layer you trained on
CTX_LEN     = 512                        # max tokens per example
BATCH_SIZE  = 8                          # tokens per GPU pass
NUM_SAMPLES = 100_000 # 524_288             # number of samples
TOP_K       = 20                         # how many examples to keep per feature
SAVE_EVERY  = 1_000_000                  # dump intermediate pickle every N batches
OUT_DIR     = pathlib.Path("feature_mining_runs/run1_hdf5") # Changed output dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------
device = torch.device("cuda:0")
dtype  = torch.bfloat16
score_dtype = torch.bfloat16
torch.backends.cuda.matmul.allow_tf32 = True

# %%
def tokens_from_generator(
    text_gen: Iterator[str],
    tokenizer: AutoTokenizer,
    batch_size: int,
    ctx_len: int
):
    while True:
        batch = [next(text_gen) for _ in range(batch_size)]
        yield tokenizer(
            batch,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=ctx_len,
            add_special_tokens=True
        )

# %%
@torch.no_grad()
def collect_activations(model, submodule, inputs):
    """Grab residual stream from `submodule` and abort forward early."""
    acts = None
    def hook(_, __, output):
        nonlocal acts
        acts = output[0] if isinstance(output, tuple) else output
        raise StopForward
    class StopForward(Exception): pass
    handle = submodule.register_forward_hook(hook)
    try:
        _ = model(**inputs)
    except StopForward:
        pass
    finally:
        handle.remove()
    return acts # [B,L,D] on GPU


# %%
# %%
import pickle, math
from collections import defaultdict
from typing import Iterator, List, Tuple

# %%

@torch.no_grad()
def mine_topk(text_gen: Iterator[str]) -> None:
    # ---------- constants for the tweaks ----------
    TOK_DTYPE = torch.int32
    # ----------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    assert tokenizer.bos_token_id is not None, "bos_token_id must be set"
    assert tokenizer.bos_token_id != tokenizer.eos_token_id, "bos_token_id and eos_token_id must be different"

    model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=dtype,
                device_map=device)
    model.eval()

    sae, _ = load_dictionary(SAE_PATH, device=device)
    sae.eval()

    layer_mod  = model.model.layers[LAYER_INDEX]
    dict_size  = sae.dict_size
    print(f"SAE dictionary size (num features): {dict_size}")

    token_iter = tokens_from_generator(text_gen, tokenizer, BATCH_SIZE, CTX_LEN)
    n_batches  = math.ceil(NUM_SAMPLES / BATCH_SIZE)
    
    # ------------------------------------------------------------------
    # Allocate running Top-K buffers
    # ------------------------------------------------------------------
    top_k_scores_FK   = torch.full((dict_size, TOP_K), -float('inf'),
                                   device=device, dtype=score_dtype)
    top_k_tokens_FKL  = torch.full((dict_size, TOP_K, CTX_LEN),
                                   tokenizer.pad_token_id,
                                   device=device, dtype=TOK_DTYPE)
    top_k_sae_acts_FKL = torch.zeros((dict_size, TOP_K, CTX_LEN),
                                     device=device, dtype=score_dtype)

    processed_samples = 0
    for step, batch in enumerate(tqdm.tqdm(token_iter, total=n_batches, desc="mining…")):
        if processed_samples >= NUM_SAMPLES:
            break
        
        current_actual_batch_size = batch["input_ids"].shape[0]
        batch_on_device = {k: v.to(device) for k, v in batch.items()}

        model_acts_BLD = collect_activations(model, layer_mod, batch_on_device)
        sae_acts_BLF   = sae.encode(model_acts_BLD)

        mask_BL = (batch_on_device["attention_mask"] == 1) & \
                  (batch_on_device["input_ids"] != tokenizer.bos_token_id)
        sae_acts_masked_BLF = sae_acts_BLF * mask_BL.unsqueeze(-1)
        peak_scores_BF, _   = sae_acts_masked_BLF.to(score_dtype).max(dim=1)

        current_scores_FB = einops.rearrange(peak_scores_BF, 'b f -> f b')
        current_tokens_BL          = batch_on_device["input_ids"].to(TOK_DTYPE)
        current_tokens_to_cat_FBL  = current_tokens_BL.unsqueeze(0).expand(dict_size, -1, -1)
        current_sae_acts_to_cat_FBL = einops.rearrange(sae_acts_masked_BLF,
                                                       'b l f -> f b l')

        # concatenate with running buffers
        combined_scores_FKplusB   = torch.cat([top_k_scores_FK,  current_scores_FB], dim=1)
        combined_tokens_FKplusB_L = torch.cat([top_k_tokens_FKL, current_tokens_to_cat_FBL], dim=1)
        combined_sae_acts_FKplusB_L = torch.cat([top_k_sae_acts_FKL,
                                                 current_sae_acts_to_cat_FBL], dim=1)
        
        new_top_k_scores_FK, top_indices_in_combined_FK = torch.topk(
                                        combined_scores_FKplusB, TOP_K, dim=1)
        top_k_scores_FK = new_top_k_scores_FK

        idx_for_gather_FKL = top_indices_in_combined_FK.unsqueeze(-1).expand(-1, -1, CTX_LEN)
        top_k_tokens_FKL   = torch.gather(combined_tokens_FKplusB_L, dim=1,
                                          index=idx_for_gather_FKL)
        top_k_sae_acts_FKL = torch.gather(combined_sae_acts_FKplusB_L, dim=1,
                                          index=idx_for_gather_FKL)
        
        processed_samples += current_actual_batch_size
        
        is_last_batch = (processed_samples >= NUM_SAMPLES) or (step + 1 >= n_batches)

        if (step + 1) % SAVE_EVERY == 0 or is_last_batch:
            # --- cast back to original dtypes for disk friendliness ----
            scores_cpu = top_k_scores_FK.cpu().to(torch.float16).numpy()
            tokens_cpu = top_k_tokens_FKL.cpu().to(torch.int32).numpy()
            sae_acts_cpu = top_k_sae_acts_FKL.cpu().to(torch.float16).numpy()

            file_suffix  = "final" if is_last_batch else f"ckpt_samples_{processed_samples}"
            filepath_h5  = OUT_DIR / f"topk_{file_suffix}.h5"
            
            print(f"Saving checkpoint to {filepath_h5} "
                  f"(processed {processed_samples} samples / {step+1} batches)")
            try:
                with h5py.File(filepath_h5, 'w', libver='latest') as f:
                    # metadata
                    f.attrs.update({
                        'MODEL_NAME'      : MODEL_NAME,
                        'SAE_PATH'        : SAE_PATH,
                        'LAYER_INDEX'     : LAYER_INDEX,
                        'CTX_LEN'         : CTX_LEN,
                        'TOP_K'           : TOP_K,
                        'processed_samples': processed_samples,
                        'num_features'    : dict_size,
                    })
                    chunk_scores    = (1, TOP_K)
                    chunk_sequences = (1, TOP_K, CTX_LEN)

                    f.create_dataset('scores',   data=scores_cpu, chunks=chunk_scores)
                    f.create_dataset('tokens',   data=tokens_cpu, chunks=chunk_sequences)
                    f.create_dataset('sae_acts', data=sae_acts_cpu, chunks=chunk_sequences)
                print("✓ saved")
            except Exception as e:
                print(f"✗ error saving HDF5 checkpoint: {e}")

            # housekeeping
            del (combined_scores_FKplusB, combined_tokens_FKplusB_L,
                 combined_sae_acts_FKplusB_L, new_top_k_scores_FK,
                 top_indices_in_combined_FK, idx_for_gather_FKL,
                 model_acts_BLD, sae_acts_BLF, sae_acts_masked_BLF,
                 peak_scores_BF, current_scores_FB,
                 current_tokens_to_cat_FBL, current_sae_acts_to_cat_FBL,
                 batch_on_device)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        if processed_samples >= NUM_SAMPLES:
            break
            
    print(f"Finished mining. Processed {processed_samples} samples.")

# %%

# if __name__ == "__main__":
from custom_generator import hf_chat_dataset_to_generator
text_iter = hf_chat_dataset_to_generator(
    dataset_name = "lmsys/lmsys-chat-1m",
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME),
    model_name   = MODEL_NAME,
    split        = "train",
    streaming    = True,
    remove_system_prompt_p=0.9,
)
mine_topk(text_iter)
# %%

import numpy as np
HDF5_FILE_PATH = pathlib.Path("/root/git/model-diffing-em/sae_analysis/05_17/feature_mining_runs/run1_hdf5/topk_final.h5")

# Load the tokenizer used during mining for decoding
TOKENIZER_NAME = MODEL_NAME # From the mining script
tokenizer_for_display = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer_for_display.pad_token is None:
    tokenizer_for_display.pad_token = tokenizer_for_display.eos_token


def load_feature_data(h5_filepath: pathlib.Path, feature_idx: int) -> List[Tuple[float, str, List[float]]]:
    """
    Loads top-K examples for a specific feature from the HDF5 file.

    Returns:
        A list of tuples, where each tuple is (score, decoded_text, list_of_activations).
    """
    if not h5_filepath.exists():
        print(f"Error: HDF5 file not found at {h5_filepath}")
        return []

    examples_for_feature = []
    with h5py.File(h5_filepath, 'r') as f:
        if 'scores' not in f or 'tokens' not in f or 'sae_acts' not in f:
            print("Error: One or more required datasets (scores, tokens, sae_acts) not found in HDF5 file.")
            return []
        
        num_features_in_file = f['scores'].shape[0]
        if feature_idx >= num_features_in_file:
            print(f"Error: feature_idx {feature_idx} is out of bounds. File has {num_features_in_file} features.")
            return []

        # Retrieve data for the specific feature
        # Slicing HDF5 datasets is memory-efficient
        feature_scores_K = f['scores'][feature_idx, :]       # Shape: (TOP_K,)
        feature_tokens_KL = f['tokens'][feature_idx, :, :]    # Shape: (TOP_K, CTX_LEN)
        feature_sae_acts_KL = f['sae_acts'][feature_idx, :, :]# Shape: (TOP_K, CTX_LEN)
        
        # Get stored dtype for SAE acts if available, otherwise assume float16
        stored_sae_acts_dtype_str = f.attrs.get('sae_acts_dtype_stored', 'torch.float16')
        # Note: If you stored as uint16 representing bfloat16, you'd need to convert back here.
        # For float16, NumPy handles it directly.

        top_k_file = feature_scores_K.shape[0] # Actual TOP_K in file for this feature

        for k_idx in range(top_k_file):
            score = float(feature_scores_K[k_idx])
            
            # Skip if score is -inf (placeholder for unpopulated entries)
            if score == -float('inf'):
                continue

            tokens_L_np = feature_tokens_KL[k_idx, :] # Single sequence of token IDs
            sae_acts_L_np = feature_sae_acts_KL[k_idx, :] # Single sequence of SAE activations

            # Decode tokens, removing padding
            # Find first pad token if any
            pad_token_id_to_check = tokenizer_for_display.pad_token_id
            
            # Ensure tokens_L_np is a Python list or 1D NumPy array for processing
            if isinstance(tokens_L_np, np.ndarray):
                actual_tokens_list = tokens_L_np.tolist()
            else: # Should be list if not ndarray from h5py
                actual_tokens_list = list(tokens_L_np)

            tokens_to_decode = actual_tokens_list
            
            # Handle cases where tokens_to_decode might be empty after stripping padding
            decoded_text = tokenizer_for_display.decode(tokens_to_decode, skip_special_tokens=False)

            # Convert SAE activations to a list of floats for display
            sae_acts_list = sae_acts_L_np.tolist() # NumPy array to Python list

            examples_for_feature.append((score, decoded_text, sae_acts_list))
            
    # The examples are already sorted by score (desc) due to how they were saved (from torch.topk)
    return examples_for_feature

# --- Choose a feature to inspect ---
feature_to_inspect = 0 # Example feature index

print(f"\n--- Top Activating Examples for Feature {feature_to_inspect} ---")
print(f"Loading from: {HDF5_FILE_PATH}\n")

loaded_examples = load_feature_data(HDF5_FILE_PATH, feature_to_inspect)

if loaded_examples:
    for i, (score, text, acts_preview) in enumerate(loaded_examples):
        print(f"Rank {i+1}: Score = {score:.4f}")
        print(f"  Text: {repr(text)}")
        print(f"  Activations: {acts_preview}") # Optional: print some activation values
        print("-" * 20)
else:
    print(f"No data loaded for feature {feature_to_inspect}.")
# %%
