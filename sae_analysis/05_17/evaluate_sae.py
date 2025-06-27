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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# %%

import sys
sys.path.append('.')

import gc
import os
import torch
import numpy as np
import einops
import transformer_lens
import functools
import plotly.graph_objects as go
import plotly.express as px
import circuitsvis as cv
import tqdm
import json
import pandas as pd

from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import Tensor
from torch.utils.data import Dataset
from jaxtyping import Int, Float
from typing import Union, Tuple, List
from sklearn.decomposition import PCA

# %%

from dictionary_learning.trainers import BatchTopKSAE

sae_path = "/workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_19/trainer_1/ae.pt"
ae = BatchTopKSAE.from_pretrained(sae_path).to("cuda:0")

# %%
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", torch_dtype=torch.bfloat16)
# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# %%

from custom_generator import hf_dataset_to_generator
from model_utils import collect_activations, get_bos_pad_eos_mask
import torch.nn.functional as F

def tokens_from_generator(text_generator, tokenizer, batch_size, max_length):
    while True:
        batch_texts = []
        for _ in range(batch_size):
            batch_texts.append(next(text_generator))

        tokenized_batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        yield tokenized_batch

# %%

@torch.no_grad()
def evaluate(generator, batch_size, n_batches, max_length, model, tokenizer, ae, module):
    """
    Evaluate a Sparse Autoencoder on model activations.
    
    Args:
        generator: Text generator function
        batch_size: Number of samples per batch
        n_batches: Number of batches to process
        max_length: Maximum token length
        model: The language model
        tokenizer: The tokenizer
        ae: The Sparse Autoencoder
        module: The specific layer/module to extract activations from
        
    Returns:
        dict: Evaluation statistics
    """
    # Initialize token iterator
    token_iterator = tokens_from_generator(generator, tokenizer, batch_size, max_length)
    
    # Initialize statistics storage
    all_l2_loss = []
    all_l1_loss = []
    all_l0_norm = []
    total_active_features = torch.zeros(ae.dict_size, device=model.device, dtype=torch.float32)
    all_cossim = []
    all_l2_ratio = []
    all_frac_variance_explained = []
    
    print(f"Starting evaluation for {n_batches} batches...")
    
    for batch_idx in tqdm.tqdm(range(n_batches)):
        batch = next(token_iterator)
    
        # 1. Extract activations from the language model
        with torch.no_grad():
            original_activations_BLD = collect_activations(
                model,
                module,
                batch
            )
    
        # 2. Prepare masks
        remove_bos = True
        if remove_bos:
            attention_mask_BL = batch["attention_mask"][:, 1:]
            original_activations_BLD = original_activations_BLD[:, 1:]
        else:
            attention_mask_BL = batch["attention_mask"]
        x = original_activations_BLD[attention_mask_BL == 1]
    
        # 3. Run the SAE
        with torch.no_grad():
            x_hat, f = ae(x, output_features=True)
    
        # 4. Compute statistics
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f > 0).float().sum(dim=-1).mean()
    
        total_active_features += f.sum(dim=0) 
    
        epsilon = 1e-8
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_norm = torch.linalg.norm(x_hat, dim=-1, keepdim=True)
        x_normed = x / (x_norm + epsilon)
        x_hat_normed = x_hat / (x_hat_norm + epsilon)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()
    
        l2_ratio = (torch.linalg.norm(x_hat, dim=-1) / (torch.linalg.norm(x, dim=-1) + epsilon)).mean()
    
        total_variance = torch.var(x, dim=0).sum()
        residual_variance = torch.var(x - x_hat, dim=0).sum()
        frac_variance_explained = (1 - residual_variance / total_variance)
    
        all_l2_loss.append(l2_loss.item())
        all_l1_loss.append(l1_loss.item())
        all_l0_norm.append(l0.item())
        all_cossim.append(cossim.item())
        all_l2_ratio.append(l2_ratio.item())
        all_frac_variance_explained.append(frac_variance_explained.item())
    
        # Clean up memory each iteration
        del original_activations_BLD, x, x_hat, f

    # Calculate and print average statistics
    stats = {}
    
    print("\n--- Evaluation Summary ---")
    if all_l2_loss:
        avg_l2_loss = np.mean(all_l2_loss)
        avg_l1_loss = np.mean(all_l1_loss)
        avg_l0_norm = np.mean(all_l0_norm)
        avg_cossim = np.mean(all_cossim)
        avg_l2_ratio = np.mean(all_l2_ratio)
        avg_frac_var_explained = np.mean(all_frac_variance_explained)
        num_ever_active_features = (total_active_features > 0).sum().item()
        
        print(f"Avg L2 Loss: {avg_l2_loss:.4f}")
        print(f"Avg L1 Loss (Sparsity): {avg_l1_loss:.4f}")
        print(f"Avg L0 Norm (Active Features per Token): {avg_l0_norm:.4f}")
        print(f"Avg Cosine Similarity: {avg_cossim:.4f}")
        print(f"Avg L2 Ratio (||x_hat|| / ||x||): {avg_l2_ratio:.4f}")
        print(f"Avg Fraction of Variance Explained: {avg_frac_var_explained:.4f}")
        print(f"Number of features ever active (sum > 0): {num_ever_active_features} / {ae.dict_size}")
        
        # Populate stats dictionary
        stats = {
            "l2_loss": avg_l2_loss,
            "l1_loss": avg_l1_loss,
            "l0_norm": avg_l0_norm,
            "cosine_similarity": avg_cossim,
            "l2_ratio": avg_l2_ratio,
            "frac_variance_explained": avg_frac_var_explained,
            "num_active_features": num_ever_active_features,
            "dict_size": ae.dict_size,
            "feature_usage_fraction": num_ever_active_features / ae.dict_size
        }
        
    # Clean up
    del token_iterator
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nEvaluation complete.")
    return stats

# %%
n_samples = 4096
batch_size = 16
n_batches = n_samples // batch_size
max_length = 1024

submodule = model.model.layers[19]

from custom_generator import hf_chat_dataset_to_generator, hf_dataset_to_generator
chat_gen = hf_chat_dataset_to_generator(dataset_name="lmsys/lmsys-chat-1m", tokenizer=tokenizer, 
                                        model_name=model_name, split="train", streaming=True,
                                        remove_system_prompt_p=0.8, include_bos=False)

pt_gen = hf_dataset_to_generator(dataset_name="monology/pile-uncopyrighted", split="train", streaming=True)

print("Evaluating chat dataset")
stats = evaluate(chat_gen, batch_size=batch_size, n_batches=n_batches, max_length=max_length, 
                model=model, tokenizer=tokenizer, ae=ae, module=submodule)

print("Evaluating pt dataset")
stats = evaluate(pt_gen, batch_size=batch_size, n_batches=n_batches, max_length=max_length, 
                model=model, tokenizer=tokenizer, ae=ae, module=submodule)

# %%
