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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
ae = BatchTopKSAE.from_pretrained(sae_path).to("cuda")
# %%

ae.eval()

# %%

model_path = 'meta-llama/Llama-3.1-8B-Instruct'
model_device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=model_device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# %%
from custom_generator import hf_chat_dataset_to_generator, hf_dataset_to_generator, local_chat_dataset_to_generator, mixed_dataset_generator

lmsys_generator = hf_chat_dataset_to_generator(dataset_name="lmsys/lmsys-chat-1m", tokenizer=tokenizer, model_name=model_path, split="train", streaming=True, remove_system_prompt_p=0.75, include_bos=False)


# %%
txt = None
for batch in lmsys_generator:
    txt = batch
    break
print(txt)
# %%
sae_layer = 19
inputs = tokenizer(txt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[sae_layer+1]
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    # exclude bos
    hidden_states = hidden_states[1:]
    hidden_states_recon = ae.decode(ae.encode(hidden_states))

    hidden_states_norms = hidden_states.norm(dim=-1)
    hidden_states_recon_norms = hidden_states_recon.norm(dim=-1)

    print(hidden_states_norms[:100])
    print(hidden_states_recon_norms[:100])

    print((hidden_states - hidden_states_recon).norm(dim=-1)[:100])



# %%


# %%
