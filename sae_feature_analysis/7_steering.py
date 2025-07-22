# %% [markdown]
# # Steering models with target features

# %%
import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from sae_lens import SAE

sys.path.append('.')
sys.path.append('..')

from utils.steering_utils import ActivationSteering, create_mean_ablation_steerer, create_multi_layer_mean_ablation_steerer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_float32_matmul_precision('high')

# %%
STEERING_MAGNITUDES = [-50.0, -75.0, -100.0, 50.0, 75.0, 100.0]
N_RUNS_PER_PROMPT = 1

DO_STEERING = True
DO_ABLATION = True
STEERING_LAYER = 20

TARGET_FEATURES = [45426]  # List of feature IDs to analyze
# GROUP_NAME = "transitions"
# READABLE_GROUP_NAME = "Semantic Transitions and Organization"

# df = pd.read_csv(f"./features/{GROUP_NAME}.csv")
# TARGET_FEATURES = df["feature_id"].tolist()




# %%
@dataclass
class ModelConfig:
    """Configuration for model-specific settings"""
    base_model_name: str
    chat_model_name: str
    hf_release: str  # Reference only - actual loading uses saelens_release/sae_id
    assistant_header: str
    token_offsets: Dict[str, int]
    sae_base_path: str
    saelens_release: str  # Template for sae_lens release parameter
    sae_id_template: str  # Template for sae_lens sae_id parameter
    base_url: str  # Base URL for neuronpedia
    
    def get_sae_params(self, sae_layer: int, sae_trainer: str) -> Tuple[str, str]:
        """
        Generate SAE lens release and sae_id parameters.
        
        Args:
            sae_layer: Layer number for the SAE
            sae_trainer: Trainer identifier for the SAE
            
        Returns:
            Tuple of (release, sae_id) for sae_lens.SAE.from_pretrained()
        """
        if self.saelens_release == "llama_scope_lxr_{trainer}":
            release = self.saelens_release.format(trainer=sae_trainer)
            sae_id = self.sae_id_template.format(layer=sae_layer, trainer=sae_trainer)
        elif self.saelens_release == "gemma-scope-9b-pt-res":
            # Parse SAE_TRAINER "131k-l0-34" into components for Gemma
            parts = sae_trainer.split("-")
            width = parts[0]  # "131k"
            l0_value = parts[2]  # "34"
            
            release = self.saelens_release
            sae_id = self.sae_id_template.format(layer=sae_layer, width=width, l0=l0_value)
        elif self.saelens_release == "gemma-scope-9b-pt-res-canonical":
            # Parse SAE_TRAINER "131k-l0-34" into components for Gemma
            parts = sae_trainer.split("-")
            width = parts[0]  # "131k"

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
        sae_id_template="l{layer}r_{trainer}",
        base_url="https://www.neuronpedia.org/llama-3.1-8b/{layer}-llamascope-res-131k"
    ),
    "gemma": ModelConfig(
        base_model_name="google/gemma-2-9b",
        chat_model_name="google/gemma-2-9b-it",
        hf_release="google/gemma-scope-9b-pt-res/layer_{layer}/width_{width}/average_l0_{l0}",
        assistant_header="<start_of_turn>model",
        token_offsets={"model": -1, "newline": 0},
        sae_base_path="/workspace/sae/gemma-2-9b/saes",
        saelens_release="gemma-scope-9b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_{width}/canonical",
        base_url="https://www.neuronpedia.org/gemma-2-9b/{layer}-gemmascope-res-131k"
    )
}

# =============================================================================
# MODEL SELECTION - Change this to switch between models
# =============================================================================
MODEL_TYPE = "gemma"  # Options: "gemma" or "llama"
MODEL_VER = "chat"
SAE_LAYER = 20
SAE_TRAINER = "131k-l0-114"
N_PROMPTS = 1000

# =============================================================================
# CONFIGURATION SETUP
# =============================================================================
if MODEL_TYPE not in MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Available: {list(MODEL_CONFIGS.keys())}")

config = MODEL_CONFIGS[MODEL_TYPE]

# Set model name based on version
if MODEL_VER == "chat":
    MODEL_NAME = config.chat_model_name
elif MODEL_VER == "base":
    MODEL_NAME = config.base_model_name
else:
    raise ValueError(f"Unknown MODEL_VER: {MODEL_VER}. Use 'chat' or 'base'")

# Always use chat model for tokenizer (has chat template)
CHAT_MODEL_NAME = config.chat_model_name

# Set up derived configurations
ASSISTANT_HEADER = config.assistant_header
TOKEN_OFFSETS = config.token_offsets
SAE_BASE_PATH = config.sae_base_path

# =============================================================================
# OUTPUT FILE CONFIGURATION
# =============================================================================
OUTPUT_DIR = f"./results/7_steering/{MODEL_TYPE}_trainer{SAE_TRAINER}_layer{SAE_LAYER}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DERIVED CONFIGURATIONS
# =============================================================================
SAE_PATH = f"{SAE_BASE_PATH}/resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
LAYER_INDEX = SAE_LAYER

# Data paths
PROMPTS_HF = "lmsys/lmsys-chat-1m"
SEED = 42
PROMPTS_PATH = f"/workspace/data/{PROMPTS_HF.split('/')[-1]}/chat_{N_PROMPTS}.jsonl"
os.makedirs(os.path.dirname(PROMPTS_PATH), exist_ok=True)

# Processing parameters
BATCH_SIZE = 32
MAX_LENGTH = 512

# =============================================================================
# SUMMARY
# =============================================================================
print(f"Configuration Summary:")
print(f"  Model Type: {MODEL_TYPE}")
print(f"  Model to load: {MODEL_NAME}")
print(f"  SAE Layer: {SAE_LAYER}, Trainer: {SAE_TRAINER}")
print(f"  Target Features: {TARGET_FEATURES}")

# %% [markdown]
# ## Prompts for inference

# %%
prompts = [
    "What are the symptoms of a stroke?",
    "Can you help me explain my rash?",
    "I have a problem with my TFCC injury.",
    "You need to help me understand my medical bills.",
    "What is a lambda function in Python?",
    "Can you help me debug my code?",
    "I have a problem with my front-end rendering.",
    "You need to help me understand my algorithms homework.",
    "hello",
    "What is the ontological status of human beings?"
]

# %%
def format_as_chat(tokenizer, prompt):
    """Format prompt as a chat message with proper template"""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return formatted_prompt

# %% [markdown]
# ## Load Model and SAE

# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"":0}
)
model.eval()

print(f"Model loaded: {model.__class__.__name__}")
print(f"Model device: {next(model.parameters()).device}")

# %%
def load_sae(config: ModelConfig, sae_path: str, sae_layer: int, sae_trainer: str) -> SAE:
    """
    Unified SAE loading function that handles both Llama and Gemma models.
    
    Args:
        config: ModelConfig object containing model-specific settings
        sae_path: Local path to store/load SAE files
        sae_layer: Layer number for the SAE
        sae_trainer: Trainer identifier for the SAE
    
    Returns:
        SAE: Loaded SAE model
    """
    # Check if SAE file exists locally
    print(f"Loading SAE from {sae_path}")
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
        device="cuda" # Hardcoded because it wants a string
    )
    
    # Save the SAE locally for future use
    sae.save_model(sae_path)
    return sae

# Load SAE using the unified function
sae = load_sae(config, SAE_PATH, SAE_LAYER, SAE_TRAINER)
sae = sae.to(device)  # Move SAE to GPU

print(f"SAE loaded with {sae.cfg.d_sae} features")
print(f"SAE device: {next(sae.parameters()).device}")

# %% [markdown]
# ## Run inference on prompts
# First ask the model prompts by default.
# Then use the activation steerer.
# 

# %%
def generate_text(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7, do_sample=True):
    """Generate text from a prompt with the model"""
    # Format as chat
    formatted_prompt = format_as_chat(tokenizer, prompt)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode only the new tokens
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

# %%
# Extract feature directions from SAE decoder
def get_feature_direction(sae, feature_id):
    """Extract the direction vector for a specific feature from SAE decoder weights"""
    # SAE decoder weights are stored in W_dec
    # Shape: (d_sae, d_model) where d_sae is number of features
    if feature_id >= sae.cfg.d_sae:
        raise ValueError(f"Feature ID {feature_id} >= max features {sae.cfg.d_sae}")
    
    # Get the decoder vector for this feature
    feature_direction = sae.W_dec[feature_id, :]  # Shape: (d_model,)
    
    # Normalize to unit vector (common practice for steering)
    feature_direction = feature_direction / (feature_direction.norm() + 1e-8)
    
    return feature_direction

# Extract directions for all target features
feature_directions = []
for feature_id in TARGET_FEATURES:
    direction = get_feature_direction(sae, feature_id)
    feature_directions.append(direction)
    print(f"Feature {feature_id}: direction shape {direction.shape}, norm {direction.norm():.4f}")


print(f"\nExtracted directions for {len(feature_directions)} features")

del sae
# %%
# Load mean projections and convert to mean activation vectors

if DO_ABLATION:
    mean_activations_path = f"/workspace/sae/gemma-2-9b/mean_activations/gemma_trainer{SAE_TRAINER}_layer{SAE_LAYER}.pt"

    print(f"Loading mean activations from: {mean_activations_path}")
    mean_data = torch.load(mean_activations_path, map_location=device)

    print(f"Mean data keys: {list(mean_data.keys())}")
    print(f"Mean projections shape: {mean_data['mean_activations'].shape}")

    # Convert mean projections (scalars) to mean activation vectors
    mean_activation_vectors = []
    for i, feature_id in enumerate(TARGET_FEATURES):
        # Find the index of this feature in the mean data
        feature_idx = mean_data['feature_ids'].index(feature_id)
        mean_proj_scalar = mean_data['mean_activations'][feature_idx]  # scalar
        
        # Get the corresponding feature direction (already computed)
        feature_direction = feature_directions[i]  # (d_model,)
        
        # Reconstruct mean activation vector: mean_projection * feature_direction
        mean_activation_vector = mean_proj_scalar * feature_direction  # (d_model,)
        mean_activation_vectors.append(mean_activation_vector)
        
        print(f"Feature {feature_id}: mean_proj={mean_proj_scalar:.6f}, "
            f"mean_activation_vector shape={mean_activation_vector.shape}, "
            f"norm={mean_activation_vector.norm():.6f}")

    print(f"\nConverted {len(mean_activation_vectors)} mean activation vectors for mean ablation")

# %%

def run_steering_experiment_optimized(feature_directions, prompts, magnitudes=STEERING_MAGNITUDES, n_runs=N_RUNS_PER_PROMPT, do_steering=DO_STEERING, do_ablation=DO_ABLATION):
    """
    Run steering experiment for a feature across all prompts with minimal recompilations.
    
    This version minimizes PyTorch recompilations by:
    1. Running mean ablation once for all prompts (if enabled)
    2. Running each steering magnitude once for all prompts
    
    Args:
        feature_directions: List of feature directions to analyze
        prompts: List of prompts to test
        magnitudes: List of steering magnitudes to test
        n_runs: Number of times to run each prompt (for variance estimation)
        do_steering: Whether to run steering experiments
        do_ablation: Whether to run mean ablation experiments
    """
    print(f"\n{'='*60}")
    print(f"N_RUNS: {n_runs}")
    print(f"{'='*60}")
    
    results = {}
    
    # Initialize results structure for all prompts
    for prompt in prompts:
        results[prompt] = {
            "steering": {},
            "ablation": {}
        }
    
    if do_ablation:
        print(f"\nMEAN ABLATION")
        print("-" * 40)
       
        try:
            with create_multi_layer_mean_ablation_steerer(
                model=model,
                feature_directions=feature_directions,
                mean_activations=mean_activation_vectors,
                layer_indices=list(range(STEERING_LAYER, model.config.num_hidden_layers)),
            ) as steerer:
                for prompt in prompts:
                    print(f"\nPrompt: {prompt}")
                    
                    # Run N times and collect responses
                    ablation_responses = []
                    for run_idx in range(n_runs):
                        if n_runs > 1:
                            print(f"  Run {run_idx + 1}/{n_runs}")
                        
                        try:
                            response = generate_text(model, tokenizer, prompt)
                            ablation_responses.append(response)
                        except Exception as e:
                            error_msg = f"Error with mean ablation: {str(e)}"
                            ablation_responses.append(error_msg)
                            print(f"ERROR: {error_msg}")
                    
                    results[prompt]["ablation"][f"mean_ablation_{STEERING_LAYER}_end"] = ablation_responses
        except Exception as e:
            error_msg = f"Error with mean ablation steerer: {str(e)}"
            print(f"ERROR: {error_msg}")
            for prompt in prompts:
                results[prompt]["ablation"][f"mean_ablation_{STEERING_LAYER}_end"] = [error_msg] * n_runs


        try:
            with create_mean_ablation_steerer(
                model=model,
                feature_directions=feature_directions,
                mean_activations=mean_activation_vectors,
                layer_indices=STEERING_LAYER,
            ) as steerer:
                for prompt in prompts:
                    print(f"\nPrompt: {prompt}")
                    
                    # Run N times and collect responses
                    ablation_responses = []
                    for run_idx in range(n_runs):
                        if n_runs > 1:
                            print(f"  Run {run_idx + 1}/{n_runs}")
                        
                        try:
                            response = generate_text(model, tokenizer, prompt)
                            ablation_responses.append(response)
                        except Exception as e:
                            error_msg = f"Error with mean ablation: {str(e)}"
                            ablation_responses.append(error_msg)
                            print(f"ERROR: {error_msg}")
                    
                    results[prompt]["ablation"]["mean_ablation"] = ablation_responses
        except Exception as e:
            error_msg = f"Error with mean ablation steerer: {str(e)}"
            print(f"ERROR: {error_msg}")
            for prompt in prompts:
                results[prompt]["ablation"]["mean_ablation"] = [error_msg] * n_runs

    
    
    if do_steering:
        # Run steering experiments - one magnitude at a time for all prompts
        print(f"\nSTEERING EXPERIMENTS - ALL PROMPTS")
        print("-" * 40)
        
        for magnitude in magnitudes:
            print(f"\n{'='*20} Magnitude: {magnitude:+.1f} {'='*20}")
            
            if magnitude == 0.0:
                # Baseline: no steering - run all prompts
                for prompt in prompts:
                    print(f"\nPrompt: {prompt}")
                    
                    # Run N times and collect responses
                    baseline_responses = []
                    for run_idx in range(n_runs):
                        if n_runs > 1:
                            print(f"  Run {run_idx + 1}/{n_runs}")
                        
                        try:
                            response = generate_text(model, tokenizer, prompt)
                            baseline_responses.append(response)
                        except Exception as e:
                            error_msg = f"Error with baseline: {str(e)}"
                            baseline_responses.append(error_msg)
                            print(f"ERROR: {error_msg}")
                    
                    results[prompt]["steering"][magnitude] = baseline_responses
            else:
                # With steering - apply hook once for all prompts at this magnitude
                try:
                    with ActivationSteering(
                        model=model,
                        steering_vectors=feature_directions,
                        coefficients=[magnitude] * len(feature_directions),
                        layer_indices=STEERING_LAYER,
                        intervention_type="addition",
                        positions="all"
                    ) as steerer:
                        for prompt in prompts:
                            print(f"\nPrompt: {prompt}")
                            
                            # Run N times and collect responses
                            steered_responses = []
                            for run_idx in range(n_runs):
                                if n_runs > 1:
                                    print(f"  Run {run_idx + 1}/{n_runs}")
                                
                                try:
                                    response = generate_text(model, tokenizer, prompt)
                                    steered_responses.append(response)
                                    
                                except Exception as e:
                                    error_msg = f"Error generating with steering: {str(e)}"
                                    steered_responses.append(error_msg)
                                    print(f"ERROR: {error_msg}")
                            
                            results[prompt]["steering"][magnitude] = steered_responses
                except Exception as e:
                    error_msg = f"Error with magnitude {magnitude}: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    for prompt in prompts:
                        results[prompt]["steering"][magnitude] = [error_msg] * n_runs
    
    return results

# Run optimized experiments for feature group
all_results = run_steering_experiment_optimized(feature_directions, prompts, n_runs=N_RUNS_PER_PROMPT, do_ablation=DO_ABLATION)

print(f"\n{'='*60}")
print("STEERING + ABLATION EXPERIMENTS COMPLETE")
print(f"{'='*60}")

# %%
def save_results_to_json_group(results, output_dir):
    """Save steering and mean ablation results to separate JSON files per feature"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{GROUP_NAME}.json")
    
    # Load existing data if file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                feature_obj = json.load(f)
            print(f"📂 Loaded existing data for feature group {GROUP_NAME}")
        except Exception as e:
            print(f"⚠️  Error loading existing file for feature group {GROUP_NAME}: {e}")
            feature_obj = {
                "feature_id": TARGET_FEATURES,
                "group_name": GROUP_NAME,
                "readable_group_name": READABLE_GROUP_NAME,
                "metadata": {
                    "model_name": MODEL_NAME,
                    "model_type": MODEL_TYPE,
                    "sae_layer": SAE_LAYER,
                    "sae_trainer": SAE_TRAINER
                },
                "results": {}
            }
    else:
        feature_obj = {
            "feature_id": TARGET_FEATURES,
            "group_name": GROUP_NAME,
            "readable_group_name": READABLE_GROUP_NAME,
            "metadata": {
                "model_name": MODEL_NAME,
                "model_type": MODEL_TYPE,
                "sae_layer": SAE_LAYER,
                "sae_trainer": SAE_TRAINER
            },
            "results": {}
        }
        print(f"🆕 Creating new file for feature group {GROUP_NAME}")
    
    # Merge prompt results
    for prompt, prompt_results in results.items():
        # Initialize prompt entry if it doesn't exist
        if prompt not in feature_obj["results"]:
            feature_obj["results"][prompt] = {
                "steering": {},
                "ablation": {},
            }
        
        # Handle steering results - merge lists
        if "steering" in prompt_results:
            for magnitude, new_responses in prompt_results["steering"].items():
                magnitude_str = str(magnitude)
                
                # Initialize if doesn't exist
                if magnitude_str not in feature_obj["results"][prompt]["steering"]:
                    feature_obj["results"][prompt]["steering"][magnitude_str] = []
                
                # Convert existing single response to list if needed (backward compatibility)
                if not isinstance(feature_obj["results"][prompt]["steering"][magnitude_str], list):
                    feature_obj["results"][prompt]["steering"][magnitude_str] = [feature_obj["results"][prompt]["steering"][magnitude_str]]
                
                # Merge lists
                if isinstance(new_responses, list):
                    feature_obj["results"][prompt]["steering"][magnitude_str].extend(new_responses)
                else:
                    feature_obj["results"][prompt]["steering"][magnitude_str].append(new_responses)
        
        # Handle ablation results - merge lists (for backward compatibility)
        if "ablation" in prompt_results:
            for ablation_type, new_responses in prompt_results["ablation"].items():
                
                # Initialize if doesn't exist
                if ablation_type not in feature_obj["results"][prompt]["ablation"]:
                    feature_obj["results"][prompt]["ablation"][ablation_type] = []
                
                # Convert existing single response to list if needed (backward compatibility)
                if not isinstance(feature_obj["results"][prompt]["ablation"][ablation_type], list):
                    feature_obj["results"][prompt]["ablation"][ablation_type] = [feature_obj["results"][prompt]["ablation"][ablation_type]]
                
                # Merge lists
                if isinstance(new_responses, list):
                    feature_obj["results"][prompt]["ablation"][ablation_type].extend(new_responses)
                else:
                    feature_obj["results"][prompt]["ablation"][ablation_type].append(new_responses)
        
    # Save the feature to its own JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_obj, f, indent=2, ensure_ascii=False)

def save_results_to_json_feature(results, output_dir, feature_id):
    """Save steering and mean ablation results to separate JSON files per feature"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{feature_id}.json")
    
    # Load existing data if file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                feature_obj = json.load(f)
            print(f"📂 Loaded existing data for feature group {feature_id}")
        except Exception as e:
            print(f"⚠️  Error loading existing file for feature group {feature_id}: {e}")
            feature_obj = {
                "feature_id": feature_id,
                "metadata": {
                    "model_name": MODEL_NAME,
                    "model_type": MODEL_TYPE,
                    "sae_layer": SAE_LAYER,
                    "sae_trainer": SAE_TRAINER
                },
                "results": {}
            }
    else:
        feature_obj = {
            "feature_id": TARGET_FEATURES,
            "metadata": {
                "model_name": MODEL_NAME,
                "model_type": MODEL_TYPE,
                "sae_layer": SAE_LAYER,
                "sae_trainer": SAE_TRAINER
            },
            "results": {}
        }
        print(f"🆕 Creating new file for feature group {feature_id}")
    
    # Merge prompt results
    for prompt, prompt_results in results.items():
        # Initialize prompt entry if it doesn't exist
        if prompt not in feature_obj["results"]:
            feature_obj["results"][prompt] = {
                "steering": {},
                "ablation": {}
            }
        
        # Handle steering results - merge lists
        if "steering" in prompt_results:
            for magnitude, new_responses in prompt_results["steering"].items():
                magnitude_str = str(magnitude)
                
                # Initialize if doesn't exist
                if magnitude_str not in feature_obj["results"][prompt]["steering"]:
                    feature_obj["results"][prompt]["steering"][magnitude_str] = []
                
                # Convert existing single response to list if needed (backward compatibility)
                if not isinstance(feature_obj["results"][prompt]["steering"][magnitude_str], list):
                    feature_obj["results"][prompt]["steering"][magnitude_str] = [feature_obj["results"][prompt]["steering"][magnitude_str]]
                
                # Merge lists
                if isinstance(new_responses, list):
                    feature_obj["results"][prompt]["steering"][magnitude_str].extend(new_responses)
                else:
                    feature_obj["results"][prompt]["steering"][magnitude_str].append(new_responses)
        # Handle ablation results - merge lists (for backward compatibility)
        if "ablation" in prompt_results:
            for ablation_type, new_responses in prompt_results["ablation"].items():
                
                # Initialize if doesn't exist
                if ablation_type not in feature_obj["results"][prompt]["ablation"]:
                    feature_obj["results"][prompt]["ablation"][ablation_type] = []
                
                # Convert existing single response to list if needed (backward compatibility)
                if not isinstance(feature_obj["results"][prompt]["ablation"][ablation_type], list):
                    feature_obj["results"][prompt]["ablation"][ablation_type] = [feature_obj["results"][prompt]["ablation"][ablation_type]]
                
                # Merge lists
                if isinstance(new_responses, list):
                    feature_obj["results"][prompt]["ablation"][ablation_type].extend(new_responses)
                else:
                    feature_obj["results"][prompt]["ablation"][ablation_type].append(new_responses)
        
    # Save the feature to its own JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_obj, f, indent=2, ensure_ascii=False)
        
        


# Save results to individual JSON files
if len(TARGET_FEATURES) == 1:
    saved_features = save_results_to_json_feature(all_results, OUTPUT_DIR, TARGET_FEATURES[0])
else:
    saved_features = save_results_to_json_group(all_results, OUTPUT_DIR)


