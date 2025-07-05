"""
Library for analyzing universal features across different models and configurations.
"""

import json
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from dictionary_learning.utils import load_dictionary
from tqdm.auto import tqdm


class StopForward(Exception):
    """Exception to stop forward pass after target layer."""
    pass


class UniversalFeatureAnalyzer:
    """Analyzer for finding universally active features across prompts."""
    
    def __init__(self, model_type: str, sae_layer: int, token_type: str, sae_trainer: int = 1):
        """
        Initialize the analyzer with model configuration.
        
        Args:
            model_type: "qwen" or "llama"
            sae_layer: SAE layer number (e.g., 11, 15)
            token_type: Token type for extraction
            sae_trainer: SAE trainer number (default: 1)
        """
        self.model_type = model_type
        self.sae_layer = sae_layer
        self.token_type = token_type
        self.sae_trainer = sae_trainer
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure model-specific settings
        self._configure_model_settings()
        
        # Processing parameters
        self.batch_size = 8
        self.max_length = 512
        self.feature_dashboard_base_url = "https://completely-touched-platypus.ngrok-free.app/"
        
    def _configure_model_settings(self):
        """Configure model-specific settings."""
        if self.model_type == "qwen":
            self.model_name = "Qwen/Qwen2.5-7B-Instruct"
            self.sae_release = "andyrdt/saes-qwen2.5-7b-instruct"
            self.assistant_header = "<|im_start|>assistant"
            self.token_offsets = {"asst": -1, "newline": 0}
            self.sae_base_path = "/workspace/sae/qwen-2.5-7b-instruct/saes"
            
        elif self.model_type == "llama":
            self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
            self.sae_release = "andyrdt/saes-llama-3.1-8b-instruct"
            self.assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
            self.token_offsets = {"asst": -2, "endheader": -1, "newline": 0}
            self.sae_base_path = "/workspace/sae/llama-3.1-8b-instruct/saes"
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'qwen' or 'llama'")
        
        # Validate token type
        if self.token_type not in self.token_offsets:
            raise ValueError(f"token_type '{self.token_type}' not available for {self.model_type}. "
                           f"Available: {list(self.token_offsets.keys())}")
        
        self.token_offset = self.token_offsets[self.token_type]
        self.sae_path = f"{self.sae_base_path}/resid_post_layer_{self.sae_layer}/trainer_{self.sae_trainer}"
        
    def load_prompts(self, prompts_path: str = "./prompts") -> pd.DataFrame:
        """Load prompts from JSONL files."""
        prompts_df = pd.DataFrame()
        for file in os.listdir(prompts_path):
            if file.endswith(".jsonl"):
                with open(os.path.join(prompts_path, file), 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        new_row = pd.DataFrame({
                            'prompt': [data['content']], 
                            'label': [data['label']]
                        })
                        prompts_df = pd.concat([prompts_df, new_row], ignore_index=True)
        return prompts_df
        
    @torch.no_grad()
    def extract_activations(self, prompts: List[str], tokenizer, model) -> torch.Tensor:
        """Extract activations from specified layer for given prompts."""
        all_activations = []
        
        # Get target layer
        target_layer = model.model.layers[self.sae_layer]
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Processing batches"):
            batch_prompts = prompts[i:i+self.batch_size]
            
            # Format prompts as chat messages
            formatted_prompts = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted_prompt)
            
            # Tokenize batch
            batch_inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to device
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            
            # Hook to capture activations
            activations = None
            
            def hook_fn(module, input, output):
                nonlocal activations
                activations = output[0] if isinstance(output, tuple) else output
                raise StopForward()
            
            # Register hook
            handle = target_layer.register_forward_hook(hook_fn)
            
            try:
                _ = model(**batch_inputs)
            except StopForward:
                pass
            finally:
                handle.remove()
            
            # Extract assistant token positions
            batch_activations = []
            for j, formatted_prompt in enumerate(formatted_prompts):
                # Get attention mask for this item
                attention_mask = batch_inputs["attention_mask"][j]
                
                # Find assistant header position
                assistant_tokens = tokenizer.encode(self.assistant_header, add_special_tokens=False)
                input_ids = batch_inputs["input_ids"][j]
                
                # Find where assistant section starts
                assistant_pos = None
                for k in range(len(input_ids) - len(assistant_tokens) + 1):
                    if torch.equal(input_ids[k:k+len(assistant_tokens)], torch.tensor(assistant_tokens).to(self.device)):
                        assistant_pos = k + len(assistant_tokens) + self.token_offset
                        break
                
                if assistant_pos is None:
                    # Fallback to last non-padding token
                    assistant_pos = attention_mask.sum().item() - 1
                
                # Ensure position is within bounds
                max_pos = attention_mask.sum().item() - 1
                assistant_pos = min(assistant_pos, max_pos)
                assistant_pos = max(assistant_pos, 0)
                
                # Extract activation at assistant position
                assistant_activation = activations[j, assistant_pos, :]
                batch_activations.append(assistant_activation.cpu())
            
            all_activations.extend(batch_activations)
        
        return torch.stack(all_activations, dim=0)
        
    @torch.no_grad()
    def get_sae_features(self, activations: torch.Tensor, sae) -> torch.Tensor:
        """Apply SAE to get feature activations."""
        activations = activations.to(self.device)
        
        feature_activations = []
        
        for i in range(0, activations.shape[0], self.batch_size):
            batch = activations[i:i+self.batch_size]
            features = sae.encode(batch)
            feature_activations.append(features.cpu())
        
        return torch.cat(feature_activations, dim=0)
        
    @torch.no_grad()
    def find_universally_active_features(self, features: torch.Tensor, activation_threshold: float = 0.01, 
                                       prompt_threshold: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find features that are active for a specified percentage of prompts.
        
        Args:
            features: Feature activations tensor of shape [num_prompts, num_features]
            activation_threshold: Minimum activation value to consider a feature "active"
            prompt_threshold: Minimum percentage of prompts (0.0 to 1.0) that must have the feature active
        
        Returns:
            universal_features: Indices of features that are active for at least prompt_threshold fraction of prompts
            universal_activations: Mean activation values for universal features (only averaging active prompts)
            num_active_prompts: Number of prompts each universal feature is active on
        """
        # Check which features are active (above threshold) for each prompt
        active_features = features > activation_threshold
        
        # Count how many prompts each feature is active for
        num_active_prompts_all = torch.sum(active_features, dim=0)
        
        # Calculate the minimum number of prompts required
        min_prompts_required = int(features.shape[0] * prompt_threshold)
        
        # Find features that are active for at least the required number of prompts
        universal_mask = num_active_prompts_all >= min_prompts_required
        universal_features = torch.where(universal_mask)[0]
        
        # Calculate mean activation values only for active prompts
        universal_activations = []
        for feature_idx in universal_features:
            # Get activations for this feature
            feature_activations = features[:, feature_idx]
            # Get mask for active prompts (above threshold)
            active_mask = feature_activations > activation_threshold
            # Calculate mean only for active prompts
            if active_mask.sum() > 0:
                mean_active = feature_activations[active_mask].mean()
            else:
                mean_active = 0.0
            universal_activations.append(mean_active)
        
        universal_activations = torch.tensor(universal_activations)
        num_active_prompts = num_active_prompts_all[universal_features]
        
        return universal_features, universal_activations, num_active_prompts
        
    def analyze_universal_features(self, tokenizer, model, sae, prompts_path: str = "./prompts", 
                                 activation_threshold: float = 0.01, prompt_threshold: float = 0.3) -> pd.DataFrame:
        """
        Complete analysis pipeline for finding universal features.
        
        Args:
            tokenizer: Loaded tokenizer
            model: Loaded model
            sae: Loaded SAE
            prompts_path: Path to prompts directory
            activation_threshold: Minimum activation value to consider a feature "active"
            prompt_threshold: Minimum percentage of prompts that must have the feature active
            
        Returns:
            DataFrame with universal features and their statistics
        """
        # Load prompts
        prompts_df = self.load_prompts(prompts_path)
        print(f"Loaded {len(prompts_df)} prompts")
        
        # Extract activations
        print("Extracting activations...")
        activations = self.extract_activations(prompts_df['prompt'].tolist(), tokenizer, model)
        
        # Get SAE features
        print("Computing SAE features...")
        features = self.get_sae_features(activations, sae)
        
        # Find universal features
        print(f"Finding features active on at least {prompt_threshold*100:.1f}% of prompts...")
        universal_features, universal_activations, num_active_prompts = self.find_universally_active_features(
            features, activation_threshold, prompt_threshold
        )
        
        # Create results DataFrame
        results = []
        source = f"{self.model_type}_trainer{self.sae_trainer}_layer{self.sae_layer}_{self.token_type}"
        
        for i, feature_idx in enumerate(universal_features):
            feature_id = feature_idx.item()
            feature_activations = features[:, feature_idx]
            
            results.append({
                'source': source,
                'feature_id': feature_id,
                'activation_mean': universal_activations[i].item(),
                'activation_max': feature_activations.max().item(),
                'activation_min': feature_activations.min().item(),
                'num_prompts': num_active_prompts[i].item(),
                'chat_desc': '',
                'pt_desc': '',
                'type': '',
                'link': f"{self.feature_dashboard_base_url}?model={self.model_type}&layer={self.sae_layer}&trainer={self.sae_trainer}&fids={feature_id}"
            })
        
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('activation_mean', ascending=False)
        
        print(f"Found {len(results_df)} universal features")
        return results_df


def load_model_components(model_type: str):
    """Load tokenizer and model for a given model type."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "qwen":
        model_name = "Qwen/Qwen2.5-7B-Instruct"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    device_map_value = device.index if device.type == 'cuda' and device.index is not None else str(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device_map_value}
    )
    model.eval()
    
    return tokenizer, model


def load_sae(model_type: str, sae_layer: int, sae_trainer: int = 1):
    """Load SAE for a given model type and layer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "qwen":
        sae_release = "andyrdt/saes-qwen2.5-7b-instruct"
        sae_base_path = "/workspace/sae/qwen-2.5-7b-instruct/saes"
    elif model_type == "llama":
        sae_release = "andyrdt/saes-llama-3.1-8b-instruct"
        sae_base_path = "/workspace/sae/llama-3.1-8b-instruct/saes"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    sae_path = f"{sae_base_path}/resid_post_layer_{sae_layer}/trainer_{sae_trainer}"
    
    # Check if SAE exists locally
    ae_file_path = os.path.join(sae_path, "ae.pt")
    config_file_path = os.path.join(sae_path, "config.json")
    
    if not (os.path.exists(ae_file_path) and os.path.exists(config_file_path)):
        print(f"SAE not found locally, downloading from {sae_release}...")
        os.makedirs(os.path.dirname(ae_file_path), exist_ok=True)
        sae_path_rel = f"resid_post_layer_{sae_layer}/trainer_{sae_trainer}"
        local_dir = sae_base_path
        hf_hub_download(repo_id=sae_release, filename=f"{sae_path_rel}/ae.pt", local_dir=local_dir)
        hf_hub_download(repo_id=sae_release, filename=f"{sae_path_rel}/config.json", local_dir=local_dir)
    
    sae, _ = load_dictionary(sae_path, device=device)
    sae.eval()
    
    return sae


def get_model_combinations():
    """Get all valid model combinations."""
    combinations = []
    
    # Qwen combinations
    for layer in [11, 15]:
        for token_type in ["asst", "newline"]:
            combinations.append(("qwen", layer, token_type))
    
    # Llama combinations  
    for layer in [11, 15]:
        for token_type in ["asst", "endheader", "newline"]:
            combinations.append(("llama", layer, token_type))
    
    return combinations