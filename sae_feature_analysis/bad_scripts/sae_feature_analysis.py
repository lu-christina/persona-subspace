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


class SAEFeatureAnalyzer:
    """Base class for SAE-based feature analysis."""
    
    def __init__(self, model_type: str = None, sae_layer: int = None, token_type: str = None, sae_trainer: int = 1):
        """
        Initialize the analyzer with model configuration.
        
        Args:
            model_type: "qwen" or "llama" (None for orchestration-only instances)
            sae_layer: SAE layer number (e.g., 11, 15) (None for orchestration-only instances)
            token_type: Token type for extraction (None for orchestration-only instances)
            sae_trainer: SAE trainer number (default: 1)
        """
        self.model_type = model_type
        self.sae_layer = sae_layer
        self.token_type = token_type
        self.sae_trainer = sae_trainer
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure model-specific settings only if all parameters are provided
        if model_type is not None and sae_layer is not None and token_type is not None:
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
        """
        Load prompts from JSONL files.
        
        Args:
            prompts_path: Either a directory containing .jsonl files, or a path to a specific .jsonl file
            
        Returns:
            DataFrame with 'prompt' and 'label' columns
        """
        prompts_df = pd.DataFrame()
        
        if os.path.isfile(prompts_path):
            # Single file
            if prompts_path.endswith(".jsonl"):
                with open(prompts_path, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        new_row = pd.DataFrame({
                            'prompt': [data['content']], 
                            'label': [data['label']]
                        })
                        prompts_df = pd.concat([prompts_df, new_row], ignore_index=True)
            else:
                raise ValueError(f"File {prompts_path} is not a .jsonl file")
        elif os.path.isdir(prompts_path):
            # Directory - process all .jsonl files
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
        else:
            raise ValueError(f"Prompts path {prompts_path} is neither a file nor a directory")
            
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
    
    def analyze_features(self, features: torch.Tensor, prompts_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Abstract method for analysis-specific logic. Must be implemented by subclasses.
        
        Args:
            features: SAE feature activations tensor
            prompts_df: DataFrame with prompt and label columns
            **kwargs: Analysis-specific parameters
            
        Returns:
            DataFrame with analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze_features method")
    
    def _calculate_feature_statistics(self, features: torch.Tensor, feature_ids: List[int], 
                                    activation_threshold: float = 0.01) -> pd.DataFrame:
        """
        Calculate statistics for specified features.
        
        Args:
            features: SAE feature activations tensor [num_prompts, num_features]
            feature_ids: List of feature IDs to calculate statistics for
            activation_threshold: Minimum activation value to consider a feature "active"
            
        Returns:
            DataFrame with feature statistics
        """
        results = []
        source = f"{self.model_type}_trainer{self.sae_trainer}_layer{self.sae_layer}"
        
        for feature_id in feature_ids:
            if feature_id < features.shape[1]:  # Check if feature exists
                feature_activations = features[:, feature_id]
                
                # Count active prompts (above threshold)
                active_mask = feature_activations > activation_threshold
                num_active_prompts = active_mask.sum().item()
                
                # Calculate mean only for active prompts
                if num_active_prompts > 0:
                    activation_mean = feature_activations[active_mask].mean().item()
                else:
                    activation_mean = 0.0
                
                results.append({
                    'feature_id': feature_id,
                    'activation_mean': activation_mean,
                    'activation_max': feature_activations.max().item(),
                    'activation_min': feature_activations.min().item(),
                    'num_prompts': num_active_prompts,
                    'chat_desc': '',
                    'pt_desc': '',
                    'type': '',
                    'source': source,
                    'token': self.token_type,
                    'link': f"{self.feature_dashboard_base_url}?model={self.model_type}&layer={self.sae_layer}&trainer={self.sae_trainer}&fids={feature_id}"
                })
        
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('activation_mean', ascending=False)
        
        return results_df
    
    def _record_prompt_activations(self, features: torch.Tensor, prompts_df: pd.DataFrame, 
                                 feature_ids: List[int], activation_threshold: float = 0.01, 
                                 output_path: str = None) -> List[Dict]:
        """
        Record prompt activations above threshold for specified features.
        
        Args:
            features: SAE feature activations tensor [num_prompts, num_features]
            prompts_df: DataFrame with prompt and label columns
            feature_ids: List of feature IDs to record activations for
            activation_threshold: Minimum activation value to record
            output_path: Path to save JSONL file (optional)
            
        Returns:
            List of activation records
        """
        prompt_activations = []
        
        for prompt_idx, (prompt, label) in enumerate(zip(prompts_df['prompt'], prompts_df['label'])):
            for feature_id in feature_ids:
                if feature_id < features.shape[1]:  # Check if feature exists
                    activation = features[prompt_idx, feature_id].item()
                    
                    # Record prompt activation if above threshold
                    if activation >= activation_threshold:
                        prompt_activations.append({
                            'model': self.model_type,
                            'layer': self.sae_layer,
                            'feature_id': feature_id,
                            'prompt': prompt,
                            'label': label,
                            'activation': activation,
                            'token_type': self.token_type
                        })
        
        # Save to JSONL file if output path provided
        if output_path and prompt_activations:
            self._save_prompt_activations(prompt_activations, 
                                        f"{self.model_type}_trainer{self.sae_trainer}_layer{self.sae_layer}", 
                                        output_path)
        
        return prompt_activations
    
    def _save_prompt_activations(self, prompt_activations: List[Dict], source: str, output_path: str):
        """
        Save prompt activations to consolidated JSONL file.
        
        Args:
            prompt_activations: List of prompt activation records
            source: Source identifier for the current analysis
            output_path: Path to the results CSV file (used to determine directory and base name)
        """
        # Create filename based on CSV output path
        csv_path = Path(output_path)
        csv_stem = csv_path.stem  # filename without extension
        csv_dir = csv_path.parent
        
        # Create JSONL filename with _activeprompts suffix
        jsonl_filename = f"{csv_stem}_activeprompts.jsonl"
        jsonl_path = csv_dir / jsonl_filename
        
        # Ensure directory exists
        os.makedirs(csv_dir, exist_ok=True)
        
        # Append to JSONL file (create if doesn't exist)
        with open(jsonl_path, 'a') as f:
            for record in prompt_activations:
                f.write(json.dumps(record) + '\n')
        
        print(f"Appended {len(prompt_activations)} prompt activations to {jsonl_path}")
    
    def run_analysis(self, output_path: str, prompts_path: str = "./prompts", **kwargs) -> pd.DataFrame:
        """
        Run complete analysis across all model combinations.
        
        Args:
            output_path: Path to save final results CSV
            prompts_path: Path to prompts directory or specific .jsonl file
            **kwargs: Analysis-specific parameters
            
        Returns:
            Combined DataFrame with all results
        """
        # Load prompts once
        print("Loading prompts...")
        prompts_df = self.load_prompts(prompts_path)
        print(f"Loaded {len(prompts_df)} prompts")
        
        # Get all combinations
        combinations = get_model_combinations()
        print(f"Running analysis for {len(combinations)} combinations...")
        
        # Organize combinations by model type for efficient loading
        combinations_by_model = {}
        for model_type, layer, token_type in combinations:
            if model_type not in combinations_by_model:
                combinations_by_model[model_type] = {}
            if layer not in combinations_by_model[model_type]:
                combinations_by_model[model_type][layer] = []
            combinations_by_model[model_type][layer].append(token_type)
        
        # Collect all results
        all_results = []
        
        # Process each model type
        for model_type in combinations_by_model:
            print(f"\n{'='*60}")
            print(f"LOADING MODEL: {model_type}")
            print(f"{'='*60}")
            
            # Load model components once for this model type
            model_loaded = False
            try:
                tokenizer, model = load_model_components(model_type)
                model_loaded = True
                print(f"✓ Loaded {model_type} model")
            except Exception as e:
                print(f"❌ Failed to load {model_type} model: {str(e)}")
                continue
            
            try:
                # Process each layer for this model
                for layer in combinations_by_model[model_type]:
                    print(f"\n{'='*40}")
                    print(f"LOADING SAE: {model_type} layer {layer}")
                    print(f"{'='*40}")
                    
                    # Load SAE once for this layer
                    sae = None
                    sae_loaded = False
                    try:
                        sae = load_sae(model_type, layer)
                        sae_loaded = True
                        print(f"✓ Loaded {model_type} layer {layer} SAE")
                    except Exception as e:
                        print(f"❌ Failed to load {model_type} layer {layer} SAE: {str(e)}")
                        continue
                    
                    try:
                        # Set up configuration for this model+layer (using first token type to configure)
                        first_token_type = combinations_by_model[model_type][layer][0]
                        self.model_type = model_type
                        self.sae_layer = layer
                        self.token_type = first_token_type
                        self._configure_model_settings()
                        
                        # Extract activations and compute features once per layer
                        print("Extracting activations...")
                        activations = self.extract_activations(prompts_df['prompt'].tolist(), tokenizer, model)
                        
                        print("Computing SAE features...")
                        features = self.get_sae_features(activations, sae)
                        
                        # Process all token types for this model+layer combination
                        for token_type in combinations_by_model[model_type][layer]:
                            print(f"\n--- Processing {model_type} layer {layer} {token_type} ---")
                            
                            try:
                                # Update analyzer configuration for this token type
                                self.model_type = model_type
                                self.sae_layer = layer
                                self.token_type = token_type
                                self._configure_model_settings()
                                
                                # Run analysis-specific logic
                                results_df = self.analyze_features(features, prompts_df, output_path=output_path, **kwargs)
                                
                                if len(results_df) > 0:
                                    all_results.append(results_df)
                                    print(f"✓ Analysis completed: {len(results_df)} results")
                                else:
                                    print("✓ No results found for this combination")
                                    
                            except Exception as e:
                                print(f"❌ Error in analysis for {model_type} layer {layer} {token_type}: {str(e)}")
                                continue
                    
                    finally:
                        # Clean up SAE
                        if sae_loaded and sae is not None:
                            del sae
                            torch.cuda.empty_cache()
                            print(f"✓ Cleaned up {model_type} layer {layer} SAE")
            
            finally:
                # Clean up model
                if model_loaded:
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()
                    print(f"✓ Cleaned up {model_type} model")
        
        # Combine and save results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df = combined_df.sort_values(['source', 'token'], ascending=[True, True])
            
            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            combined_df.to_csv(output_path, index=False)
            
            print(f"\n{'='*60}")
            print(f"ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"Total results: {len(combined_df)}")
            print(f"Results saved to: {output_path}")
            
            # Show breakdown by source
            source_counts = combined_df['source'].value_counts()
            print(f"\nBreakdown by source:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} results")
            
            return combined_df
        else:
            print("\n❌ No results found across any combinations")
            # Create empty file
            empty_df = pd.DataFrame()
            empty_df.to_csv(output_path, index=False)
            return empty_df


class UniversalFeatureAnalyzer(SAEFeatureAnalyzer):
    """Analyzer for finding universally active features across prompts."""
    
    def analyze_features(self, features: torch.Tensor, prompts_df: pd.DataFrame, 
                        activation_threshold: float = 0.01, prompt_threshold: float = 0.3, **kwargs) -> pd.DataFrame:
        """
        Find universally active features.
        
        Args:
            features: SAE feature activations tensor
            prompts_df: DataFrame with prompt and label columns
            activation_threshold: Minimum activation value to consider a feature "active"
            prompt_threshold: Minimum percentage of prompts that must have the feature active
            
        Returns:
            DataFrame with universal features and their statistics
        """
        print(f"Finding features active on at least {prompt_threshold*100:.1f}% of prompts...")
        universal_features, universal_activations, num_active_prompts = self._find_universally_active_features(
            features, activation_threshold, prompt_threshold
        )
        
        # Get feature IDs and calculate statistics
        feature_ids = [feature_idx.item() for feature_idx in universal_features]
        results_df = self._calculate_feature_statistics(features, feature_ids, activation_threshold)
        
        print(f"Found {len(results_df)} universal features")
        return results_df
        
    @torch.no_grad()
    def _find_universally_active_features(self, features: torch.Tensor, activation_threshold: float = 0.01, 
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


class SpecificFeatureAnalyzer(SAEFeatureAnalyzer):
    """Analyzer for getting activations of specific features on given prompts."""
    
    def __init__(self, model_type: str = None, sae_layer: int = None, token_type: str = None, sae_trainer: int = 1, features_csv_path: str = None):
        """
        Initialize the analyzer.
        
        Args:
            model_type: Model type (can be None for run_analysis)
            sae_layer: SAE layer (can be None for run_analysis)  
            token_type: Token type (can be None for run_analysis)
            sae_trainer: SAE trainer number
            features_csv_path: Path to CSV with features to analyze (required for run_analysis)
        """
        self.features_csv_path = features_csv_path
        super().__init__(model_type, sae_layer, token_type, sae_trainer)
    
    def analyze_features(self, features: torch.Tensor, prompts_df: pd.DataFrame, 
                        features_csv_path: str = None, record_prompts: bool = False, 
                        activation_threshold: float = 0.01, output_path: str = None, **kwargs) -> pd.DataFrame:
        """
        Calculate statistics for specific features.
        
        Args:
            features: SAE feature activations tensor
            prompts_df: DataFrame with prompt and label columns
            features_csv_path: Path to CSV file with 'feature_id' and 'source' columns
            record_prompts: Whether to record prompts that activate features to JSONL
            activation_threshold: Minimum activation value to consider feature "active"
            output_path: Path for saving JSONL file (if record_prompts=True)
            
        Returns:
            DataFrame with feature statistics (one row per feature)
        """
        # Use passed path or instance path
        csv_path = features_csv_path or self.features_csv_path
        if not csv_path:
            raise ValueError("features_csv_path must be provided either in __init__ or analyze_features")
            
        # Load features from CSV
        features_df = self._load_features_from_csv(csv_path)
        
        # Filter features that match this analyzer's source AND token
        current_source = f"{self.model_type}_trainer{self.sae_trainer}_layer{self.sae_layer}"
        matching_features = features_df[
            (features_df['source'] == current_source) & 
            (features_df['token'] == self.token_type)
        ]
        
        if len(matching_features) == 0:
            print(f"No features found for source {current_source} and token {self.token_type}")
            return pd.DataFrame()
        
        feature_ids = matching_features['feature_id'].tolist()
        print(f"Found {len(feature_ids)} features for source {current_source} and token {self.token_type}")
        
        # Calculate feature statistics using shared method
        print(f"Calculating statistics for {len(feature_ids)} specific features...")
        results_df = self._calculate_feature_statistics(features, feature_ids, activation_threshold)
        
        # Filter out features that are not active (num_prompts = 0)
        active_features_df = results_df[results_df['num_prompts'] > 0]
        filtered_count = len(results_df) - len(active_features_df)
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} inactive features (0 prompts above threshold)")
        
        # Record prompt activations if requested
        if record_prompts and output_path and len(active_features_df) > 0:
            active_feature_ids = active_features_df['feature_id'].tolist()
            prompt_activations = self._record_prompt_activations(
                features, prompts_df, active_feature_ids, activation_threshold, output_path
            )
            print(f"Recorded {len(prompt_activations)} prompt activations above threshold {activation_threshold}")
        
        print(f"Calculated statistics for {len(active_features_df)} active features")
        return active_features_df
    
    def _load_features_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load features from CSV file.
        
        Args:
            csv_path: Path to CSV file with 'feature_id' and 'source' columns
            
        Returns:
            DataFrame with feature_id and source columns
        """
        features_df = pd.read_csv(csv_path)
        required_columns = ['feature_id', 'source']
        
        for col in required_columns:
            if col not in features_df.columns:
                raise ValueError(f"CSV file must contain '{col}' column. Found columns: {list(features_df.columns)}")
        
        print(f"Loaded {len(features_df)} features from {csv_path}")
        return features_df


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