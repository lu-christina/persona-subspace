import os
import torch as t
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Literal, Any, Dict, Union
import datetime
import einops
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

class HookManager:
    """Manages hooks and their cleanup"""
    def __init__(self):
        self.hooks = []
    
    def add_hook(self, hook):
        self.hooks.append(hook)
    
    def remove_all(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def get_model_layers(model: Union[AutoModelForCausalLM, PeftModel]) -> Tuple[List[Any], Any, int, bool]:
    """
    Get model layers, embedding layer, count, and whether it's fine-tuned.
    
    Returns:
        Tuple of (layers, embed_layer, num_layers, is_finetuned)
    """
    is_ft = isinstance(model, PeftModel)
    
    if is_ft:
        # For PEFT models
        base_model = model.base_model.model
        if hasattr(base_model, 'model'):  # LlamaForCausalLM structure
            layers = base_model.model.layers
            embed = base_model.model.embed_tokens
        else:
            layers = base_model.layers
            embed = base_model.embed_tokens
    else:
        # For base models
        if hasattr(model, 'model'):  # LlamaForCausalLM structure
            layers = model.model.layers
            embed = model.model.embed_tokens
        else:
            if hasattr(model, 'layers'):
                layers = model.layers
                embed = model.embed_tokens
            elif hasattr(model, 'transformer'):
                #gpt2
                layers = model.transformer.h
                embed = model.transformer.wte
            else:
                raise NotImplementedError
    
    num_layers = len(layers)
    return layers, embed, num_layers, is_ft


def extract_difference_vectors_hooks(
    formatted_toks: dict,
    model: Union[AutoModelForCausalLM, PeftModel],
    token_position: int = -2,
    save_path: Optional[str] = None
) -> t.Tensor:
    """
    Extract difference vectors using PyTorch hooks instead of nnsight.
    
    Args:
        formatted_toks: Tokenized input with prompts (batch will be split in half)
        model: Model to extract from
        token_position: Token position to extract activations from (default -2)
        save_path: Optional path to save the tensor
        
    Returns:
        Tensor of shape (num_layers, d_model) containing mean difference vectors
    """
    batch_size = formatted_toks['input_ids'].shape[0]
    assert batch_size % 2 == 0, f"Expected even batch size for splitting, got {batch_size}"
    
    half_size = batch_size // 2
    layers, _, num_layers, _ = get_model_layers(model)
    
    # Storage for activations
    layer_activations = {}
    hook_manager = HookManager()
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Extract activation at token_position
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store activation at specified token position
            layer_activations[layer_idx] = hidden_states[:, token_position, :].detach()
        return hook_fn
    
    # Register hooks
    for i in range(num_layers):
        hook = layers[i].register_forward_hook(make_hook(i))
        hook_manager.add_hook(hook)
    
    # Forward pass
    with t.no_grad():
        model(**formatted_toks)
    
    # Remove hooks
    hook_manager.remove_all()
    
    # Compute differences
    layer_diffs = []
    for i in range(num_layers):
        acts = layer_activations[i]
        first_half_acts = acts[:half_size]
        second_half_acts = acts[half_size:]
        
        first_half_mean = first_half_acts.mean(dim=0)
        second_half_mean = second_half_acts.mean(dim=0)
        diff = second_half_mean - first_half_mean
        layer_diffs.append(diff)
    
    difference_tensor = t.stack(layer_diffs, dim=0)
    
    if save_path:
        t.save(difference_tensor, save_path)
        print(f"Saved difference vectors to {save_path}")
    
    return difference_tensor


def create_user_token_mask(
    prompt_batch: List[str],
    formatted_tokens: dict,
    system_prompt: str,
    tokenizer: AutoTokenizer
) -> t.Tensor:
    """
    Create a mask indicating which tokens correspond to user content.
    
    Args:
        prompt_batch: List of original user prompts
        formatted_tokens: Tokenized batch with chat template applied
        system_prompt: System prompt used in formatting
        tokenizer: Tokenizer used for encoding
        
    Returns:
        Boolean tensor of shape (batch_size, seq_len) where True indicates user tokens
    """
    batch_size = formatted_tokens['input_ids'].shape[0]
    seq_len = formatted_tokens['input_ids'].shape[1]
    
    mask = t.zeros((batch_size, seq_len), dtype=t.bool, device=formatted_tokens['input_ids'].device)
    
    for i, prompt in enumerate(prompt_batch):
        # Tokenize just the user content
        user_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        # Find where these tokens appear in the full sequence
        full_tokens = formatted_tokens['input_ids'][i].tolist()
        
        # Find the user token subsequence
        user_start = None
        for j in range(len(full_tokens) - len(user_tokens) + 1):
            if full_tokens[j:j+len(user_tokens)] == user_tokens:
                user_start = j
                break
        
        if user_start is not None:
            user_end = user_start + len(user_tokens)
            mask[i, user_start:user_end] = True
        else:
            raise ValueError(f"Could not find exact user token match for prompt {i}")
    
    return mask


def prepare_steering_vectors(
    steering_vectors: Optional[Dict[t.Tensor, float]],
    layer_to_steer: Union[int, Literal['all'], List[int]],
    d_model: int,
    num_layers: int
) -> Tuple[t.Tensor, Optional[List[t.Tensor]]]:
    """
    Prepare steering vectors for application.
    
    Returns:
        Tuple of (total_steering, steering_vec_list)
    """
    if steering_vectors:
        # Combine all steering vectors
        first_vector, first_multiplier = next(iter(steering_vectors.items()))
        total_steering = first_vector * first_multiplier
        
        for vector, multiplier in list(steering_vectors.items())[1:]:
            total_steering = total_steering + vector * multiplier
    else:
        total_steering = t.zeros(d_model, device="cuda")
    
    # Prepare vector list for multi-layer steering
    steering_vec_list = None
    if layer_to_steer == 'all' or isinstance(layer_to_steer, list):
        assert total_steering.shape == (num_layers, d_model), f"Expected shape ({num_layers}, {d_model}), got {total_steering.shape}"
        steering_vec_list = t.unbind(total_steering, dim=0)
    
    return total_steering, steering_vec_list


class SteeringHookManager:
    """Manages steering hooks during generation"""
    def __init__(self, model, layers, layer_to_steer, steering_vectors, d_model):
        self.model = model
        self.layers = layers
        self.layer_to_steer = layer_to_steer
        self.steering_vectors = steering_vectors
        self.d_model = d_model
        self.hooks = []
        self.first_pass_mask = None
        self.is_first_forward = True
        self.thinking_mask = None  # Track which sequences have seen periods
    
    def create_steering_hook(self, layer_idx, steering_vector):
        """Create a hook that applies steering to a specific layer"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            
            # Determine which mask to use
            if self.is_first_forward and self.first_pass_mask is not None:
                # First forward pass with user mask
                # current_mask shape: (batch, original_seq_len)
                # hidden_states shape: (batch, seq_len, d_model)
                if seq_len == self.first_pass_mask.shape[1]:
                    # Full sequence pass
                    mask_expanded = einops.repeat(self.first_pass_mask, 'batch seq -> batch seq d_model', d_model=hidden_states.shape[-1])
                elif seq_len == 1:
                    self.is_first_forward = False
                    mask_expanded = t.zeros_like(hidden_states, dtype=t.bool)
                else:
                    raise ValueError(f"Sequence length mismatch: {seq_len} != {self.first_pass_mask.shape[1]}")
                    mask_expanded = t.ones_like(hidden_states, dtype=t.bool)
            elif self.thinking_mask is not None:
                # Subsequent passes with thinking mask
                # thinking_mask shape: (batch,)
                # hidden_states shape: (batch, 1, d_model) for new tokens
                # Expand thinking mask to match hidden states shape
                mask_expanded = einops.repeat(self.thinking_mask, 'batch -> batch seq d_model', seq=seq_len, d_model=hidden_states.shape[-1])
            else:
                # No mask active
                mask_expanded = t.ones_like(hidden_states, dtype=t.bool)
            
            # Apply steering
            steering_expanded = einops.repeat(steering_vector, 'd_model -> batch seq d_model', batch=batch_size, seq=seq_len)
            modified_hidden = hidden_states + mask_expanded * steering_expanded
            
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            else:
                return modified_hidden
        
        return hook_fn
    
    def add_steering_hooks(self, total_steering, steering_vec_list=None):
        """Add steering hooks to specified layers"""
        self.remove_hooks()  # Clean up any existing hooks
        
        if self.layer_to_steer == 'all':
            for i, vec in enumerate(steering_vec_list):
                hook = self.layers[i].register_forward_hook(self.create_steering_hook(i, vec))
                self.hooks.append(hook)
        elif isinstance(self.layer_to_steer, list):
            for i in self.layer_to_steer:
                vec = steering_vec_list[i] if steering_vec_list else total_steering
                hook = self.layers[i].register_forward_hook(self.create_steering_hook(i, vec))
                self.hooks.append(hook)
        else:  # Single layer
            hook = self.layers[self.layer_to_steer].register_forward_hook(
                self.create_steering_hook(self.layer_to_steer, total_steering)
            )
            self.hooks.append(hook)
    
    def update_mask(self, new_mask):
        """Update the current mask"""
        self.first_pass_mask = new_mask
    
    def update_thinking_mask(self, new_thinking_mask):
        """Update thinking mask"""
        self.thinking_mask = new_thinking_mask
        
    def set_first_forward(self, is_first):
        """Set whether this is the first forward pass"""
        self.is_first_forward = is_first
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def steer_and_generate(
    prompt_list: List[str],
    model: Union[AutoModelForCausalLM, PeftModel],
    tokenizer: AutoTokenizer,
    steering_vectors: Optional[Dict[t.Tensor, float]] = None,
    batch_size: int = 4,
    max_new_tokens: int = 10000,
    temperature: float = 0.6,
    layer_to_steer: Union[int, Literal['all'], List[int]] = 9,
    d_model: int = 8192,
    system_prompt: str = "detailed thinking on",
    steer_on_user: bool = True,
    steer_on_thinking: bool = True
) -> Tuple[List[str], List[str], List[Any], List[Any]]:
    """
    Generate steered responses using PyTorch hooks.
    
    Returns:
        Tuple of (full_responses, model_only_responses, tok_batches, out_steered_list)
    """
    layers, embed, num_layers, _ = get_model_layers(model)
    if layer_to_steer == 'all' or isinstance(layer_to_steer, list):
        assert next(iter(steering_vectors.keys())).shape[0] == num_layers, f"Expected {num_layers} steering vectors, got {len(steering_vectors.keys())}"
    
    total_steering, steering_vec_list = prepare_steering_vectors(
        steering_vectors, layer_to_steer, d_model, num_layers
    )
    
    # Format prompts
    formatted_string_list = []
    for p in prompt_list:
        question_string = tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_string_list.append(question_string)
    
    # Create batches
    tok_batches = []
    prompt_batches = []
    for i in range(0, len(formatted_string_list), batch_size):
        batch_strings = formatted_string_list[i:i+batch_size]
        batch_prompts = prompt_list[i:i+batch_size]
        
        tok_batch = tokenizer(
            batch_strings, 
            add_special_tokens=False, 
            return_tensors="pt", 
            padding=True
        ).to("cuda")
        
        tok_batches.append(tok_batch)
        prompt_batches.append(batch_prompts)
    
    full_responses = []
    model_only_responses = []
    out_steered_list = []
    
    for tok_batch, prompt_batch in tqdm(zip(tok_batches, prompt_batches), total=len(tok_batches)):
        current_batch_size = tok_batch['input_ids'].shape[0]
        
        # Create user token mask if needed
        user_mask = None
        if steering_vectors is not None and steer_on_user:
            user_mask = create_user_token_mask(prompt_batch, tok_batch, system_prompt, tokenizer)
        
        # Initialize steering hook manager
        hook_manager = SteeringHookManager(model, layers, layer_to_steer, steering_vectors, d_model)
        
        # Generate with steering
        if steering_vectors is None:
            # No steering, just generate normally
            with t.no_grad():
                outputs = model.generate(
                    **tok_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id
                )
            out_steered = outputs
        
        elif steer_on_user and not steer_on_thinking:
            # Simple case: only steer on user tokens
            hook_manager.first_pass_mask = user_mask
            hook_manager.add_steering_hooks(total_steering, steering_vec_list)
            
            with t.no_grad():
                outputs = model.generate(
                    **tok_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            hook_manager.remove_hooks()
            out_steered = outputs
        
        else:
            # Complex case: thinking steering (with or without user steering)
            # Initialize hook manager with user mask if applicable
            if steer_on_user and user_mask is not None:
                hook_manager.first_pass_mask = user_mask
            else:
                hook_manager.first_pass_mask = None
            
            # Track which sequences have seen a period (batch,) shape
            has_seen_period = t.zeros(current_batch_size, dtype=t.bool, device="cuda")
            
            # Add steering hooks
            hook_manager.add_steering_hooks(total_steering, steering_vec_list)
            
            # Manual generation loop
            full_input_ids = tok_batch['input_ids']
            attention_mask = tok_batch.get('attention_mask', t.ones_like(full_input_ids))
            
            period_token_id = 13
            past_key_values = None
            
            with t.no_grad():
                for step in range(max_new_tokens):
                    if past_key_values is None:
                        # First forward pass over the full prompt
                        outputs = model(
                            input_ids=full_input_ids,
                            attention_mask=attention_mask,
                            use_cache=True
                        )
                        logits = outputs.logits[:, -1:, :]  # Keep batch dim
                    else:
                        # For cached generation, we need to pass only new tokens
                        # Some models expect attention_mask to match input_ids length
                        outputs = model(
                            input_ids=next_tokens,
                            attention_mask=t.ones((current_batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device),
                            use_cache=True,
                            past_key_values=past_key_values
                        )
                        logits = outputs.logits
                    
                    past_key_values = outputs.past_key_values
                    
                    # Sample next token
                    if temperature > 0:
                        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                        next_tokens = t.multinomial(probs, num_samples=1)
                    else:
                        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    
                    # After first forward, we're in single-token mode
                    if step == 0:
                        hook_manager.set_first_forward(False)
                    
                    # Update period tracking BEFORE next forward pass
                    if steer_on_thinking:
                        is_period = (next_tokens.squeeze(-1) == period_token_id)
                        has_seen_period = t.logical_or(has_seen_period, is_period)
                        hook_manager.update_thinking_mask(has_seen_period)
                    
                    # Append to sequence
                    full_input_ids = t.cat([full_input_ids, next_tokens], dim=1)
                    attention_mask = t.cat([
                        attention_mask,
                        t.ones((current_batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    ], dim=1)
                    
                    # Check for EOS
                    if (next_tokens == tokenizer.eos_token_id).all():
                        break
            
            out_steered = full_input_ids
            hook_manager.remove_hooks()
        
        out_steered_list.append(out_steered)
        
        # Decode responses
        full_decoded = tokenizer.batch_decode(out_steered, skip_special_tokens=True)
        full_responses.extend(full_decoded)
        
        # Decode model-only responses
        model_only_decoded = []
        for i, full_output in enumerate(out_steered):
            input_length = tok_batch['input_ids'][i].shape[0]
            model_only_output = tokenizer.decode(full_output[input_length:], skip_special_tokens=True)
            model_only_decoded.append(model_only_output)
        
        model_only_responses.extend(model_only_decoded)
        t.cuda.empty_cache()
    
    return full_responses, model_only_responses, tok_batches, out_steered_list

# Re-export key functions with same names as original
extract_difference_vectors = extract_difference_vectors_hooks
