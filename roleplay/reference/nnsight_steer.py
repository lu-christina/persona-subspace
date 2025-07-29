#%%
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
from mpmath import j1
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from nnsight import LanguageModel
from nnsight.intervention.envoy import Envoy
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Literal, Any, cast
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
import datetime
import einops
import torch.nn.functional as F
load_dotenv()
#%%
def extract_difference_vectors(
    formatted_toks: dict,
    lma: LanguageModel, 
    token_position: int = -2,
    save_path: Optional[str] = None) -> t.Tensor:
    """
    Extract difference vectors between first half and second half of prompts at specified token position.
    
    Args:
        formatted_toks: Tokenized input with prompts (batch will be split in half)
        lma: LanguageModel to trace through (could be lma_base or lma)
        token_position: Token position to extract activations from (default -2)
        save_path: Optional path to save the tensor
        
    Returns:
        Tensor of shape (num_layers, d_model) containing mean difference vectors (second_half - first_half)
        
    Example:
        # Create real and fake prompts (5 real + 5 fake = 10 total)
        real_prompts = [real_p0, real_p1, real_p2, real_p3, real_p4]
        fake_prompts = [fake_p0, fake_p1, fake_p2, fake_p3, fake_p4]
        formatted_strings = [format_input(p, add_generation_prompt=False) for p in real_prompts + fake_prompts]
        formatted_toks = tokenizer(formatted_strings, padding=True, return_tensors="pt").to("cuda")
        
        # Extract difference vectors (fake - real)
        diff_vectors = extract_difference_vectors(formatted_toks, lma_base)
    """
    batch_size = formatted_toks['input_ids'].shape[0]
    assert batch_size % 2 == 0, f"Expected even batch size for splitting, got {batch_size}"
    
    half_size = batch_size // 2
    
    # Check if this is a finetuned model to access layers correctly
    ft = isinstance(lma._model, PeftModel)
    if ft:
        num_layers = len(lma.base_model.model.model.layers)
        layers = lma.base_model.model.model.layers
        print(f"ft model")
    else:
        num_layers = len(lma.model.layers)
        layers = lma.model.layers
        print(f"non ft model")
    layer_diffs = []
    
    with lma.trace(formatted_toks) as tr:
        for i in range(num_layers):
            # Extract activations at token position for all prompts
            layer_acts = layers[i].output[0][:, token_position, :]  # Shape: (batch_size, d_model)
            
            # Split into first half and second half
            first_half_acts = layer_acts[:half_size]   # First half (e.g., real prompts)
            second_half_acts = layer_acts[half_size:]  # Second half (e.g., fake prompts)
            
            # Compute mean difference (second_half - first_half)
            first_half_mean = first_half_acts.mean(dim=0)
            second_half_mean = second_half_acts.mean(dim=0)
            diff = second_half_mean - first_half_mean
            diff = diff.save()
            layer_diffs.append(diff)
    
    # Convert to actual values and stack into tensor of shape (num_layers, d_model)
    layer_diffs = [diff.value.detach() for diff in layer_diffs]
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
        prompt_batch: List of original user prompts for this batch
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
        # Tokenize just the user content separately
        user_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        # Find where these tokens appear in the full sequence
        full_tokens = formatted_tokens['input_ids'][i].tolist()
        
        # Find the user token subsequence in the full sequence
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
            # Fallback: if exact match fails, try to find user content by excluding system parts
            # This handles cases where tokenization might differ slightly
            print(f"Warning: Could not find exact user token match for prompt {i}, using fallback")
            # Apply steering to middle portion as fallback (skip system prompt and generation tokens)
            start_pos = len(full_tokens) // 4  # Rough estimate
            end_pos = len(full_tokens) * 3 // 4
            mask[i, start_pos:end_pos] = True
    
    return mask


def apply_steering_to_layer(
    layer_envoy:Envoy,
    steering_vector: t.Tensor,
    steering_mask: t.Tensor
) -> None:
    """
    Apply steering vector only to user token positions.
    
    Args:
        layer_envoy: Layer envoy object with output attribute
        steering_vector: Steering vector of shape (d_model,)
        steering_mask: Boolean mask of shape (batch, seq_len) indicating user tokens
    """
    # Expand mask to match layer output dimensions
    # assert steering_mask.shape[0] == layer_envoy.output[0].shape[0], f"Batch size mismatch: {steering_mask.shape[0]} != {layer_envoy.output[0].shape[0]}"
    # assert steering_mask.shape[1] == layer_envoy.output[0].shape[1], f"Sequence length mismatch: {steering_mask.shape[1]} != {layer_envoy.output[0].shape[1]}"

    mask_expanded = steering_mask.unsqueeze(-1).expand_as(layer_envoy.output[0])  # (batch, seq_len, d_model)
    
    # Apply steering only where mask is True
    steering_expanded = steering_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    layer_envoy.output[0][:,:,:] = layer_envoy.output[0][:,:,:] + mask_expanded * steering_expanded


def get_model_info(lma: LanguageModel) -> Tuple[List, int, bool, Envoy]:
    """
    Get model layers, number of layers, and whether it's fine-tuned.
    
    Returns:
        Tuple of (layers, num_layers, is_finetuned)
    """
    is_ft = isinstance(lma._model, PeftModel)
    if is_ft:
        layers = lma.base_model.model.model.layers
        embed = lma.base_model.model.model.embed_tokens
        num_layers = len(layers)
    else:
        layers = lma.model.layers
        num_layers = len(layers)
        embed = lma.model.embed_tokens
    
    return layers, num_layers, is_ft, embed


def prepare_steering_vectors(
    steering_vectors: dict[t.Tensor, float] | None,
    layer_to_steer: int | Literal['all'] | List[int],
    d_model: int,
    model_len: int
) -> Tuple[t.Tensor, List[t.Tensor] | None]:
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
        assert total_steering.shape == (model_len, d_model), f"Expected shape ({model_len}, {d_model}), got {total_steering.shape}"
        steering_vec_list = t.unbind(total_steering, dim=0)
    
    return total_steering, steering_vec_list


def steer_and_generate(
    prompt_list: List[str],
    lma: LanguageModel,
    tokenizer: AutoTokenizer,
    steering_vectors: dict[t.Tensor, float] | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 10000,
    temperature: float = 0.6,
    layer_to_steer: int | Literal['all'] | List[int] = 9,
    d_model: int = 8192,
    system_prompt: str = "detailed thinking on",
    steer_on_user: bool = True,
    steer_on_thinking: bool = True
) -> Tuple[List[str], List[str], List[Any], List[Any]]:
    """
    Generate steered responses for a list of prompts with optional user-token-only steering.
    
    Args:
        prompt_list: List of prompts to process
        lma: LanguageModel instance
        tokenizer: AutoTokenizer instance
        steering_vectors: Dict mapping tensors to their multipliers (default: None)
        batch_size: Number of prompts to process in each batch
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for generation
        layer_to_steer: Layer(s) to apply steering to
        d_model: Model dimension
        system_prompt: System prompt to use
        
    Returns:
        Tuple of (full_responses, model_only_responses, tok_batches, out_steered_list)
        - full_responses: Complete decoded outputs including prompts
        - model_only_responses: Only the generated parts (excluding input prompts)
        - tok_batches: List of tokenized batches
        - out_steered_list: List of raw output tensors

    Example steering vector: {difference_vector: 0.5}
    """
    layers, model_len, is_ft, embed = get_model_info(lma)
    total_steering, steering_vec_list = prepare_steering_vectors(
        steering_vectors, layer_to_steer, d_model, model_len
    )

    # Format prompts with chat template
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
            padding=True,
            padding_side="left"
        ).to("cuda")
        
        tok_batches.append(tok_batch)
        prompt_batches.append(batch_prompts)
    # print tokenization to make sure it's right, also check the size of the masks, also steer on only one layer. 
    full_responses = []
    model_only_responses = []
    out_steered_list = []
    
    for tok_batch, prompt_batch in tqdm(zip(tok_batches, prompt_batches), total=len(tok_batches)):
        # Create user token mask if steering is enabled and user-only steering is requested
        user_mask = None
        if steering_vectors is not None and steer_on_user:
            user_mask = create_user_token_mask(prompt_batch, tok_batch, system_prompt, tokenizer)
        
        # Generate with or without steering
        if steering_vectors is None:
            with lma.generate(tok_batch, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id) as gen:
                out_steered = lma.generator.output.save()
        elif steer_on_user and not steer_on_thinking:
            # Simple case: only steer on user tokens, no thinking steering
            with lma.generate(tok_batch, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id) as gen:
                # Apply steering to user tokens only at the beginning
                if layer_to_steer == 'all':
                    for i in range(model_len):
                        apply_steering_to_layer(layers[i], steering_vec_list[i], user_mask)
                elif isinstance(layer_to_steer, list):
                    for i in layer_to_steer:
                        apply_steering_to_layer(layers[i], steering_vec_list[i], user_mask)
                else:  # Single layer
                    apply_steering_to_layer(layers[layer_to_steer], total_steering, user_mask)
                
                out_steered = lma.generator.output.save()
        else:
            # Complex case: need thinking steering (with or without user steering)
            # create a mask for which batch position haven't
            # gotten like the first sentence of thinking tokens done yet.
            mask_for_first_period = t.zeros(tok_batch['input_ids'].shape[0], dtype = t.bool, device = "cuda")

            # First generation phase - generate up to 150 tokens with original logic
            max_phase1_tokens = min(150, max_new_tokens)
            
            with lma.generate(tok_batch, max_new_tokens=max_phase1_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id) as gen:
                # Apply steering to specified layers
                for j in range(max_phase1_tokens):
                    if j == 0:
                        if steer_on_user:
                            if layer_to_steer == 'all':
                                for i in range(model_len):
                                    apply_steering_to_layer(layers[i], steering_vec_list[i], user_mask)
                                    layers[i].next()
                                        
                            elif isinstance(layer_to_steer, list):
                                for i in layer_to_steer:
                                    apply_steering_to_layer(layers[i], steering_vec_list[i], user_mask)
                                    layers[i].next()
                            else:  # Single layer
                                apply_steering_to_layer(layers[layer_to_steer], total_steering, user_mask)
                                layers[layer_to_steer].next()
                        else:
                            # No user steering, just call next
                            if layer_to_steer == 'all':
                                for i in range(model_len):
                                    layers[i].next()  
                            elif isinstance(layer_to_steer, list):
                                for i in layer_to_steer:
                                    layers[i].next()
                            else:  # Single layer
                                layers[layer_to_steer].next()

                    else:
                        if steer_on_thinking:
                            #update mask
                            cur_period = embed.input.squeeze() == 13
                            # assert embed.input.shape[0] == mask_for_first_period.shape[0], f"Batch size mismatch: {embed.input.shape[0]} != {mask_for_first_period.shape[0]}"
                            # assert embed.input.shape[1] == 1
                            mask_for_first_period = t.logical_or(cur_period, mask_for_first_period.detach())
                            #go through each layer, steer, then call next
                            if layer_to_steer == 'all':
                                for i in range(model_len):
                                    apply_steering_to_layer(layers[i], steering_vec_list[i], mask_for_first_period.unsqueeze(-1))
                                    layers[i].next()
                            elif isinstance(layer_to_steer, list):
                                for i in layer_to_steer:
                                    apply_steering_to_layer(layers[i], steering_vec_list[i], mask_for_first_period.unsqueeze(-1))
                                    layers[i].next()
                            else:  # Single layer
                                apply_steering_to_layer(layers[layer_to_steer], total_steering, mask_for_first_period.unsqueeze(-1))
                                layers[layer_to_steer].next()
                    embed.next()
                
                phase1_output = lma.generator.output.save()
            
            # Check if we need to continue generation
            remaining_tokens = max_new_tokens - max_phase1_tokens
            
            if remaining_tokens > 0:
                # Find where we've seen periods in phase 1
                batch_size = phase1_output.shape[0]
                period_token_id = 13
                
                # Create mask for phase 2 - need to track which positions had periods
                phase2_length = phase1_output.shape[1]
                phase2_mask_for_first_period = t.zeros((batch_size, phase2_length), dtype=t.bool, device="cuda")
                
                # Find positions after first period for each sequence
                for b in range(batch_size):
                    period_positions = (phase1_output[b] == period_token_id).nonzero(as_tuple=True)[0]
                    if len(period_positions) > 0:
                        first_period_pos = period_positions[0].item()
                        phase2_mask_for_first_period[b, first_period_pos + 1:] = True
                
                # Create attention mask for phase 2
                phase2_attention_mask = t.ones_like(phase1_output, dtype=t.long, device="cuda")
                if 'attention_mask' in tok_batch:
                    orig_mask_length = tok_batch['attention_mask'].shape[1]
                    phase2_attention_mask[:, :orig_mask_length] = tok_batch['attention_mask']
                
                # Continue generation with phase 2
                phase2_input = {
                    'input_ids': phase1_output,
                    'attention_mask': phase2_attention_mask
                }
                
                with lma.generate(phase2_input, max_new_tokens=remaining_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id) as gen:
                    # Continue with the same pattern using .next()
                    for i in range(remaining_tokens):
                        if i == 0:
                            # Apply initial steering to the existing sequence based on our masks
                            if steer_on_user and user_mask is not None:
                                # Create combined mask for initial steering
                                combined_initial_mask = t.zeros((batch_size, phase2_length), dtype=t.bool, device="cuda")
                                user_mask_length = user_mask.shape[1]
                                combined_initial_mask[:, :user_mask_length] = user_mask
                                combined_initial_mask = t.logical_or(combined_initial_mask, phase2_mask_for_first_period)
                                
                                if layer_to_steer == 'all':
                                    for l in range(model_len):
                                        apply_steering_to_layer(layers[l], steering_vec_list[l], combined_initial_mask)
                                        layers[l].next()
                                elif isinstance(layer_to_steer, list):
                                    for l in layer_to_steer:
                                        apply_steering_to_layer(layers[l], steering_vec_list[l], combined_initial_mask)
                                        layers[l].next()
                                else:
                                    apply_steering_to_layer(layers[layer_to_steer], total_steering, combined_initial_mask)
                                    layers[layer_to_steer].next()
                            else:
                                # Just apply thinking steering
                                if layer_to_steer == 'all':
                                    for l in range(model_len):
                                        apply_steering_to_layer(layers[l], steering_vec_list[l], phase2_mask_for_first_period)
                                        layers[l].next()
                                elif isinstance(layer_to_steer, list):
                                    for l in layer_to_steer:
                                        apply_steering_to_layer(layers[l], steering_vec_list[l], phase2_mask_for_first_period)
                                        layers[l].next()
                                else:
                                    apply_steering_to_layer(layers[layer_to_steer], total_steering, phase2_mask_for_first_period)
                                    layers[layer_to_steer].next()
                        else:
                            # For subsequent tokens, apply thinking steering to all
                            if steer_on_thinking:
                                if layer_to_steer == 'all':
                                    for l in range(model_len):
                                        layers[l].output[0][:,:,:] = layers[l].output[0][:,:,:] + steering_vec_list[l].unsqueeze(0).unsqueeze(0)
                                        layers[l].next()
                                elif isinstance(layer_to_steer, list):
                                    for l in layer_to_steer:
                                        layers[l].output[0][:,:,:] = layers[l].output[0][:,:,:] + steering_vec_list[l].unsqueeze(0).unsqueeze(0)
                                        layers[l].next()
                                else:
                                    layers[layer_to_steer].output[0][:,:,:] = layers[layer_to_steer].output[0][:,:,:] + total_steering.unsqueeze(0).unsqueeze(0)
                                    layers[layer_to_steer].next()
                        embed.next()
                    
                    out_steered = lma.generator.output.save()
            else:
                out_steered = phase1_output
        
        out_steered_list.append(out_steered)
        
        # Decode responses
        full_decoded = tokenizer.batch_decode(out_steered)
        full_decoded = [d.replace(tokenizer.eos_token, '').replace('<|end_of_text|>', '') for d in full_decoded]
        full_responses.extend(full_decoded)
        
        # Decode model-only responses (excluding input prompts)
        model_only_decoded = []
        for i, full_output in enumerate(out_steered):
            input_length = tok_batch['input_ids'][i].shape[0]
            model_only_output = tokenizer.decode(full_output[input_length:])
            model_only_output = model_only_output.replace(tokenizer.eos_token, '').replace('<|end_of_text|>', '')
            model_only_decoded.append(model_only_output)
        
        model_only_responses.extend(model_only_decoded)
        t.cuda.empty_cache()
    
    return full_responses, model_only_responses, tok_batches, out_steered_list