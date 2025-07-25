import torch
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('.')
sys.path.append('..')

class StopForward(Exception):
    """Exception to stop forward pass after target layer."""
    pass

def load_model(model_name):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # More stable than float16
        device_map={"": 0}  # Put everything on GPU 0
    )
    model.eval()
    return model, tokenizer

def format_as_chat(tokenizer, prompt):
    """Format prompt as a chat message with proper template"""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted_prompt


def format_as_chat_swapped(tokenizer, prompt):
    """Format prompt as a chat message with proper template"""
    messages = [{"role": "user", "content": "Hello."}, {"role": "model", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    parts = formatted_prompt.rsplit('model', 1)
    if len(parts) == 2:
        formatted_prompt = 'user'.join(parts)

    return formatted_prompt

def find_newline_position(input_ids, tokenizer, device):
    """Find the position of the newline token in the assistant section"""
    # Try to find '\n\n' token first
    try:
        newline_token_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
        newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
        if len(newline_positions) > 0:
            return newline_positions[-1].item()  # Use the last occurrence
    except:
        pass
    
    # Fallback to single '\n' token
    try:
        newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
        if len(newline_positions) > 0:
            return newline_positions[-1].item()
    except:
        pass
    
    # Final fallback to last token
    return len(input_ids) - 1

def extract_activation_at_newline(model, tokenizer, prompt, layer=15, swap=False):
    """Extract activation at the newline token
    
    Args:
        layer: int for single layer or list of ints for multiple layers
    
    Returns:
        If layer is int: torch.Tensor (backward compatibility)
        If layer is list: dict {layer_idx: torch.Tensor}
    """
    # Handle backward compatibility
    if isinstance(layer, int):
        single_layer_mode = True
        layer_list = [layer]
    else:
        single_layer_mode = False
        layer_list = layer
    
    # Format as chat
    if swap:
        formatted_prompt = format_as_chat_swapped(tokenizer, prompt)
    else:
        formatted_prompt = format_as_chat(tokenizer, prompt)
    
    # Tokenize
    tokens = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"].to(model.device)
    
    # Find newline position
    newline_pos = find_newline_position(input_ids[0], tokenizer, model.device)
    
    # Dictionary to store activations from multiple layers
    activations = {}
    handles = []
    
    # Create hooks for all requested layers
    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            # Extract the activation tensor (handle tuple output)
            act_tensor = output[0] if isinstance(output, tuple) else output
            activations[layer_idx] = act_tensor[0, newline_pos, :].cpu()
        return hook_fn
    
    # Register hooks for all target layers
    for layer_idx in layer_list:
        target_layer = model.model.layers[layer_idx]
        handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
        handles.append(handle)
    
    try:
        with torch.no_grad():
            _ = model(input_ids)  # Full forward pass to capture all layers
    finally:
        # Clean up all hooks
        for handle in handles:
            handle.remove()
    
    # Check that we captured all requested activations
    for layer_idx in layer_list:
        if layer_idx not in activations:
            raise ValueError(f"Failed to extract activation for layer {layer_idx} with prompt: {prompt[:50]}...")
    
    # Return format based on input type
    if single_layer_mode:
        return activations[layer_list[0]]
    else:
        return activations

def extract_activations_for_prompts(model, tokenizer, prompts, layer=15, swap=False):
    """Extract activations for a list of prompts
    
    Args:
        layer: int for single layer or list of ints for multiple layers
        
    Returns:
        If layer is int: torch.Tensor of shape (num_prompts, hidden_size)
        If layer is list: dict {layer_idx: torch.Tensor of shape (num_prompts, hidden_size)}
    """
    # Handle backward compatibility
    single_layer_mode = isinstance(layer, int)
    
    if single_layer_mode:
        # Single layer mode - maintain original behavior
        activations = []
        for prompt in prompts:
            try:
                activation = extract_activation_at_newline(model, tokenizer, prompt, layer, swap=swap)
                activations.append(activation)
                print(f"✓ Extracted activation for: {prompt[:50]}...")
            except Exception as e:
                print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")
        
        return torch.stack(activations) if activations else None
    
    else:
        # Multi-layer mode - extract all layers in single forward passes
        layer_activations = {layer_idx: [] for layer_idx in layer}
        
        for prompt in prompts:
            try:
                activation_dict = extract_activation_at_newline(model, tokenizer, prompt, layer, swap=swap)
                for layer_idx in layer:
                    layer_activations[layer_idx].append(activation_dict[layer_idx])
                print(f"✓ Extracted activations for layers {layer} for: {prompt[:50]}...")
            except Exception as e:
                print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")
        
        # Convert lists to tensors for each layer
        result = {}
        for layer_idx in layer:
            if layer_activations[layer_idx]:
                result[layer_idx] = torch.stack(layer_activations[layer_idx])
            else:
                result[layer_idx] = None
        
        return result

def compute_contrast_vector(positive_activations, negative_activations):
    """Compute contrast vector: positive_mean - negative_mean
    
    Args:
        positive_activations: torch.Tensor or dict {layer_idx: torch.Tensor}
        negative_activations: torch.Tensor or dict {layer_idx: torch.Tensor}
        
    Returns:
        If inputs are tensors: (contrast_vector, positive_mean, negative_mean)
        If inputs are dicts: dict {layer_idx: (contrast_vector, positive_mean, negative_mean)}
    """
    # Check if inputs are dictionaries (multi-layer mode)
    if isinstance(positive_activations, dict) and isinstance(negative_activations, dict):
        # Multi-layer mode
        results = {}
        
        # Get all layers (should be the same in both dictionaries)
        layers = set(positive_activations.keys()) & set(negative_activations.keys())
        
        for layer_idx in layers:
            print(f"Layer {layer_idx} has {positive_activations[layer_idx].shape} and {negative_activations[layer_idx].shape}")
            if positive_activations[layer_idx] is not None and negative_activations[layer_idx] is not None:
                positive_mean = positive_activations[layer_idx].mean(dim=0)
                negative_mean = negative_activations[layer_idx].mean(dim=0)
                contrast_vector = positive_mean - negative_mean
                results[layer_idx] = (contrast_vector, positive_mean, negative_mean)
            else:
                results[layer_idx] = None
        
        return results
    
    else:
        # Single layer mode - maintain original behavior
        positive_mean = positive_activations.mean(dim=0)
        negative_mean = negative_activations.mean(dim=0)
        contrast_vector = positive_mean - negative_mean
        return contrast_vector, positive_mean, negative_mean

def project_onto_contrast(activations, contrast_vector):
    """Project activations onto contrast vector"""
    # Normalize contrast vector
    contrast_norm = torch.norm(contrast_vector)
    if contrast_norm == 0:
        return torch.zeros(activations.shape[0])
    
    # Project each activation
    projections = []
    for activation in activations:
        projection = torch.dot(activation, contrast_vector) / contrast_norm
        projections.append(projection.item())
    
    return np.array(projections)

def generate_text(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7, do_sample=True, chat_format=True, swap=False):
    """Generate text from a prompt with the model"""
    # Format as chat
    if chat_format:
        if swap:
            formatted_prompt = format_as_chat_swapped(tokenizer, prompt)
        else:
            formatted_prompt = format_as_chat(tokenizer, prompt)
    else:
        formatted_prompt = prompt
    
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
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    return generated_text.strip()
    