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

def extract_activation_at_newline(model, tokenizer, prompt, layer=15):
    """Extract activation at the newline token with early stopping"""
    # Format as chat
    formatted_prompt = format_as_chat(tokenizer, prompt)
    
    # Tokenize
    tokens = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"].to(model.device)
    
    # Find newline position
    newline_pos = find_newline_position(input_ids[0], tokenizer, model.device)
    
    # Get target layer
    target_layer = model.model.layers[layer]
    
    # Hook to capture activations and stop forward pass
    activation = None
    
    def hook_fn(module, input, output):
        nonlocal activation
        # Extract the activation tensor (handle tuple output)
        act_tensor = output[0] if isinstance(output, tuple) else output
        activation = act_tensor[0, newline_pos, :].cpu()  # Extract at newline position
        raise StopForward()
    
    # Register hook
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(input_ids)
    except StopForward:
        pass  # Expected - we stopped the forward pass
    finally:
        handle.remove()
    
    if activation is None:
        raise ValueError(f"Failed to extract activation for prompt: {prompt[:50]}...")
    
    return activation

def extract_activations_for_prompts(model, tokenizer, prompts, layer=15):
    """Extract activations for a list of prompts"""
    activations = []
    for prompt in prompts:
        try:
            activation = extract_activation_at_newline(model, tokenizer, prompt, layer)
            activations.append(activation)
            print(f"✓ Extracted activation for: {prompt[:50]}...")
        except Exception as e:
            print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")
    
    return torch.stack(activations) if activations else None

def compute_contrast_vector(positive_activations, negative_activations):
    """Compute contrast vector: positive_mean - negative_mean"""
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

def generate_text(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7, do_sample=True, chat_format=True):
    """Generate text from a prompt with the model"""
    # Format as chat
    if chat_format:
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
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()
    