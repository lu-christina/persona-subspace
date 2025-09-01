import torch
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('.')
sys.path.append('..')

class StopForward(Exception):
    """Exception to stop forward pass after target layer."""
    pass

def load_model(model_name, device=None):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Use specific device or auto device mapping
    if device is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Use all GPUs
        )
    model.eval()
    return model, tokenizer

def format_as_chat(tokenizer, prompt, **chat_kwargs):
    """Format prompt as a chat message with proper template"""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
    )

    return formatted_prompt


def format_as_chat_swapped(tokenizer, prompt, **chat_kwargs):
    """Format prompt as a chat message with proper template"""
    messages = [{"role": "user", "content": "Hello."}, {"role": "model", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
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


def extract_full_activations(model, tokenizer, conversation, layer=None, chat_format=True, **chat_kwargs):
    """Extract full activations for a conversation
    
    Args:
        layer: int for single layer or list of ints for multiple layers or None for all layers
        chat_format: whether to apply chat template
        **chat_kwargs: additional arguments for apply_chat_template
    
    Returns:
        If layer is int: torch.Tensor (backward compatibility) in shape (num_tokens, hidden_size)
        If layer is None or list: torch.Tensor in shape (num_layers, num_tokens, hidden_size)
    """
    # Handle backward compatibility
    if isinstance(layer, int):
        single_layer_mode = True
        layer_list = [layer]
    elif isinstance(layer, list):
        single_layer_mode = False
        layer_list = layer
    else:
        single_layer_mode = False
        layer_list = list(range(len(model.model.layers)))
    
    if chat_format:
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
        )
    else:
        formatted_prompt = conversation

    # Tokenize
    tokens = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"].to(model.device)
    
    # Dictionary to store activations from multiple layers
    activations = []
    handles = []
    
    # Create hooks for all requested layers
    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            # Extract the activation tensor (handle tuple output)
            act_tensor = output[0] if isinstance(output, tuple) else output
            activations.append(act_tensor[0, :, :].cpu())
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
    
    activations = torch.stack(activations)
    
    # Return format based on input type
    if single_layer_mode:
        return activations
    else:
        return activations


def extract_activation_at_newline(model, tokenizer, prompt, layer=15, swap=False, **chat_kwargs):
    """Extract activation at the newline token
    
    Args:
        layer: int for single layer or list of ints for multiple layers
        swap: whether to use swapped chat format
        **chat_kwargs: additional arguments for apply_chat_template
    
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
        formatted_prompt = format_as_chat_swapped(tokenizer, prompt, **chat_kwargs)
    else:
        formatted_prompt = format_as_chat(tokenizer, prompt, **chat_kwargs)
    
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

def extract_activations_for_prompts(model, tokenizer, prompts, layer=15, swap=False, **chat_kwargs):
    """Extract activations for a list of prompts
    
    Args:
        layer: int for single layer or list of ints for multiple layers
        swap: whether to use swapped chat format
        **chat_kwargs: additional arguments for apply_chat_template
        
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
                activation = extract_activation_at_newline(model, tokenizer, prompt, layer, swap=swap, **chat_kwargs)
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
                activation_dict = extract_activation_at_newline(model, tokenizer, prompt, layer, swap=swap, **chat_kwargs)
                for layer_idx in layer:
                    layer_activations[layer_idx].append(activation_dict[layer_idx])
                print(f"✓ Extracted activations for: {prompt[:50]}...")
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
    """Project activations onto contrast vector
    
    Args:
        activations: torch.Tensor - either single activation (1D) or batch of activations (2D)
        contrast_vector: torch.Tensor - contrast vector to project onto
        
    Returns:
        float (if single activation) or np.array (if batch of activations)
    """
    # Handle single activation case (matching compute_projection signature)
    if activations.ndim == 1:
        # Ensure tensors are on same device and dtype
        activations = activations.to(contrast_vector.device).float()
        contrast_vector = contrast_vector.float()
        
        # Scalar projection: (h · v) / ||v||
        contrast_norm = torch.norm(contrast_vector)
        if contrast_norm == 0:
            return 0.0
        
        projection = torch.dot(activations, contrast_vector) / contrast_norm
        return projection.item()
    
    # Handle batch case (original behavior)
    else:
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

def eos_suppressor(tokenizer):
    """Create a logits processor that suppresses EOS token"""
    eos_token_id = tokenizer.eos_token_id
    
    def suppress_eos_processor(input_ids, scores):
        if eos_token_id is not None:
            scores[:, eos_token_id] = -float('inf')
        return scores
    
    return suppress_eos_processor

def capture_hidden_state(model, input_ids, layer, position=-1):
    """Capture hidden state at specified layer and position
    
    Args:
        model: The transformer model
        input_ids: Input token IDs tensor
        layer: Layer index to capture from
        position: Token position to capture (-1 for last token)
        
    Returns:
        torch.Tensor: Hidden state at the specified layer and position
    """
    captured_state = None
    
    def capture_hook(module, input, output):
        nonlocal captured_state
        # Handle tuple outputs (some models return (hidden_states, ...))
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Capture the hidden state at specified position
        captured_state = hidden_states[0, position, :].clone().cpu()
    
    # Register hook on target layer
    layer_module = model.model.layers[layer]
    hook_handle = layer_module.register_forward_hook(capture_hook)
    
    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        hook_handle.remove()
    
    if captured_state is None:
        raise ValueError(f"Failed to capture hidden state at layer {layer}, position {position}")
    
    return captured_state

def sample_next_token(model, tokenizer, input_ids, suppress_eos=True):
    """Sample next token from model logits
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        input_ids: Current input token IDs
        suppress_eos: Whether to suppress EOS token
        
    Returns:
        tuple: (next_token_id, updated_input_ids)
    """
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Suppress EOS token if requested
        if suppress_eos:
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None:
                logits[eos_token_id] = -float('inf')
        
        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        
        # Update input_ids
        updated_input_ids = torch.cat([
            input_ids, 
            torch.tensor([[next_token_id]], device=input_ids.device)
        ], dim=1)
        
        return next_token_id, updated_input_ids

def generate_text(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7, do_sample=True, chat_format=True, swap=False, **chat_kwargs):
    """Generate text from a prompt with the model"""
    # Format as chat
    if chat_format:
        if swap:
            formatted_prompt = format_as_chat_swapped(tokenizer, prompt, **chat_kwargs)
        else:
            formatted_prompt = format_as_chat(tokenizer, prompt, **chat_kwargs)
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
    
def is_qwen_model(model_name_or_tokenizer):
    """Check if this is a Qwen model based on model name or tokenizer."""
    if isinstance(model_name_or_tokenizer, str):
        return 'qwen' in model_name_or_tokenizer.lower()
    else:
        # Check tokenizer name_or_path attribute
        tokenizer_name = getattr(model_name_or_tokenizer, 'name_or_path', '').lower()
        return 'qwen' in tokenizer_name

def is_gemma_model(model_name_or_tokenizer):
    """Check if this is a Gemma model based on model name or tokenizer."""
    if isinstance(model_name_or_tokenizer, str):
        return 'gemma' in model_name_or_tokenizer.lower()
    else:
        # Check tokenizer name_or_path attribute
        tokenizer_name = getattr(model_name_or_tokenizer, 'name_or_path', '').lower()
        return 'gemma' in tokenizer_name

def is_llama_model(model_name_or_tokenizer):
    """Check if this is a Llama model based on model name or tokenizer."""
    if isinstance(model_name_or_tokenizer, str):
        model_lower = model_name_or_tokenizer.lower()
        return 'llama' in model_lower or 'meta-llama' in model_lower
    else:
        # Check tokenizer name_or_path attribute
        tokenizer_name = getattr(model_name_or_tokenizer, 'name_or_path', '').lower()
        return 'llama' in tokenizer_name or 'meta-llama' in tokenizer_name

def get_response_indices_qwen(conversation, tokenizer, per_turn=False, **chat_kwargs):
    """
    Qwen-specific implementation for extracting response token indices.
    
    Qwen uses <|im_start|>assistant and <|im_end|> tokens, and may include
    thinking tokens that need to be filtered out when thinking is disabled.
    """
    if per_turn:
        all_turn_indices = []
    else:
        response_indices = []
    
    # Check if thinking is enabled
    enable_thinking = chat_kwargs.get('enable_thinking', False)
    
    # Get the full formatted conversation
    full_formatted = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
    )
    full_tokens = tokenizer(full_formatted, add_special_tokens=False)
    all_token_ids = full_tokens['input_ids']
    
    # Get special token IDs for Qwen
    try:
        im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        assistant_token_id = tokenizer.convert_tokens_to_ids('assistant')
        
        # Thinking tokens (may not exist in all Qwen variants)
        try:
            think_start_id = tokenizer.convert_tokens_to_ids('<think>')
            think_end_id = tokenizer.convert_tokens_to_ids('</think>')
        except (KeyError, ValueError):
            think_start_id = None
            think_end_id = None
            
    except (KeyError, ValueError):
        # Fallback if special tokens not found
        return get_response_indices_simple(conversation, tokenizer, per_turn, **chat_kwargs)
    
    # Find assistant response sections
    i = 0
    while i < len(all_token_ids):
        # Look for <|im_start|>assistant pattern
        if (i + 1 < len(all_token_ids) and 
            all_token_ids[i] == im_start_id and 
            all_token_ids[i + 1] == assistant_token_id):
            
            # Found start of assistant response, skip the <|im_start|>assistant tokens
            response_start = i + 2
            
            # Find the corresponding <|im_end|>
            response_end = None
            for j in range(response_start, len(all_token_ids)):
                if all_token_ids[j] == im_end_id:
                    response_end = j  # Don't include the <|im_end|> token
                    break
            
            if response_end is not None:
                # Extract tokens in this range
                raw_turn_indices = list(range(response_start, response_end))
                
                # Filter out thinking tokens if thinking disabled
                if not enable_thinking and think_start_id is not None and think_end_id is not None:
                    filtered_indices = []
                    skip_until_think_end = False
                    
                    for idx in raw_turn_indices:
                        token_id = all_token_ids[idx]
                        
                        # Check if we hit a <think> token
                        if token_id == think_start_id:
                            skip_until_think_end = True
                            continue
                        
                        # Check if we hit a </think> token
                        if token_id == think_end_id:
                            skip_until_think_end = False
                            continue
                        
                        # Skip tokens that are inside thinking blocks
                        if skip_until_think_end:
                            continue
                        
                        # Include all tokens that are not inside thinking blocks
                        filtered_indices.append(idx)
                    
                    # Clean up extracted text by removing extra whitespace/newlines at boundaries
                    # but preserve internal spaces
                    if filtered_indices:
                        # Get the text to check for leading/trailing cleanup
                        extracted_token_ids = [all_token_ids[i] for i in filtered_indices]
                        extracted_text = tokenizer.decode(extracted_token_ids)
                        
                        # If text starts/ends with excessive whitespace, find better boundaries
                        if extracted_text.strip() != extracted_text:
                            # Remove leading whitespace-only tokens
                            while (filtered_indices and 
                                   tokenizer.decode([all_token_ids[filtered_indices[0]]]).strip() == ''):
                                filtered_indices.pop(0)
                            
                            # Remove trailing whitespace-only tokens
                            while (filtered_indices and 
                                   tokenizer.decode([all_token_ids[filtered_indices[-1]]]).strip() == ''):
                                filtered_indices.pop()
                    
                    turn_indices = filtered_indices
                else:
                    turn_indices = raw_turn_indices
                
                if per_turn:
                    all_turn_indices.append(turn_indices)
                else:
                    response_indices.extend(turn_indices)
                
                i = response_end + 1
            else:
                # No matching <|im_end|> found, skip this token
                i += 1
        else:
            i += 1
    
    return all_turn_indices if per_turn else response_indices

def get_response_indices_llama(conversation, tokenizer, per_turn=False, **chat_kwargs):
    """
    Llama-specific implementation for extracting response token indices.
    
    Llama uses <|start_header_id|>assistant<|end_header_id|> and <|eot_id|> tokens.
    """
    if per_turn:
        all_turn_indices = []
    else:
        response_indices = []
    
    # Get the full formatted conversation
    full_formatted = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
    )
    full_tokens = tokenizer(full_formatted, add_special_tokens=False)
    all_token_ids = full_tokens['input_ids']
    
    # Get special token IDs for Llama
    try:
        start_header_id = tokenizer.convert_tokens_to_ids('<|start_header_id|>')
        end_header_id = tokenizer.convert_tokens_to_ids('<|end_header_id|>')
        eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        assistant_token_id = tokenizer.convert_tokens_to_ids('assistant')
    except (KeyError, ValueError):
        # Fallback if special tokens not found
        return get_response_indices_simple(conversation, tokenizer, per_turn, **chat_kwargs)
    
    # Find assistant response sections
    i = 0
    while i < len(all_token_ids):
        # Look for <|start_header_id|>assistant<|end_header_id|> pattern
        if (i + 2 < len(all_token_ids) and 
            all_token_ids[i] == start_header_id and 
            all_token_ids[i + 1] == assistant_token_id and 
            all_token_ids[i + 2] == end_header_id):
            
            # Found start of assistant response, skip the header tokens and any following newlines
            response_start = i + 3
            
            # Skip any immediate newline tokens after the header
            while (response_start < len(all_token_ids) and 
                   tokenizer.decode([all_token_ids[response_start]]).strip() == ''):
                response_start += 1
            
            # Find the corresponding <|eot_id|>
            response_end = None
            for j in range(response_start, len(all_token_ids)):
                if all_token_ids[j] == eot_id:
                    response_end = j  # Don't include the <|eot_id|> token
                    break
            
            if response_end is not None:
                # Remove trailing whitespace tokens before <|eot_id|>
                while (response_end > response_start and 
                       tokenizer.decode([all_token_ids[response_end - 1]]).strip() == ''):
                    response_end -= 1
                
                turn_indices = list(range(response_start, response_end))
                
                if per_turn:
                    all_turn_indices.append(turn_indices)
                else:
                    response_indices.extend(turn_indices)
                
                i = response_end + 1
            else:
                # No matching <|eot_id|> found, skip this token
                i += 1
        else:
            i += 1
    
    return all_turn_indices if per_turn else response_indices

def get_response_indices_gemma(conversation, tokenizer, per_turn=False, **chat_kwargs):
    """
    Gemma-specific implementation using the original offset mapping approach.
    """
    if per_turn:
        all_turn_indices = []
    else:
        response_indices = []
    
    # Process conversation incrementally to find assistant response boundaries
    for i, turn in enumerate(conversation):
        if turn['role'] != 'assistant':
            continue
            
        # Get conversation up to but not including this assistant turn
        conversation_before = conversation[:i]
        
        # Get conversation up to and including this assistant turn  
        conversation_including = conversation[:i+1]
        
        # Format and tokenize both versions
        if conversation_before:
            before_formatted = tokenizer.apply_chat_template(
                conversation_before, tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
            before_tokens = tokenizer(before_formatted, add_special_tokens=False)
            before_length = len(before_tokens['input_ids'])
        else:
            before_length = 0
            
        including_formatted = tokenizer.apply_chat_template(
            conversation_including, tokenize=False, add_generation_prompt=False, **chat_kwargs
        )
        including_tokens = tokenizer(including_formatted, add_special_tokens=False)
        including_length = len(including_tokens['input_ids'])
        
        # Find the actual content of this assistant response (excluding formatting tokens)
        assistant_content = turn['content'].strip()
        
        # Collect indices for this turn
        turn_indices = []
        
        # Find where the assistant content appears in the formatted text
        content_start_in_formatted = including_formatted.find(assistant_content)
        if content_start_in_formatted != -1:
            content_end_in_formatted = content_start_in_formatted + len(assistant_content)
            
            # Convert character positions to token indices using offset mapping
            tokens_with_offsets = tokenizer(including_formatted, return_offsets_mapping=True, add_special_tokens=False)
            offset_mapping = tokens_with_offsets['offset_mapping']
            
            # Find tokens that overlap with the assistant content
            for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                if (start_char >= content_start_in_formatted and start_char < content_end_in_formatted) or \
                   (end_char > content_start_in_formatted and end_char <= content_end_in_formatted) or \
                   (start_char < content_start_in_formatted and end_char > content_end_in_formatted):
                    turn_indices.append(token_idx)
        else:
            # Fallback to original method if content not found
            assistant_start = before_length
            assistant_end = including_length
            turn_indices.extend(range(assistant_start, assistant_end))
        
        # Store indices based on per_turn flag
        if per_turn:
            all_turn_indices.append(turn_indices)
        else:
            response_indices.extend(turn_indices)
    
    return all_turn_indices if per_turn else response_indices

def get_response_indices_simple(conversation, tokenizer, per_turn=False, **chat_kwargs):
    """
    Simple fallback implementation using range-based approach.
    """
    if per_turn:
        all_turn_indices = []
    else:
        response_indices = []
    
    # Process conversation incrementally to find assistant response boundaries
    for i, turn in enumerate(conversation):
        if turn['role'] != 'assistant':
            continue
            
        # Get conversation up to but not including this assistant turn
        conversation_before = conversation[:i]
        
        # Get conversation up to and including this assistant turn  
        conversation_including = conversation[:i+1]
        
        # Format and tokenize both versions
        if conversation_before:
            before_formatted = tokenizer.apply_chat_template(
                conversation_before, tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
            before_tokens = tokenizer(before_formatted, add_special_tokens=False)
            before_length = len(before_tokens['input_ids'])
        else:
            before_length = 0
            
        including_formatted = tokenizer.apply_chat_template(
            conversation_including, tokenize=False, add_generation_prompt=False, **chat_kwargs
        )
        including_tokens = tokenizer(including_formatted, add_special_tokens=False)
        including_length = len(including_tokens['input_ids'])
        
        # The assistant response tokens are between before_length and including_length
        assistant_start = before_length
        assistant_end = including_length
        
        turn_indices = list(range(assistant_start, assistant_end))
        
        # Store indices based on per_turn flag
        if per_turn:
            all_turn_indices.append(turn_indices)
        else:
            response_indices.extend(turn_indices)
    
    return all_turn_indices if per_turn else response_indices

def get_response_indices(conversation, tokenizer, model_name=None, per_turn=False, **chat_kwargs):
    """
    Get every token index of the model's response.
    
    Args:
        conversation: List of dict with 'role' and 'content' keys
        tokenizer: Tokenizer to apply chat template and tokenize
        model_name: Model name to determine which extraction method to use
        per_turn: Bool, if True return list of lists (one per assistant turn), 
                 if False return single flat list (default, backward compatible)
        **chat_kwargs: additional arguments for apply_chat_template
    
    Returns:
        response_indices: list of token positions where the model is responding (per_turn=False)
                         or list of lists of token positions per turn (per_turn=True)
    """
    # Determine model type
    if model_name and is_qwen_model(model_name):
        return get_response_indices_qwen(conversation, tokenizer, per_turn, **chat_kwargs)
    elif model_name and (is_gemma_model(model_name) or is_llama_model(model_name)):
        # Gemma and Llama can use the same offset mapping approach
        return get_response_indices_gemma(conversation, tokenizer, per_turn, **chat_kwargs)
    elif model_name is None:
        # Try to detect from tokenizer
        if is_qwen_model(tokenizer):
            return get_response_indices_qwen(conversation, tokenizer, per_turn, **chat_kwargs)
        elif is_gemma_model(tokenizer) or is_llama_model(tokenizer):
            # Gemma and Llama can use the same approach
            return get_response_indices_gemma(conversation, tokenizer, per_turn, **chat_kwargs)
        else:
           raise ValueError(f"Unsupported model: {model_name}. Supported models: Qwen, Gemma, Llama. "
                        f"For other models, pass model_name=None to use fallback method.")
    else:
        # Unsupported model
        raise ValueError(f"Unsupported model: {model_name}. Supported models: Qwen, Gemma, Llama. "
                        f"For other models, pass model_name=None to use fallback method.")

def mean_response_activation(activations, conversation, tokenizer, model_name=None, **chat_kwargs):
    """
    Get the mean activation of the model's response to the user's message.
    """
    # get the token positions of model responses
    response_indices = get_response_indices(conversation, tokenizer, model_name, **chat_kwargs)

    # get the mean activation of the model's response to the user's message
    mean_activation = activations[:, response_indices, :].mean(dim=1)
    return mean_activation

def mean_response_activation_per_turn(activations, conversation, tokenizer, model_name=None, **chat_kwargs):
    """
    Get the mean activation for each of the model's response turns.
    
    Args:
        activations: Tensor with shape (layers, tokens, features)
        conversation: List of dict with 'role' and 'content' keys
        tokenizer: Tokenizer to apply chat template and tokenize
        model_name: Model name to determine which extraction method to use
        **chat_kwargs: additional arguments for apply_chat_template
    
    Returns:
        List[torch.Tensor]: List of mean activations, one per assistant turn
    """
    # Get token positions for each assistant turn
    response_indices_per_turn = get_response_indices(conversation, tokenizer, model_name, per_turn=True, **chat_kwargs)
    
    # Calculate mean activation for each turn
    mean_activations_per_turn = []
    
    for turn_indices in response_indices_per_turn:
        if len(turn_indices) > 0:
            # Get mean activation for this turn's tokens
            turn_mean_activation = activations[:, turn_indices, :].mean(dim=1)
            mean_activations_per_turn.append(turn_mean_activation)
    
    return mean_activations_per_turn