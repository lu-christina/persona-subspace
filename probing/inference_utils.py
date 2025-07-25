"""
Inference utilities for using vLLM to chat with offline HuggingFace models.

This module provides helper functions for:
- Loading and managing vLLM models directly (no server overhead)
- Single message and multi-turn conversations
- Batch processing of multiple prompts
- Proper resource management and cleanup

Uses vLLM's LLM class directly for efficient offline inference.
"""

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from vllm import LLM, SamplingParams
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import vLLM: {e}")
    print("vLLM functionality will not be available. Please install/fix vLLM if needed.")
    
    # Create dummy classes to allow the module to load
    class LLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("vLLM not properly installed")
        
    class SamplingParams:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("vLLM not properly installed")
    
    def random_uuid():
        import uuid
        return str(uuid.uuid4())
    
    VLLM_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global registry of active models
_active_models: Dict[str, 'VLLMModelWrapper'] = {}


@dataclass
class VLLMModelWrapper:
    """Wrapper class for managing vLLM models with direct access."""
    model_name: str
    llm: LLM
    tokenizer: AutoTokenizer
    max_model_len: int
    tensor_parallel_size: int
    
    def __post_init__(self):
        """Register this model in the global registry."""
        _active_models[self.model_name] = self
    
    def close(self):
        """Clean up the vLLM model and remove from registry."""
        try:
            if self.model_name in _active_models:
                del _active_models[self.model_name]
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"Closed vLLM model {self.model_name}")
        except Exception as e:
            logger.error(f"Error closing model {self.model_name}: {e}")
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()


def load_vllm_model(
    model_name: str,
    max_model_len: int = 8192,
    tensor_parallel_size: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    **kwargs
) -> VLLMModelWrapper:
    """
    Load a HuggingFace model directly using vLLM's LLM class.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        max_model_len: Maximum sequence length (default: 8192)
        tensor_parallel_size: Number of GPUs to use (default: auto-detect)
        gpu_memory_utilization: GPU memory utilization ratio (default: 0.9)
        dtype: Model data type (default: "auto")
        **kwargs: Additional arguments passed to vLLM LLM constructor
    
    Returns:
        VLLMModelWrapper: Wrapper object for the loaded model
        
    Raises:
        RuntimeError: If model loading fails or vLLM not properly installed
    """
    # Check if vLLM is properly installed
    if not VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM is not properly installed or configured. "
            "Please ensure vLLM is installed and compatible with your PyTorch version. "
            "You may need to reinstall vLLM: pip uninstall vllm && pip install vllm"
        )
    
    # Check if model is already loaded
    if model_name in _active_models:
        logger.info(f"Model {model_name} already loaded, returning existing instance")
        return _active_models[model_name]
    
    try:
        # Auto-detect tensor parallel size if not specified
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if tensor_parallel_size == 0:
                tensor_parallel_size = 1
        
        logger.info(f"Loading vLLM model: {model_name} with {tensor_parallel_size} GPUs")
        
        # Load the model using vLLM's LLM class
        llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
            **kwargs
        )
        
        # Load the tokenizer separately for chat template support
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create wrapper
        wrapper = VLLMModelWrapper(
            model_name=model_name,
            llm=llm,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size
        )
        
        logger.info(f"Successfully loaded vLLM model: {model_name}")
        return wrapper
        
    except Exception as e:
        logger.error(f"Failed to load vLLM model {model_name}: {e}")
        raise RuntimeError(f"Failed to load vLLM model {model_name}: {e}")


def close_vllm_model(model_wrapper: VLLMModelWrapper) -> None:
    """
    Clean shutdown of vLLM deployment.
    
    Args:
        model_wrapper: VLLMModelWrapper instance to close
    """
    if model_wrapper:
        model_wrapper.close()


@contextmanager
def vllm_model_context(model_name: str, **kwargs):
    """
    Context manager for automatic vLLM model cleanup.
    
    Args:
        model_name: HuggingFace model identifier
        **kwargs: Arguments passed to load_vllm_model
        
    Yields:
        VLLMModelWrapper: The loaded model wrapper
        
    Example:
        with vllm_model_context("meta-llama/Llama-3.1-8B-Instruct") as model:
            response = chat_with_model(model, "Hello!")
    """
    model = None
    try:
        model = load_vllm_model(model_name, **kwargs)
        yield model
    finally:
        if model:
            model.close()


def format_chat_prompt(
    message: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Format messages for chat template.
    
    Args:
        message: Current user message
        system_prompt: Optional system message
        conversation_history: Previous conversation as list of dicts with 'role' and 'content' keys
        
    Returns:
        List of message dictionaries formatted for the model
    """
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    return messages


def chat_with_model(
    model_wrapper: VLLMModelWrapper,
    message: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    **kwargs
) -> str:
    """
    Send a single message and get response using direct vLLM generation.
    
    Args:
        model_wrapper: VLLMModelWrapper from load_vllm_model()
        message: User message string
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum response tokens (default: 512)
        top_p: Top-p sampling parameter (default: 0.9)
        system_prompt: Optional system message
        conversation_history: Previous conversation history
        **kwargs: Additional generation parameters
        
    Returns:
        Generated response string
        
    Raises:
        RuntimeError: If inference fails
    """
    try:
        # Format the messages
        formatted_messages = format_chat_prompt(message, system_prompt, conversation_history)
        
        # Apply chat template using the tokenizer
        if hasattr(model_wrapper.tokenizer, 'apply_chat_template'):
            prompt = model_wrapper.tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation if no chat template
            prompt = ""
            for msg in formatted_messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Generate response
        outputs = model_wrapper.llm.generate([prompt], sampling_params)
        
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("No output generated from model")
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text
        
    except Exception as e:
        logger.error(f"Error in chat_with_model: {e}")
        raise RuntimeError(f"Chat inference failed: {e}")


def chat_conversation(
    model_wrapper: VLLMModelWrapper,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    **kwargs
) -> str:
    """
    Handle multi-turn conversations using direct vLLM generation.
    
    Args:
        model_wrapper: VLLMModelWrapper instance
        messages: List of message dicts with keys "role" ("user"/"assistant"/"system") and "content"
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum response tokens (default: 512)
        top_p: Top-p sampling parameter (default: 0.9)
        **kwargs: Additional generation parameters
        
    Returns:
        Generated response string
    """
    try:
        # Apply chat template using the tokenizer
        if hasattr(model_wrapper.tokenizer, 'apply_chat_template'):
            prompt = model_wrapper.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation if no chat template
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Generate response
        outputs = model_wrapper.llm.generate([prompt], sampling_params)
        
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("No output generated from model")
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text
        
    except Exception as e:
        logger.error(f"Error in chat_conversation: {e}")
        raise RuntimeError(f"Conversation inference failed: {e}")


def continue_conversation(
    model_wrapper: VLLMModelWrapper,
    conversation_history: List[Dict[str, str]],
    new_message: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    **kwargs
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Add message to existing conversation and get response.
    
    Args:
        model_wrapper: VLLMModelWrapper instance
        conversation_history: Previous conversation as list of message dicts
        new_message: New user message to append
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum response tokens (default: 512)
        **kwargs: Additional generation parameters
        
    Returns:
        Tuple of (response_string, updated_conversation_history)
    """
    # Add new user message to history
    updated_history = conversation_history.copy()
    updated_history.append({"role": "user", "content": new_message})
    
    # Get response
    response = chat_conversation(model_wrapper, updated_history, temperature, max_tokens, **kwargs)
    
    # Add assistant response to history
    updated_history.append({"role": "assistant", "content": response})
    
    return response, updated_history


def batch_chat(
    model_wrapper: VLLMModelWrapper,
    messages: List[str],
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
    progress: bool = True,
    **kwargs
) -> List[str]:
    """
    Process multiple prompts efficiently using vLLM's native batch generation.
    
    Args:
        model_wrapper: VLLMModelWrapper instance
        messages: List of user message strings
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum response tokens (default: 512)
        top_p: Top-p sampling parameter (default: 0.9)
        system_prompt: Optional system message applied to all
        progress: Show progress bar (default: True)
        **kwargs: Additional generation parameters
        
    Returns:
        List of response strings (same order as input)
    """
    try:
        # Prepare all prompts
        prompts = []
        for message in messages:
            formatted_messages = format_chat_prompt(message, system_prompt)
            
            # Apply chat template
            if hasattr(model_wrapper.tokenizer, 'apply_chat_template'):
                prompt = model_wrapper.tokenizer.apply_chat_template(
                    formatted_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback: simple concatenation if no chat template
                prompt = ""
                for msg in formatted_messages:
                    if msg["role"] == "system":
                        prompt += f"System: {msg['content']}\n"
                    elif msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Assistant: {msg['content']}\n"
                prompt += "Assistant: "
            
            prompts.append(prompt)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Generate all responses in batch
        if progress:
            logger.info(f"Processing batch of {len(prompts)} prompts...")
        
        outputs = model_wrapper.llm.generate(prompts, sampling_params)
        
        # Extract generated texts
        responses = []
        for output in outputs:
            if output.outputs:
                generated_text = output.outputs[0].text.strip()
                responses.append(generated_text)
            else:
                responses.append("")
                logger.warning("Empty output received for a prompt")
        
        if progress:
            logger.info(f"Completed batch processing of {len(responses)} prompts")
        
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch_chat: {e}")
        raise RuntimeError(f"Batch chat inference failed: {e}")


def batch_chat_with_system(
    model_wrapper: VLLMModelWrapper,
    messages: List[str],
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    progress: bool = True,
    **kwargs
) -> List[str]:
    """
    Batch chat with consistent system prompt.
    
    Args:
        model_wrapper: VLLMModelWrapper instance
        messages: List of user message strings
        system_prompt: System message applied to all conversations
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum response tokens (default: 512)
        top_p: Top-p sampling parameter (default: 0.9)
        progress: Show progress bar (default: True)
        **kwargs: Additional generation parameters
        
    Returns:
        List of response strings
    """
    return batch_chat(
        model_wrapper, messages, temperature, max_tokens, top_p, system_prompt, progress, **kwargs
    )


def batch_conversation_chat(
    model_wrapper: VLLMModelWrapper,
    conversations: List[List[Dict[str, str]]],
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    progress: bool = True,
    **kwargs
) -> List[str]:
    """
    Process multiple conversations with history efficiently using vLLM's native batch generation.
    
    Args:
        model_wrapper: VLLMModelWrapper instance
        conversations: List of conversation histories, where each conversation is a list of 
                      message dicts with keys "role" ("user"/"assistant"/"system") and "content"
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum response tokens (default: 512)
        top_p: Top-p sampling parameter (default: 0.9)
        progress: Show progress bar (default: True)
        **kwargs: Additional generation parameters
        
    Returns:
        List of response strings (same order as input conversations)
    """
    try:
        # Prepare all prompts from conversation histories
        prompts = []
        for conversation in conversations:
            # Apply chat template to each conversation
            if hasattr(model_wrapper.tokenizer, 'apply_chat_template'):
                prompt = model_wrapper.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback: simple concatenation if no chat template
                prompt = ""
                for msg in conversation:
                    if msg["role"] == "system":
                        prompt += f"System: {msg['content']}\n"
                    elif msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Assistant: {msg['content']}\n"
                prompt += "Assistant: "
            
            prompts.append(prompt)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Generate all responses in batch
        if progress:
            logger.info(f"Processing batch of {len(prompts)} conversations...")
        
        outputs = model_wrapper.llm.generate(prompts, sampling_params)
        
        # Extract generated texts
        responses = []
        for output in outputs:
            if output.outputs:
                generated_text = output.outputs[0].text.strip()
                responses.append(generated_text)
            else:
                responses.append("")
                logger.warning("Empty output received for a conversation")
        
        if progress:
            logger.info(f"Completed batch processing of {len(responses)} conversations")
        
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch_conversation_chat: {e}")
        raise RuntimeError(f"Batch conversation chat inference failed: {e}")


def batch_continue_conversations(
    model_wrapper: VLLMModelWrapper,
    conversation_histories: List[List[Dict[str, str]]],
    new_messages: List[str],
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    progress: bool = True,
    **kwargs
) -> Tuple[List[str], List[List[Dict[str, str]]]]:
    """
    Continue multiple ongoing conversations by adding new user messages and getting responses.
    
    Args:
        model_wrapper: VLLMModelWrapper instance
        conversation_histories: List of existing conversation histories, where each is a list of 
                               message dicts with keys "role" and "content"
        new_messages: List of new user messages to add to each conversation
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum response tokens (default: 512)
        top_p: Top-p sampling parameter (default: 0.9)
        progress: Show progress bar (default: True)
        **kwargs: Additional generation parameters
        
    Returns:
        Tuple of (response_strings, updated_conversation_histories)
        
    Raises:
        ValueError: If lengths of conversation_histories and new_messages don't match
    """
    if len(conversation_histories) != len(new_messages):
        raise ValueError(
            f"Length mismatch: {len(conversation_histories)} conversations "
            f"but {len(new_messages)} new messages"
        )
    
    # Create updated conversations with new user messages
    updated_conversations = []
    for history, new_message in zip(conversation_histories, new_messages):
        updated_history = history.copy()
        updated_history.append({"role": "user", "content": new_message})
        updated_conversations.append(updated_history)
    
    # Get responses using batch processing
    responses = batch_conversation_chat(
        model_wrapper, 
        updated_conversations, 
        temperature, 
        max_tokens, 
        top_p, 
        progress, 
        **kwargs
    )
    
    # Add assistant responses to conversation histories
    final_conversations = []
    for conversation, response in zip(updated_conversations, responses):
        final_conversation = conversation.copy()
        final_conversation.append({"role": "assistant", "content": response})
        final_conversations.append(final_conversation)
    
    return responses, final_conversations


def get_available_models() -> List[str]:
    """
    List currently deployed vLLM models.
    
    Returns:
        List of model names that are currently loaded
    """
    return list(_active_models.keys())


def get_model_info(model_wrapper: VLLMModelWrapper) -> Dict[str, Any]:
    """
    Get information about loaded model.
    
    Args:
        model_wrapper: VLLMModelWrapper instance
        
    Returns:
        Dict containing model metadata
    """
    return {
        "model_name": model_wrapper.model_name,
        "max_model_len": model_wrapper.max_model_len,
        "tensor_parallel_size": model_wrapper.tensor_parallel_size,
        "is_active": model_wrapper.model_name in _active_models,
        "tokenizer_info": {
            "vocab_size": model_wrapper.tokenizer.vocab_size,
            "model_max_length": model_wrapper.tokenizer.model_max_length,
            "has_chat_template": hasattr(model_wrapper.tokenizer, 'apply_chat_template')
        },
        "gpu_info": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
    }


def cleanup_all_models():
    """Clean up all active vLLM models."""
    models_to_close = list(_active_models.values())
    for model in models_to_close:
        model.close()
    _active_models.clear()
    logger.info("Cleaned up all vLLM models")


# Register cleanup on module import
import atexit
atexit.register(cleanup_all_models)


# Example usage
if __name__ == "__main__":
    """
    Example usage of the inference utilities.
    
    To run these examples:
    python inference_utils.py
    """
    
    # Example 1: Simple chat with context manager
    def example_simple_chat():
        print("=== Example 1: Simple Chat ===")
        with vllm_model_context("microsoft/DialoGPT-medium") as model:
            response = chat_with_model(model, "Hello, how are you?")
            print(f"Response: {response}")
    
    # Example 2: Multi-turn conversation
    def example_conversation():
        print("\n=== Example 2: Multi-turn Conversation ===")
        model = load_vllm_model("microsoft/DialoGPT-medium", max_model_len=1024)
        
        conversation = []
        
        # First exchange
        response1, conversation = continue_conversation(
            model, conversation, "What's your favorite programming language?"
        )
        print(f"User: What's your favorite programming language?")
        print(f"Assistant: {response1}")
        
        # Second exchange
        response2, conversation = continue_conversation(
            model, conversation, "Can you write a simple function in that language?"
        )
        print(f"User: Can you write a simple function in that language?")
        print(f"Assistant: {response2}")
        
        model.close()
    
    # Example 3: Batch processing
    def example_batch_processing():
        print("\n=== Example 3: Batch Processing ===")
        model = load_vllm_model("microsoft/DialoGPT-medium", max_model_len=512)
        
        questions = [
            "What is machine learning?",
            "How does neural network work?",
            "What are the benefits of deep learning?",
            "Explain gradient descent"
        ]
        
        responses = batch_chat(
            model, 
            questions, 
            system_prompt="You are a helpful AI assistant that explains technical concepts clearly.",
            max_tokens=200,
            temperature=0.7
        )
        
        for q, r in zip(questions, responses):
            print(f"Q: {q}")
            print(f"A: {r}\n")
        
        model.close()
    
    # Example 4: Batch conversation processing
    def example_batch_conversations():
        print("\n=== Example 4: Batch Conversation Processing ===")
        model = load_vllm_model("microsoft/DialoGPT-medium", max_model_len=1024)
        
        # Example conversations with history
        conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "What are its main features?"}
            ],
            [
                {"role": "user", "content": "Tell me about machine learning."},
                {"role": "assistant", "content": "Machine learning is a subset of AI."},
                {"role": "user", "content": "What are common algorithms?"}
            ],
            [
                {"role": "system", "content": "You are an expert in mathematics."},
                {"role": "user", "content": "Explain calculus briefly."}
            ]
        ]
        
        responses = batch_conversation_chat(
            model,
            conversations,
            temperature=0.7,
            max_tokens=150
        )
        
        for i, response in enumerate(responses):
            print(f"Conversation {i+1} response: {response}\n")
        
        model.close()
    
    # Example 5: Continuing multiple conversations
    def example_continue_conversations():
        print("\n=== Example 5: Continue Multiple Conversations ===")
        model = load_vllm_model("microsoft/DialoGPT-medium", max_model_len=1024)
        
        # Initial conversation histories
        histories = [
            [
                {"role": "user", "content": "Hello, I'm learning Python."},
                {"role": "assistant", "content": "Great! Python is an excellent language to learn."}
            ],
            [
                {"role": "user", "content": "I'm interested in AI."},
                {"role": "assistant", "content": "AI is a fascinating field with many applications."}
            ]
        ]
        
        # New messages to continue each conversation
        new_messages = [
            "Can you recommend some Python resources?",
            "What should I learn first in AI?"
        ]
        
        responses, updated_histories = batch_continue_conversations(
            model,
            histories,
            new_messages,
            temperature=0.7,
            max_tokens=100
        )
        
        for i, (response, history) in enumerate(zip(responses, updated_histories)):
            print(f"Conversation {i+1}:")
            print(f"  Latest response: {response}")
            print(f"  Total messages in history: {len(history)}\n")
        
        model.close()
    
    # Example 6: Model information
    def example_model_info():
        print("\n=== Example 6: Model Information ===")
        model = load_vllm_model("microsoft/DialoGPT-medium")
        
        info = get_model_info(model)
        print("Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"Available models: {get_available_models()}")
        
        model.close()
    
    # Uncomment the examples you want to run:
    print("Inference utils loaded successfully!")
    print("To test the examples, uncomment the function calls below in the script.")
    
    # example_simple_chat()
    # example_conversation() 
    # example_batch_processing()
    # example_batch_conversations()
    # example_continue_conversations()
    # example_model_info()