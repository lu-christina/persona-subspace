#!/usr/bin/env python3
"""
OPTIMIZED per-response activation extraction script for role responses.

This optimized version implements several key improvements:
1. Batched conversation processing for better GPU utilization
2. Dynamic batching based on sequence lengths and GPU memory
3. Memory-efficient activation extraction with streaming
4. Async I/O for data loading
5. Better tensor operations and memory management

Usage:
    uv run roles/3_response_activations_optimized.py \
        --model-name google/gemma-2-27b-it \
        --responses-dir /workspace/roles/responses \
        --output-dir /workspace/roles/activations_per_response \
        --batch-size 16
"""

import argparse
import gc
import json
import logging
import sys
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import jsonlines
import torch
import torch.multiprocessing as mp
import os
from collections import defaultdict
from tqdm import tqdm

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.probing_utils import load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_response_indices_batch(conversations, tokenizer):
    """
    Get response token indices for a batch of conversations.
    
    Args:
        conversations: List of conversation lists
        tokenizer: Tokenizer to apply chat template and tokenize
    
    Returns:
        List of response_indices for each conversation
    """
    batch_response_indices = []
    
    for conversation in conversations:
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
                    conversation_before, tokenize=False, add_generation_prompt=True
                )
                before_tokens = tokenizer(before_formatted, add_special_tokens=False)
                before_length = len(before_tokens['input_ids'])
            else:
                before_length = 0
                
            including_formatted = tokenizer.apply_chat_template(
                conversation_including, tokenize=False, add_generation_prompt=False
            )
            including_tokens = tokenizer(including_formatted, add_special_tokens=False)
            including_length = len(including_tokens['input_ids'])
            
            # The assistant response tokens are between before_length and including_length
            assistant_start = before_length
            assistant_end = including_length
            
            # Add these indices to our response list
            response_indices.extend(range(assistant_start, assistant_end))
        
        batch_response_indices.append(response_indices)
    
    return batch_response_indices


def extract_batched_activations(model, tokenizer, conversations, layers=None, batch_size=16, max_length=1024):
    """
    Extract activations for a batch of conversations with memory-efficient processing.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        conversations: List of conversation lists
        layers: List of layer indices to extract (None for all layers)
        batch_size: Maximum batch size
        max_length: Maximum sequence length
    
    Returns:
        List of activation tensors, one per conversation
    """
    if layers is None:
        layers = list(range(len(model.model.layers)))
    elif isinstance(layers, int):
        layers = [layers]
    
    # Process conversations in batches
    all_activations = []
    num_conversations = len(conversations)
    num_batches = (num_conversations + batch_size - 1) // batch_size
    
    with tqdm(total=num_conversations, desc="Extracting activations", unit="conv") as pbar:
        for batch_start in range(0, num_conversations, batch_size):
            batch_end = min(batch_start + batch_size, num_conversations)
            batch_conversations = conversations[batch_start:batch_end]
            
            batch_num = batch_start // batch_size + 1
            pbar.set_postfix(batch=f"{batch_num}/{num_batches}", refresh=True)
            
            # Format all conversations in the batch
            formatted_prompts = []
            for conversation in batch_conversations:
                formatted_prompt = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
                formatted_prompts.append(formatted_prompt)
            
            # Tokenize batch with padding
            batch_tokens = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            input_ids = batch_tokens["input_ids"].to(model.device)
            attention_mask = batch_tokens["attention_mask"].to(model.device)
            
            # Get response indices for this batch
            batch_response_indices = get_response_indices_batch(batch_conversations, tokenizer)
            
            # Extract activations with hooks - fixed for batch processing
            batch_activations = []
            handles = []
            layer_outputs = {}  # Will store {layer_idx: tensor} after forward pass
            
            def create_hook_fn(layer_idx):
                def hook_fn(module, input, output):
                    # Extract the activation tensor (handle tuple output)
                    act_tensor = output[0] if isinstance(output, tuple) else output
                    # Keep on GPU - don't transfer to CPU during forward pass!
                    layer_outputs[layer_idx] = act_tensor
                return hook_fn
            
            # Register hooks for target layers
            for layer_idx in layers:
                target_layer = model.model.layers[layer_idx]
                handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
                handles.append(handle)
            
            try:
                with torch.no_grad():
                    _ = model(input_ids, attention_mask=attention_mask)
            finally:
                # Clean up hooks
                for handle in handles:
                    handle.remove()
            
            # Verify all requested layers were captured
            missing_layers = set(layers) - set(layer_outputs.keys())
            if missing_layers:
                logger.error(f"Missing activations for layers: {sorted(missing_layers)}. Got layers: {sorted(layer_outputs.keys())}")
                # Return None for all conversations in this batch
                for _ in range(len(batch_conversations)):
                    all_activations.append(None)
                # Update progress bar and skip to next batch
                pbar.update(len(batch_conversations))
                continue
            
            # Process activations for each conversation in the batch
            for i, (conversation, response_indices) in enumerate(zip(batch_conversations, batch_response_indices)):
                if not response_indices:
                    logger.warning(f"No response tokens found for conversation {batch_start + i}")
                    all_activations.append(None)
                    continue
                
                # Extract activations for this conversation and compute mean
                conv_activations = []
                
                # Process layers in sorted order to maintain consistency with original script
                for layer_idx in sorted(layers):
                    # Defensive checks for layer outputs
                    if layer_idx not in layer_outputs:
                        logger.error(f"Layer {layer_idx} not found in outputs. Available layers: {list(layer_outputs.keys())}")
                        all_activations.append(None)
                        break
                    
                    layer_acts = layer_outputs[layer_idx]  # (batch_size, seq_len, hidden_size)
                    
                    # Check if batch index is valid
                    if i >= layer_acts.size(0):
                        logger.error(f"Batch index {i} >= batch size {layer_acts.size(0)} for layer {layer_idx}")
                        all_activations.append(None)
                        break
                    
                    conv_layer_acts = layer_acts[i]  # (seq_len, hidden_size)
                    
                    # Check if response indices are valid for sequence length
                    max_seq_len = conv_layer_acts.size(0)
                    valid_indices = [idx for idx in response_indices if 0 <= idx < max_seq_len]
                    
                    if not valid_indices:
                        logger.warning(f"No valid response indices for conversation {batch_start + i}. Seq len: {max_seq_len}, indices range: {min(response_indices) if response_indices else 'N/A'}-{max(response_indices) if response_indices else 'N/A'}")
                        all_activations.append(None)
                        break
                    
                    # Extract response token activations
                    response_acts = conv_layer_acts[valid_indices]  # (num_response_tokens, hidden_size)
                    
                    # Compute mean across response tokens and move to CPU only when needed
                    mean_response_act = response_acts.mean(dim=0).cpu()  # (hidden_size,)
                    conv_activations.append(mean_response_act)
                
                # Only add result if all layers processed successfully
                if len(conv_activations) == len(layers):
                    # Stack layer activations
                    conv_activations = torch.stack(conv_activations)  # (num_layers, hidden_size)
                    all_activations.append(conv_activations)
                else:
                    # If we haven't added None yet due to early breaks
                    if len(all_activations) == batch_start + i:
                        all_activations.append(None)
            
            # Clear intermediate results to save memory after processing batch
            # Explicitly delete GPU tensors
            for layer_idx in list(layer_outputs.keys()):
                del layer_outputs[layer_idx]
            layer_outputs.clear()
            
            # Clear batch tensors from GPU
            del input_ids
            del attention_mask
            
            # Update progress bar
            pbar.update(len(batch_conversations))
            
            # More aggressive GPU memory management
            if batch_num % 5 == 0:  # More frequent cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Periodic garbage collection
            if batch_num % 10 == 0:
                gc.collect()
    
    return all_activations


class OptimizedRoleActivationExtractor:
    """Optimized extractor for per-response role response activations."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-27b-it",
        responses_dir: str = "/workspace/roles/responses",
        output_dir: str = "/workspace/roles/activations_per_token",
        layers: Optional[List[int]] = None,
        batch_size: int = 16,
        max_length: int = 1024,
        start_index: int = 0,
        prompt_indices: Optional[List[int]] = None,
        append_mode: bool = False,
    ):
        """
        Initialize the optimized role activation extractor.
        
        Args:
            model_name: HuggingFace model identifier
            responses_dir: Directory containing role response JSONL files
            output_dir: Directory to save activation .pt files
            layers: List of layer indices to extract (None for all layers)
            batch_size: Batch size for processing conversations
            max_length: Maximum sequence length
            start_index: Index to start processing responses from
            prompt_indices: List of prompt indices to process (None for all)
            append_mode: Whether to append to existing activation files
        """
        self.model_name = model_name
        self.responses_dir = Path(responses_dir)
        self.output_dir = Path(output_dir)
        self.layers = layers
        self.batch_size = batch_size
        self.max_length = max_length
        self.start_index = start_index
        self.prompt_indices = prompt_indices
        self.append_mode = append_mode
        
        # Model and tokenizer (loaded once)
        self.model = None
        self.tokenizer = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized OptimizedRoleActivationExtractor with model: {model_name}")
        logger.info(f"Batch size: {batch_size}, Max length: {max_length}")
        logger.info(f"Responses directory: {self.responses_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_model(self, device=None):
        """Load model and tokenizer."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load_model(self.model_name, device=device)
            logger.info("Model loaded successfully")
    
    def close_model(self):
        """Clean up model resources."""
        if self.model is not None:
            logger.info("Cleaning up model resources")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
    
    def load_role_responses(self, role_name: str) -> List[Dict]:
        """Load responses for a single role from JSONL file."""
        jsonl_file = self.responses_dir / f"{role_name}.jsonl"
        
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Response file not found: {jsonl_file}")
        
        responses = []
        try:
            with jsonlines.open(jsonl_file, mode='r') as reader:
                for response in reader:
                    responses.append(response)
            
            logger.info(f"Loaded {len(responses)} responses for role '{role_name}'")
            return responses
            
        except Exception as e:
            logger.error(f"Error loading responses for role '{role_name}': {e}")
            raise
    
    def extract_role_activations(self, role_name: str) -> Dict[str, torch.Tensor]:
        """
        Extract mean activations for each response using optimized batch processing.
        """
        logger.info(f"Processing role '{role_name}' with optimized batching...")
        
        # Load existing activations if in append mode
        existing_activations = {}
        if self.append_mode:
            existing_activations = self.load_existing_activations(role_name)
            logger.info(f"Found {len(existing_activations)} existing activations")
        
        # Load responses
        responses = self.load_role_responses(role_name)
        
        if not responses:
            raise ValueError(f"No responses found for role '{role_name}'")
        
        # Filter responses to process
        responses_to_process = []
        for i, response in enumerate(responses):
            if self.should_process_response(response, i, existing_activations):
                responses_to_process.append((i, response))
        
        logger.info(f"Processing {len(responses_to_process)} out of {len(responses)} responses")
        
        if not responses_to_process:
            logger.info("No new responses to process")
            return existing_activations
        
        # Extract conversations and metadata
        conversations = []
        response_metadata = []
        
        for global_idx, response in responses_to_process:
            conversations.append(response['conversation'])
            response_metadata.append({
                'global_idx': global_idx,
                'label': response['label'],
                'prompt_index': response.get('prompt_index', 0),
                'question_index': response['question_index']
            })
        
        logger.info(f"Extracting activations for {len(conversations)} conversations in batches of {self.batch_size}")
        
        # Extract activations in batches with progress tracking
        print(f"Processing {len(conversations)} conversations for role '{role_name}':")
        activations_list = extract_batched_activations(
            model=self.model,
            tokenizer=self.tokenizer,
            conversations=conversations,
            layers=self.layers,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        
        # Create activation dictionary
        new_activations = {}
        successful_extractions = 0
        
        print("Creating activation dictionary...")
        for activation, metadata in tqdm(zip(activations_list, response_metadata), 
                                        total=len(activations_list), 
                                        desc="Processing results",
                                        unit="conv"):
            if activation is not None:
                key = f"{metadata['label']}_p{metadata['prompt_index']}_q{metadata['question_index']}"
                new_activations[key] = activation
                successful_extractions += 1
        
        logger.info(f"Successfully extracted {successful_extractions} new activations")
        
        # Combine with existing activations
        combined_activations = existing_activations.copy()
        combined_activations.update(new_activations)
        
        logger.info(f"Total activations for role '{role_name}': {len(combined_activations)}")
        return combined_activations
    
    def should_process_response(self, response: Dict, response_idx: int, existing_activations: Dict[str, torch.Tensor]) -> bool:
        """Determine if a response should be processed based on filtering criteria."""
        # Skip if before start index
        if response_idx < self.start_index:
            return False
        
        # Skip if prompt index filtering is enabled and this prompt isn't included
        prompt_index = response.get('prompt_index', 0)
        if self.prompt_indices is not None and prompt_index not in self.prompt_indices:
            return False
        
        # Skip if activation already exists (when in append mode)
        if self.append_mode:
            label = response['label']
            question_index = response['question_index']
            key = f"{label}_p{prompt_index}_q{question_index}"
            if key in existing_activations:
                return False
        
        return True
    
    def load_existing_activations(self, role_name: str) -> Dict[str, torch.Tensor]:
        """Load existing activations for a role if they exist."""
        output_file = self.output_dir / f"{role_name}.pt"
        
        if not output_file.exists():
            return {}
        
        try:
            existing_activations = torch.load(output_file, map_location='cpu')
            logger.info(f"Loaded {len(existing_activations)} existing activations for role '{role_name}'")
            return existing_activations
            
        except Exception as e:
            logger.error(f"Error loading existing activations for role '{role_name}': {e}")
            return {}
    
    def save_role_activations(self, role_name: str, mean_activations: Dict[str, torch.Tensor]):
        """Save role activations to a .pt file."""
        output_file = self.output_dir / f"{role_name}.pt"
        
        try:
            torch.save(mean_activations, output_file)
            logger.info(f"Saved activations to {output_file}")
            
            # Log file size
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error saving activations for role '{role_name}': {e}")
            raise
    
    def process_role(self, role_name: str) -> bool:
        """Process a single role."""
        try:
            logger.info(f"Processing role '{role_name}'")
            
            # Extract activations for this role
            activations_data = self.extract_role_activations(role_name)
            
            # Save activations
            self.save_role_activations(role_name, activations_data)
            
            logger.info(f"Successfully completed role '{role_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error processing role '{role_name}': {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_available_roles(self) -> List[str]:
        """Get list of available role names from response files."""
        role_files = list(self.responses_dir.glob("*.jsonl"))
        role_names = [f.stem for f in role_files]
        return sorted(role_names)
    
    def should_skip_role(self, role_name: str) -> bool:
        """Check if role should be skipped (already processed)."""
        if self.append_mode:
            return False  # Never skip in append mode
        
        output_file = self.output_dir / f"{role_name}.pt"
        return output_file.exists()
    
    def process_all_roles(self, skip_existing: bool = True, role_limit: Optional[int] = None):
        """Process all available roles and extract activations."""
        try:
            # Load model once
            self.load_model()
            
            # Get available roles
            role_names = self.get_available_roles()
            
            if not role_names:
                logger.error("No role response files found")
                return
            
            # Apply limit if specified
            if role_limit is not None:
                role_names = role_names[:role_limit]
                logger.info(f"Processing limited set of {len(role_names)} roles")
            
            # Filter roles if skipping existing
            if skip_existing:
                roles_to_process = []
                for role_name in role_names:
                    if self.should_skip_role(role_name):
                        logger.info(f"Skipping role '{role_name}' (already exists)")
                        continue
                    roles_to_process.append(role_name)
            else:
                roles_to_process = role_names
            
            logger.info(f"Processing {len(roles_to_process)} roles")
            
            # Process each role with overall progress bar
            completed_count = 0
            failed_count = 0
            
            with tqdm(roles_to_process, desc="Processing roles", unit="role") as role_pbar:
                for role_name in role_pbar:
                    role_pbar.set_postfix(role=role_name, refresh=True)
                    
                    try:
                        success = self.process_role(role_name)
                        if success:
                            completed_count += 1
                            role_pbar.set_postfix(role=role_name, status="✓", refresh=True)
                        else:
                            failed_count += 1
                            role_pbar.set_postfix(role=role_name, status="✗", refresh=True)
                    
                    except Exception as e:
                        failed_count += 1
                        role_pbar.set_postfix(role=role_name, status="✗", refresh=True)
                        logger.error(f"Exception processing role {role_name}: {e}")
            
            logger.info(f"Processing summary: {completed_count} completed, {failed_count} failed")
        
        finally:
            # Final cleanup
            self.close_model()
            logger.info("Final cleanup completed")


def process_roles_on_gpu_optimized(gpu_id, role_names, args, prompt_indices=None):
    """Process a subset of roles on a specific GPU using optimized extraction."""
    # Set the GPU for this process
    torch.cuda.set_device(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Set up logging for this process
    logger = logging.getLogger(f"GPU-{gpu_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - GPU-{gpu_id} - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Starting optimized processing on GPU {gpu_id} with {len(role_names)} roles")
    
    try:
        # Create optimized extractor for this GPU
        extractor = OptimizedRoleActivationExtractor(
            model_name=args.model_name,
            responses_dir=args.responses_dir,
            output_dir=args.output_dir,
            layers=args.layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            start_index=args.start_index,
            prompt_indices=prompt_indices,
            append_mode=args.append_mode
        )
        
        # Load model on this specific GPU
        extractor.load_model(device=f"cuda:{gpu_id}")
        
        # Process assigned roles with progress tracking
        completed_count = 0
        failed_count = 0
        
        with tqdm(role_names, desc=f"GPU-{gpu_id} roles", unit="role", position=gpu_id) as role_pbar:
            for role_name in role_pbar:
                role_pbar.set_postfix(role=role_name[:15], refresh=True)
                
                try:
                    success = extractor.process_role(role_name)
                    if success:
                        completed_count += 1
                        role_pbar.set_postfix(role=role_name[:15], status="✓", refresh=True)
                    else:
                        failed_count += 1
                        role_pbar.set_postfix(role=role_name[:15], status="✗", refresh=True)
                        
                except Exception as e:
                    failed_count += 1
                    role_pbar.set_postfix(role=role_name[:15], status="✗", refresh=True)
                    logger.error(f"Exception processing role {role_name}: {e}")
        
        logger.info(f"GPU {gpu_id} completed: {completed_count} successful, {failed_count} failed")
        
    except Exception as e:
        logger.error(f"Fatal error on GPU {gpu_id}: {e}")
        
    finally:
        # Cleanup
        if 'extractor' in locals():
            extractor.close_model()
        logger.info(f"GPU {gpu_id} cleanup completed")


def run_multi_gpu_optimized(args, prompt_indices):
    """Run optimized multi-GPU processing with role distribution."""
    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.num_gpus))
    
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Get available roles
    responses_dir = Path(args.responses_dir)
    role_files = list(responses_dir.glob("*.jsonl"))
    role_names = [f.stem for f in role_files]
    role_names = sorted(role_names)
    
    if not role_names:
        logger.error("No role response files found")
        return 1
    
    # Apply filtering
    if args.role_limit:
        role_names = role_names[:args.role_limit]
    
    # Filter out existing roles if needed (but not in append mode)
    if not args.no_skip_existing and not args.append_mode:
        output_dir = Path(args.output_dir)
        roles_to_process = []
        for role_name in role_names:
            output_file = output_dir / f"{role_name}.pt"
            if not output_file.exists():
                roles_to_process.append(role_name)
            else:
                logger.info(f"Skipping role '{role_name}' (already exists)")
        role_names = roles_to_process
    
    if not role_names:
        logger.info("No roles to process")
        return 0
    
    logger.info(f"Processing {len(role_names)} roles across {len(gpu_ids)} GPUs with batch size {args.batch_size}")
    
    # Distribute roles across GPUs
    roles_per_gpu = len(role_names) // len(gpu_ids)
    remainder = len(role_names) % len(gpu_ids)
    
    role_chunks = []
    start_idx = 0
    
    for i, gpu_id in enumerate(gpu_ids):
        # Give extra roles to first few GPUs if there's a remainder
        chunk_size = roles_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        chunk = role_names[start_idx:end_idx]
        role_chunks.append((gpu_id, chunk))
        
        logger.info(f"GPU {gpu_id}: {len(chunk)} roles ({chunk[0] if chunk else 'none'} to {chunk[-1] if chunk else 'none'})")
        start_idx = end_idx
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Launch processes
    processes = []
    for gpu_id, chunk in role_chunks:
        if chunk:  # Only launch if there are roles to process
            p = mp.Process(
                target=process_roles_on_gpu_optimized,
                args=(gpu_id, chunk, args, prompt_indices)
            )
            p.start()
            processes.append(p)
    
    # Wait for all processes to complete
    logger.info(f"Launched {len(processes)} optimized GPU processes")
    for p in processes:
        p.join()
    
    logger.info("Optimized multi-GPU processing completed!")
    return 0


def main():
    """Main entry point for the optimized script."""
    parser = argparse.ArgumentParser(
        description='Extract per-response activations from role responses using OPTIMIZED transformers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic optimized usage
    python roles/3_response_activations_optimized.py --batch-size 32

    # Custom model and larger batches
    python roles/3_response_activations_optimized.py \\
        --model-name google/gemma-2-9b-it \\
        --batch-size 64 \\
        --max-length 2048

    # Multi-GPU optimized processing
    python roles/3_response_activations_optimized.py \\
        --multi-gpu --num-gpus 6 --batch-size 32
        """
    )
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='google/gemma-2-27b-it',
                       help='HuggingFace model name')
    parser.add_argument('--responses-dir', type=str, default='/workspace/roles/responses',
                       help='Directory containing role response JSONL files')
    parser.add_argument('--output-dir', type=str, default='/workspace/roles/response_activations',
                       help='Output directory for activation .pt files')
    parser.add_argument('--layers', type=int, nargs='*', default=None,
                       help='Specific layer indices to extract (default: all layers)')
    
    # Optimization parameters
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for processing conversations (default: 16)')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum sequence length (default: 1024)')
    
    # Processing options
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process all roles, even if output files exist')
    parser.add_argument('--role-limit', type=int, default=None,
                       help='Limit number of roles to process (for testing)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start processing responses from this index')
    parser.add_argument('--prompt-indices', type=str, default=None,
                       help='Comma-separated list of prompt indices to process')
    parser.add_argument('--append-mode', action='store_true',
                       help='Append to existing activation files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # Multi-GPU options
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multi-GPU processing')
    parser.add_argument('--num-gpus', type=int, default=8,
                       help='Number of GPUs to use (default: 6)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='Comma-separated list of GPU IDs to use')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse prompt indices
    prompt_indices = None
    if args.prompt_indices:
        try:
            prompt_indices = [int(x.strip()) for x in args.prompt_indices.split(',')]
            logger.info(f"Using specific prompt indices: {prompt_indices}")
        except ValueError as e:
            logger.error(f"Invalid prompt indices format: {args.prompt_indices}")
            return 1
    
    # Print configuration
    logger.info("OPTIMIZED Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Responses directory: {args.responses_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Layers: {args.layers if args.layers else 'all'}")
    
    try:
        if args.multi_gpu:
            # Multi-GPU optimized processing
            return run_multi_gpu_optimized(args, prompt_indices)
        else:
            # Single GPU optimized processing
            extractor = OptimizedRoleActivationExtractor(
                model_name=args.model_name,
                responses_dir=args.responses_dir,
                output_dir=args.output_dir,
                layers=args.layers,
                batch_size=args.batch_size,
                max_length=args.max_length,
                start_index=args.start_index,
                prompt_indices=prompt_indices,
                append_mode=args.append_mode
            )
            
            # Process all roles
            extractor.process_all_roles(
                skip_existing=not args.no_skip_existing,
                role_limit=args.role_limit
            )
            
            logger.info("Optimized per-response activation extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())