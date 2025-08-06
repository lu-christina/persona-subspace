#!/usr/bin/env python3
"""
Per-response activation extraction script for trait responses.

This script reads trait response JSONL files and extracts model activations
using the transformers library. For each trait, it processes all conversations
and saves the mean activation for each individual response.

For each response, it computes the mean activation across all response tokens,
resulting in one tensor per response with shape (num_layers, hidden_size).

Usage:
    uv run traits/3_activations_per_token.py \
        --model-name google/gemma-2-27b-it \
        --responses-dir /workspace/traits/responses \
        --output-dir /workspace/traits/activations_per_response
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import jsonlines
import torch
import torch.multiprocessing as mp
import os

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.probing_utils import load_model, extract_full_activations


def get_response_indices(conversation, tokenizer):
    """
    Get every token index of the model's response.
    
    Args:
        conversation: List of dict with 'role' and 'content' keys
        tokenizer: Tokenizer to apply chat template and tokenize
    
    Returns:
        response_indices: list of token positions where the model is responding
    """
    # Apply chat template to the full conversation
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
        # We need to account for any generation prompt tokens that get removed
        assistant_start = before_length
        assistant_end = including_length
        
        # Add these indices to our response list
        response_indices.extend(range(assistant_start, assistant_end))
    
    return response_indices


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TraitActivationExtractorPerResponse:
    """Extractor for per-response trait response activations using transformers hooks."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-27b-it",
        responses_dir: str = "/workspace/traits/responses",
        output_dir: str = "/workspace/traits/activations_per_token",
        layers: Optional[List[int]] = None,
        start_index: int = 0,
        prompt_indices: Optional[List[int]] = None,
        append_mode: bool = False,
    ):
        """
        Initialize the trait activation extractor.
        
        Args:
            model_name: HuggingFace model identifier
            responses_dir: Directory containing trait response JSONL files
            output_dir: Directory to save activation .pt files
            layers: List of layer indices to extract (None for all layers)
            start_index: Index to start processing responses from (skip earlier ones)
            prompt_indices: List of prompt indices to process (None for all)
            append_mode: Whether to append to existing activation files
        """
        self.model_name = model_name
        self.responses_dir = Path(responses_dir)
        self.output_dir = Path(output_dir)
        self.layers = layers
        self.start_index = start_index
        self.prompt_indices = prompt_indices
        self.append_mode = append_mode
        
        # Model and tokenizer (loaded once)
        self.model = None
        self.tokenizer = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TraitActivationExtractorPerResponse with model: {model_name}")
        logger.info(f"Responses directory: {self.responses_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        if layers is not None:
            logger.info(f"Target layers: {layers}")
        else:
            logger.info("Target layers: all layers")
    
    def load_model(self, device=None):
        """Load model and tokenizer using probing_utils."""
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
    
    def load_trait_responses(self, trait_name: str) -> List[Dict]:
        """
        Load responses for a single trait from JSONL file.
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            List of response dictionaries
        """
        jsonl_file = self.responses_dir / f"{trait_name}.jsonl"
        
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Response file not found: {jsonl_file}")
        
        responses = []
        try:
            with jsonlines.open(jsonl_file, mode='r') as reader:
                for response in reader:
                    responses.append(response)
            
            logger.info(f"Loaded {len(responses)} responses for trait '{trait_name}'")
            return responses
            
        except Exception as e:
            logger.error(f"Error loading responses for trait '{trait_name}': {e}")
            raise
    
    def extract_conversation_activations(self, conversation: List[Dict]) -> torch.Tensor:
        """
        Extract activations for a single conversation.
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            Activation tensor of shape (num_layers, num_tokens, hidden_size)
        """
        try:
            # Extract activations using the full conversation
            activations = extract_full_activations(
                model=self.model,
                tokenizer=self.tokenizer,
                conversation=conversation,
                layer=self.layers
            )
            
            return activations
            
        except Exception as e:
            logger.error(f"Error extracting activations for conversation: {e}")
            raise
    
    def process_trait(self, trait_name: str) -> bool:
        """
        Process a single trait.
        
        Args:
            trait_name: Name of the trait to process
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing trait '{trait_name}'")
            
            # Extract activations for this trait
            activations_data = self.extract_trait_activations(trait_name)
            
            # Save activations
            self.save_trait_activations(trait_name, activations_data)
            
            logger.info(f"Successfully completed trait '{trait_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error processing trait '{trait_name}': {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def should_process_response(self, response: Dict, response_idx: int, existing_activations: Dict[str, torch.Tensor]) -> bool:
        """
        Determine if a response should be processed based on filtering criteria.
        
        Args:
            response: Response dictionary
            response_idx: Global response index in the list
            existing_activations: Dictionary of existing activations
            
        Returns:
            True if response should be processed
        """
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
    
    def extract_trait_activations(self, trait_name: str) -> Dict[str, torch.Tensor]:
        """
        Extract mean activations for each individual response using new naming scheme.
        Each response gets its own tensor representing the mean activation across all its response tokens.
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            Dictionary containing mean activations for each individual response
        """
        logger.info(f"Processing trait '{trait_name}'...")
        
        # Load existing activations if in append mode
        existing_activations = {}
        if self.append_mode:
            existing_activations = self.load_existing_activations(trait_name)
            existing_activations = self.convert_old_format_keys(existing_activations)
            logger.info(f"Found {len(existing_activations)} existing activations")
        
        # Load responses
        responses = self.load_trait_responses(trait_name)
        
        if not responses:
            raise ValueError(f"No responses found for trait '{trait_name}'")
        
        # Filter responses to process
        responses_to_process = []
        for i, response in enumerate(responses):
            if self.should_process_response(response, i, existing_activations):
                responses_to_process.append((i, response))
        
        logger.info(f"Processing {len(responses_to_process)} out of {len(responses)} responses")
        
        if not responses_to_process:
            logger.info("No new responses to process")
            return existing_activations
        
        # Extract activations for filtered responses
        new_activations = {}
        
        for i, (global_idx, response) in enumerate(responses_to_process):
            try:
                # Extract response metadata
                conversation = response['conversation']
                label = response['label']
                prompt_index = response.get('prompt_index', 0)
                question_index = response['question_index']
                
                # Create new format key
                key = f"{label}_p{prompt_index}_q{question_index}"
                
                # Extract full activations for the conversation
                activations = self.extract_conversation_activations(conversation)  # (num_layers, seq_len, hidden_size)
                
                # Get response token indices
                response_indices = get_response_indices(conversation, self.tokenizer)
                
                if not response_indices:
                    logger.warning(f"No response tokens found for conversation {global_idx} (key: {key})")
                    continue
                
                # Extract activations only for response tokens
                response_token_activations = activations[:, response_indices, :]  # (num_layers, num_response_tokens, hidden_size)
                
                # Compute mean across response tokens for this conversation
                mean_response_activation = response_token_activations.mean(dim=1)  # (num_layers, hidden_size)
                
                # Store with new format key
                new_activations[key] = mean_response_activation
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(responses_to_process)} new conversations for '{trait_name}'")
                
            except Exception as e:
                logger.error(f"Error processing conversation {global_idx} for trait '{trait_name}': {e}")
                raise
        
        logger.info(f"Successfully extracted {len(new_activations)} new activations")
        
        # Combine with existing activations
        combined_activations = existing_activations.copy()
        combined_activations.update(new_activations)
        
        logger.info(f"Total activations for trait '{trait_name}': {len(combined_activations)}")
        return combined_activations
    
    def save_trait_activations(self, trait_name: str, mean_activations: Dict[str, torch.Tensor]):
        """
        Save per-token mean trait activations to a .pt file.
        
        Args:
            trait_name: Name of the trait
            mean_activations: Dictionary containing mean activations for each token position
        """
        output_file = self.output_dir / f"{trait_name}.pt"
        
        try:
            torch.save(mean_activations, output_file)
            logger.info(f"Saved per-token mean activations to {output_file}")
            
            # Log file size
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error saving activations for trait '{trait_name}': {e}")
            raise
    
    def load_existing_activations(self, trait_name: str) -> Dict[str, torch.Tensor]:
        """
        Load existing activations for a trait if they exist.
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            Dictionary of existing activations (empty if file doesn't exist)
        """
        output_file = self.output_dir / f"{trait_name}.pt"
        
        if not output_file.exists():
            return {}
        
        try:
            existing_activations = torch.load(output_file, map_location='cpu')
            logger.info(f"Loaded {len(existing_activations)} existing activations for trait '{trait_name}'")
            return existing_activations
            
        except Exception as e:
            logger.error(f"Error loading existing activations for trait '{trait_name}': {e}")
            return {}
    
    def convert_old_format_keys(self, activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert old format activation keys to new format.
        
        Old format: pos_0, pos_1, neg_0, neg_1, default_0, etc.
        New format: pos_p0_q0, pos_p0_q1, neg_p0_q0, neg_p0_q1, default_p0_q0, etc.
        
        Args:
            activations: Dictionary with old format keys
            
        Returns:
            Dictionary with new format keys
        """
        converted = {}
        
        for old_key, tensor in activations.items():
            # Check if key is already in new format
            if '_p' in old_key and '_q' in old_key:
                converted[old_key] = tensor
                continue
            
            # Parse old format key (e.g., "pos_5" -> label="pos", idx=5)
            parts = old_key.split('_')
            if len(parts) != 2:
                # Unknown format, keep as-is
                converted[old_key] = tensor
                continue
            
            label, idx_str = parts
            try:
                idx = int(idx_str)
            except ValueError:
                # Not a number, keep as-is
                converted[old_key] = tensor
                continue
            
            # Convert to new format
            # Old format used sequential numbering across all responses
            # For 20 questions per prompt, we can derive prompt_index and question_index
            prompt_index = 0  # Old format assumed first prompt variant
            question_index = idx % 20  # Questions were 0-19 within each prompt
            
            new_key = f"{label}_p{prompt_index}_q{question_index}"
            converted[new_key] = tensor
            
            if old_key != new_key:
                logger.debug(f"Converted activation key: '{old_key}' -> '{new_key}'")
        
        return converted
    
    def should_skip_trait(self, trait_name: str) -> bool:
        """
        Check if trait should be skipped (already processed).
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            True if trait should be skipped
        """
        if self.append_mode:
            return False  # Never skip in append mode
        
        output_file = self.output_dir / f"{trait_name}.pt"
        return output_file.exists()
    
    def get_available_traits(self) -> List[str]:
        """
        Get list of available trait names from response files.
        
        Returns:
            List of trait names
        """
        trait_files = list(self.responses_dir.glob("*.jsonl"))
        trait_names = [f.stem for f in trait_files]
        return sorted(trait_names)
    
    def process_all_traits(self, skip_existing: bool = True, trait_limit: Optional[int] = None):
        """
        Process all available traits and extract activations.
        
        Args:
            skip_existing: Skip traits with existing output files
            trait_limit: Limit number of traits to process (for testing)
        """
        try:
            # Load model once
            self.load_model()
            
            # Get available traits
            trait_names = self.get_available_traits()
            
            if not trait_names:
                logger.error("No trait response files found")
                return
            
            
            # Apply limit if specified
            if trait_limit is not None:
                trait_names = trait_names[:trait_limit]
                logger.info(f"Processing limited set of {len(trait_names)} traits")
            
            # Filter traits if skipping existing
            if skip_existing:
                traits_to_process = []
                for trait_name in trait_names:
                    if self.should_skip_trait(trait_name):
                        logger.info(f"Skipping trait '{trait_name}' (already exists)")
                        continue
                    traits_to_process.append(trait_name)
            else:
                traits_to_process = trait_names
            
            logger.info(f"Processing {len(traits_to_process)} traits")
            
            # Process each trait sequentially
            completed_count = 0
            failed_count = 0
            
            for i, trait_name in enumerate(traits_to_process, 1):
                logger.info(f"Processing trait {i}/{len(traits_to_process)}: {trait_name}")
                
                try:
                    success = self.process_trait(trait_name)
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1
                
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception processing trait {trait_name}: {e}")
            
            logger.info(f"Processing summary: {completed_count} completed, {failed_count} failed")
        
        finally:
            # Final cleanup
            self.close_model()
            logger.info("Final cleanup completed")


def process_traits_on_gpu(gpu_id, trait_names, args, prompt_indices=None):
    """Process a subset of traits on a specific GPU."""
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
    
    logger.info(f"Starting processing on GPU {gpu_id} with {len(trait_names)} traits")
    
    try:
        # Create extractor for this GPU
        extractor = TraitActivationExtractorPerResponse(
            model_name=args.model_name,
            responses_dir=args.responses_dir,
            output_dir=args.output_dir,
            layers=args.layers,
            start_index=args.start_index,
            prompt_indices=prompt_indices,
            append_mode=args.append_mode
        )
        
        # Load model on this specific GPU
        extractor.load_model(device=f"cuda:{gpu_id}")
        
        # Process assigned traits
        completed_count = 0
        failed_count = 0
        
        for i, trait_name in enumerate(trait_names, 1):
            logger.info(f"Processing trait {i}/{len(trait_names)}: {trait_name}")
            
            try:
                success = extractor.process_trait(trait_name)
                if success:
                    completed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Exception processing trait {trait_name}: {e}")
        
        logger.info(f"GPU {gpu_id} completed: {completed_count} successful, {failed_count} failed")
        
    except Exception as e:
        logger.error(f"Fatal error on GPU {gpu_id}: {e}")
        
    finally:
        # Cleanup
        if 'extractor' in locals():
            extractor.close_model()
        logger.info(f"GPU {gpu_id} cleanup completed")


def run_multi_gpu_processing(args, prompt_indices):
    """Run multi-GPU processing with trait distribution."""
    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.num_gpus))
    
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Get available traits
    responses_dir = Path(args.responses_dir)
    trait_files = list(responses_dir.glob("*.jsonl"))
    trait_names = [f.stem for f in trait_files]
    trait_names = sorted(trait_names)
    
    if not trait_names:
        logger.error("No trait response files found")
        return 1
    
    # Apply filtering
    if args.trait_limit:
        trait_names = trait_names[:args.trait_limit]
    
    
    # Filter out existing traits if needed (but not in append mode)
    if not args.no_skip_existing and not args.append_mode:
        output_dir = Path(args.output_dir)
        traits_to_process = []
        for trait_name in trait_names:
            output_file = output_dir / f"{trait_name}.pt"
            if not output_file.exists():
                traits_to_process.append(trait_name)
            else:
                logger.info(f"Skipping trait '{trait_name}' (already exists)")
        trait_names = traits_to_process
    
    if not trait_names:
        logger.info("No traits to process")
        return 0
    
    logger.info(f"Processing {len(trait_names)} traits across {len(gpu_ids)} GPUs")
    
    # Distribute traits across GPUs
    traits_per_gpu = len(trait_names) // len(gpu_ids)
    remainder = len(trait_names) % len(gpu_ids)
    
    trait_chunks = []
    start_idx = 0
    
    for i, gpu_id in enumerate(gpu_ids):
        # Give extra traits to first few GPUs if there's a remainder
        chunk_size = traits_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        chunk = trait_names[start_idx:end_idx]
        trait_chunks.append((gpu_id, chunk))
        
        logger.info(f"GPU {gpu_id}: {len(chunk)} traits ({chunk[0] if chunk else 'none'} to {chunk[-1] if chunk else 'none'})")
        start_idx = end_idx
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Launch processes
    processes = []
    for gpu_id, chunk in trait_chunks:
        if chunk:  # Only launch if there are traits to process
            p = mp.Process(
                target=process_traits_on_gpu,
                args=(gpu_id, chunk, args, prompt_indices)
            )
            p.start()
            processes.append(p)
    
    # Wait for all processes to complete
    logger.info(f"Launched {len(processes)} GPU processes")
    for p in processes:
        p.join()
    
    logger.info("Multi-GPU processing completed!")
    return 0


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Extract per-response activations from trait responses using transformers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python traits/3_activations_per_token.py

    # Custom model and directories
    python traits/3_activations_per_token.py \\
        --model-name google/gemma-2-9b-it \\
        --responses-dir /path/to/responses \\
        --output-dir /path/to/activations

    # Extract specific layers only
    python traits/3_activations_per_token.py \\
        --layers 15 16 17


    # Test with limited number of traits
    python traits/3_activations_per_token.py \\
        --trait-limit 5

    # Multi-GPU processing with 2 H100s
    python traits/3_activations_per_token.py \\
        --multi-gpu --num-gpus 2

    # Multi-GPU with specific GPU IDs
    python traits/3_activations_per_token.py \\
        --multi-gpu --gpu-ids "0,1"

        """
    )
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='google/gemma-2-27b-it',
                       help='HuggingFace model name (default: google/gemma-2-27b-it)')
    parser.add_argument('--responses-dir', type=str, default='/workspace/traits/responses',
                       help='Directory containing trait response JSONL files')
    parser.add_argument('--output-dir', type=str, default='/root/git/persona-subspace/traits/data/response_activations',
                       help='Output directory for activation .pt files')
    parser.add_argument('--layers', type=int, nargs='*', default=None,
                       help='Specific layer indices to extract (default: all layers)')
    
    # Processing options
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process all traits, even if output files exist')
    parser.add_argument('--trait-limit', type=int, default=None,
                       help='Limit number of traits to process (for testing)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start processing responses from this index (skip earlier ones)')
    parser.add_argument('--prompt-indices', type=str, default=None,
                       help='Comma-separated list of prompt indices to process (e.g., "1,2,3,4")')
    parser.add_argument('--append-mode', action='store_true',
                       help='Append to existing activation files instead of overwriting')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # Multi-GPU options
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multi-GPU processing (one GPU per trait)')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs to use for parallel processing (default: 2)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='Comma-separated list of GPU IDs to use (e.g., "0,1")')
    
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
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Responses directory: {args.responses_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Layers: {args.layers if args.layers else 'all'}")
    logger.info(f"  Start index: {args.start_index}")
    logger.info(f"  Prompt indices: {prompt_indices if prompt_indices else 'all'}")
    logger.info(f"  Append mode: {args.append_mode}")
    logger.info(f"  Skip existing: {not args.no_skip_existing}")
    if args.trait_limit:
        logger.info(f"  Trait limit: {args.trait_limit}")
    
    try:
        if args.multi_gpu:
            # Multi-GPU processing
            return run_multi_gpu_processing(args, prompt_indices)
        else:
            # Single GPU processing (original code)
            extractor = TraitActivationExtractorPerResponse(
                model_name=args.model_name,
                responses_dir=args.responses_dir,
                output_dir=args.output_dir,
                layers=args.layers,
                start_index=args.start_index,
                prompt_indices=prompt_indices,
                append_mode=args.append_mode
            )
            
            # Process all traits
            extractor.process_all_traits(
                skip_existing=not args.no_skip_existing,
                trait_limit=args.trait_limit
            )
            
            logger.info("Per-response activation extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())