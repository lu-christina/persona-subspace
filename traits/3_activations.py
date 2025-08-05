#!/usr/bin/env python3
"""
Activation extraction script for trait responses.

This script reads trait response JSONL files and extracts model activations
using the transformers library. For each trait, it processes all conversations
and saves the activation tensors to .pt files.

The script processes each conversation individually (no input padding) and then
pads the resulting activation tensors for consistent batch dimensions.

Usage:
    uv run traits/3_activations.py \
        --model-name google/gemma-2-27b-it \
        --responses-dir /workspace/traits/responses \
        --output-dir /workspace/traits/activations
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


class TraitActivationExtractor:
    """Extractor for trait response activations using transformers hooks."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-27b-it",
        responses_dir: str = "/workspace/traits/responses",
        output_dir: str = "/workspace/traits/activations",
        layers: Optional[List[int]] = None
    ):
        """
        Initialize the trait activation extractor.
        
        Args:
            model_name: HuggingFace model identifier
            responses_dir: Directory containing trait response JSONL files
            output_dir: Directory to save activation .pt files
            layers: List of layer indices to extract (None for all layers)
        """
        self.model_name = model_name
        self.responses_dir = Path(responses_dir)
        self.output_dir = Path(output_dir)
        self.layers = layers
        
        # Model and tokenizer (loaded once)
        self.model = None
        self.tokenizer = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TraitActivationExtractor with model: {model_name}")
        logger.info(f"Responses directory: {self.responses_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        if layers is not None:
            logger.info(f"Target layers: {layers}")
        else:
            logger.info("Target layers: all layers")
    
    def load_model(self):
        """Load model and tokenizer using probing_utils."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load_model(self.model_name)
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
    
    def extract_trait_activations(self, trait_name: str) -> Dict[str, torch.Tensor]:
        """
        Extract mean activations for all responses of a single trait, grouped by system prompt type.
        Only averages over assistant response tokens, not all tokens.
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            Dictionary containing mean activations for each system prompt type
        """
        logger.info(f"Processing trait '{trait_name}'...")
        
        # Load responses
        responses = self.load_trait_responses(trait_name)
        
        if not responses:
            raise ValueError(f"No responses found for trait '{trait_name}'")
        
        # Initialize running sums and counts for each label
        running_sums = {}
        token_counts = {}
        
        logger.info(f"Extracting activations for {len(responses)} conversations...")
        
        for i, response in enumerate(responses):
            try:
                # Extract conversation and label
                conversation = response['conversation']
                label = response['label']
                
                # Extract full activations for the conversation
                activations = self.extract_conversation_activations(conversation)  # (num_layers, seq_len, hidden_size)
                
                # Get response token indices
                response_indices = get_response_indices(conversation, self.tokenizer)
                
                if not response_indices:
                    logger.warning(f"No response tokens found for conversation {i}")
                    continue
                
                # Extract activations only for response tokens
                response_activations = activations[:, response_indices, :]  # (num_layers, num_response_tokens, hidden_size)
                
                # Initialize running sums for this label if first time
                if label not in running_sums:
                    num_layers, _, hidden_size = activations.shape
                    running_sums[label] = torch.zeros(num_layers, hidden_size, dtype=activations.dtype)
                    token_counts[label] = 0
                
                # Add to running sum
                response_sum = response_activations.sum(dim=1)  # (num_layers, hidden_size)
                running_sums[label] += response_sum
                token_counts[label] += len(response_indices)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(responses)} conversations for '{trait_name}'")
                
            except Exception as e:
                logger.error(f"Error processing conversation {i} for trait '{trait_name}': {e}")
                raise
        
        logger.info(f"Successfully extracted activations for {len(responses)} conversations")
        
        # Compute final means
        mean_activations = {}
        for label in running_sums:
            if token_counts[label] > 0:
                mean_activations[label] = running_sums[label] / token_counts[label]
                logger.info(f"Computed mean activation for '{label}': {mean_activations[label].shape} (averaged over {token_counts[label]} response tokens)")
            else:
                logger.warning(f"No response tokens found for label '{label}'")
        
        # Compute difference vectors if we have the required labels
        if 'pos' in mean_activations and 'neg' in mean_activations:
            mean_activations['pos_neg'] = mean_activations['pos'] - mean_activations['neg']
            logger.info(f"Computed pos - neg difference vector: {mean_activations['pos_neg'].shape}")
        
        if 'pos' in mean_activations and 'default' in mean_activations:
            mean_activations['pos_default'] = mean_activations['pos'] - mean_activations['default']
            logger.info(f"Computed pos - default difference vector: {mean_activations['pos_default'].shape}")
        
        return mean_activations
    
    def save_trait_activations(self, trait_name: str, mean_activations: Dict[str, torch.Tensor]):
        """
        Save mean trait activations to a .pt file.
        
        Args:
            trait_name: Name of the trait
            mean_activations: Dictionary containing mean activations for each system prompt type
        """
        output_file = self.output_dir / f"{trait_name}.pt"
        
        try:
            torch.save(mean_activations, output_file)
            logger.info(f"Saved mean activations to {output_file}")
            
            # Log file size
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error saving activations for trait '{trait_name}': {e}")
            raise
    
    def should_skip_trait(self, trait_name: str) -> bool:
        """
        Check if trait should be skipped (already processed).
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            True if trait should be skipped
        """
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
    
    def process_all_traits(self, skip_existing: bool = True, trait_limit: Optional[int] = None, subset: Optional[str] = None):
        """
        Process all available traits and extract activations.
        
        Args:
            skip_existing: Skip traits with existing output files
            trait_limit: Limit number of traits to process (for testing)
            subset: Filter traits by 'even' or 'odd' indices for parallel processing
        """
        try:
            # Load model once
            self.load_model()
            
            # Get available traits
            trait_names = self.get_available_traits()
            
            if not trait_names:
                logger.error("No trait response files found")
                return
            
            # Apply subset filtering (even/odd) if specified
            if subset is not None:
                if subset.lower() == 'even':
                    trait_names = [trait_names[i] for i in range(0, len(trait_names), 2)]
                    logger.info(f"Processing even-indexed traits: {len(trait_names)} traits")
                elif subset.lower() == 'odd':
                    trait_names = [trait_names[i] for i in range(1, len(trait_names), 2)]
                    logger.info(f"Processing odd-indexed traits: {len(trait_names)} traits")
                else:
                    logger.warning(f"Invalid subset '{subset}'. Must be 'even' or 'odd'. Processing all traits.")
            
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


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Extract activations from trait responses using transformers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python traits/3_activations.py

    # Custom model and directories
    python traits/3_activations.py \\
        --model-name google/gemma-2-9b-it \\
        --responses-dir /path/to/responses \\
        --output-dir /path/to/activations

    # Extract specific layers only
    python traits/3_activations.py \\
        --layers 15 16 17

    # Process even-indexed traits (for parallel processing)
    python traits/3_activations.py --subset even

    # Process odd-indexed traits (for parallel processing)
    python traits/3_activations.py --subset odd

    # Test with limited number of traits
    python traits/3_activations.py \\
        --trait-limit 5

    # Process all traits with auto GPU detection
    python traits/3_activations.py
        """
    )
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='google/gemma-2-27b-it',
                       help='HuggingFace model name (default: google/gemma-2-27b-it)')
    parser.add_argument('--responses-dir', type=str, default='/workspace/traits/responses',
                       help='Directory containing trait response JSONL files')
    parser.add_argument('--output-dir', type=str, default='./traits/data/activations',
                       help='Output directory for activation .pt files')
    parser.add_argument('--layers', type=int, nargs='*', default=None,
                       help='Specific layer indices to extract (default: all layers)')
    
    # Processing options
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process all traits, even if output files exist')
    parser.add_argument('--trait-limit', type=int, default=None,
                       help='Limit number of traits to process (for testing)')
    parser.add_argument('--subset', type=str, choices=['even', 'odd'], default=None,
                       help='Process only even or odd indexed traits for parallel processing')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Responses directory: {args.responses_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Layers: {args.layers if args.layers else 'all'}")
    logger.info(f"  Skip existing: {not args.no_skip_existing}")
    if args.trait_limit:
        logger.info(f"  Trait limit: {args.trait_limit}")
    if args.subset:
        logger.info(f"  Subset: {args.subset}")
    
    try:
        # Create extractor
        extractor = TraitActivationExtractor(
            model_name=args.model_name,
            responses_dir=args.responses_dir,
            output_dir=args.output_dir,
            layers=args.layers
        )
        
        # Process all traits
        extractor.process_all_traits(
            skip_existing=not args.no_skip_existing,
            trait_limit=args.trait_limit,
            subset=args.subset
        )
        
        logger.info("Activation extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())