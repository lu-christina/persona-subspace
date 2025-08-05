#!/usr/bin/env python3
"""
Batch inference script for generating trait responses using vLLM.

This script processes trait files and generates model responses for positive, negative,
and default (baseline) instructions using a specified vLLM model across multiple GPUs.

For each trait, it generates responses to the first 20 questions with three different
instruction types:
- pos: Using positive trait instruction 
- neg: Using negative trait instruction
- default: No system prompt (baseline)

Results are saved as JSONL files (one per trait) in the specified output directory.

Usage:
    uv run traits/2_generate_responses.py \
        --model-name google/gemma-2-9b-it \
        --traits-dir /root/git/persona-subspace/traits/data \
        --output-dir /workspace/traits
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import jsonlines

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.inference_utils import load_vllm_model, batch_conversation_chat, close_vllm_model, cleanup_all_models

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TraitResponseGenerator:
    """Generator for trait-based model responses using vLLM batch inference."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-27b-it",
        traits_dir: str = "/root/git/persona-subspace/traits/data",
        output_dir: str = "/workspace/traits",
        max_model_len: int = 4096,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        question_count: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9
    ):
        """
        Initialize the trait response generator.
        
        Args:
            model_name: HuggingFace model identifier
            traits_dir: Directory containing trait JSON files
            output_dir: Directory to save JSONL response files
            max_model_len: Maximum model context length
            tensor_parallel_size: Number of GPUs to use (auto-detect if None)
            gpu_memory_utilization: GPU memory utilization ratio
            question_count: Number of questions to process per trait (default: 20)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
        """
        self.model_name = model_name
        self.traits_dir = Path(traits_dir)
        self.output_dir = Path(output_dir)
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.question_count = question_count
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Model wrapper (loaded lazily)
        self.model_wrapper = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TraitResponseGenerator with model: {model_name}")
        logger.info(f"Traits directory: {self.traits_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Processing {self.question_count} questions per trait")
    
    def load_model(self):
        """Load the vLLM model with multi-GPU support."""
        if self.model_wrapper is not None:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading vLLM model: {self.model_name}")
        self.model_wrapper = load_vllm_model(
            model_name=self.model_name,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization
        )
        logger.info(f"Model loaded successfully with {self.model_wrapper.tensor_parallel_size} GPUs")
    
    def close_model(self):
        """Clean up the model resources."""
        if self.model_wrapper is not None:
            logger.info("Closing model...")
            close_vllm_model(self.model_wrapper)
            self.model_wrapper = None
    
    def load_trait_files(self) -> Dict[str, Dict]:
        """
        Load all trait JSON files from the traits directory.
        
        Returns:
            Dict mapping trait names to their data (instructions and questions)
        """
        trait_files = {}
        
        # Find all JSON files (excluding subdirectories and special files)
        json_files = []
        for file_path in self.traits_dir.iterdir():
            if (file_path.is_file() and 
                file_path.suffix == '.json' and 
                not file_path.name.startswith('processing_summary') and
                not file_path.name.endswith('.backup') and
                not file_path.parent.name == 'descriptions'):
                json_files.append(file_path)
        
        logger.info(f"Found {len(json_files)} trait files to process")
        
        for file_path in json_files:
            trait_name = file_path.stem
            try:
                with open(file_path, 'r') as f:
                    trait_data = json.load(f)
                
                # Validate required fields
                if not all(key in trait_data for key in ['instruction', 'questions']):
                    logger.warning(f"Skipping {trait_name}: missing required fields")
                    continue
                
                trait_files[trait_name] = trait_data
                logger.debug(f"Loaded trait: {trait_name}")
                
            except Exception as e:
                logger.error(f"Error loading trait file {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(trait_files)} trait files")
        return trait_files
    
    def format_gemma_prompt(self, instruction: Optional[str], question: str) -> str:
        """
        Format prompt for Gemma models (no system prompt support).
        
        Args:
            instruction: Optional instruction text (None for default/baseline)
            question: Question text
            
        Returns:
            Formatted prompt string
        """
        if instruction is None:
            # Default case: just the question
            return question
        else:
            # Instruction + question
            return f"{instruction}\n\n{question}"
    
    def generate_trait_responses(self, trait_name: str, trait_data: Dict) -> List[Dict]:
        """
        Generate responses for a single trait across all instruction types.
        
        Args:
            trait_name: Name of the trait
            trait_data: Trait data containing instructions and questions
            
        Returns:
            List of response dictionaries for JSONL output
        """
        instructions = trait_data['instruction']
        questions = trait_data['questions'][:self.question_count]
        
        logger.info(f"Processing trait '{trait_name}' with {len(questions)} questions")
        
        # Extract positive and negative instructions
        pos_instruction = None
        neg_instruction = None
        
        for inst in instructions:
            if 'pos' in inst:
                pos_instruction = inst['pos']
            if 'neg' in inst:
                neg_instruction = inst['neg']
        
        if pos_instruction is None or neg_instruction is None:
            logger.error(f"Missing pos/neg instructions for trait {trait_name}")
            return []
        
        # Prepare conversation batches for each instruction type
        all_conversations = []
        all_metadata = []
        
        # Positive instruction conversations
        for i, question in enumerate(questions):
            formatted_prompt = self.format_gemma_prompt(pos_instruction, question)
            conversation = [{"role": "user", "content": formatted_prompt}]
            all_conversations.append(conversation)
            all_metadata.append({
                "system_prompt": pos_instruction,
                "label": "pos",
                "question_index": i,
                "question": question
            })
        
        # Negative instruction conversations  
        for i, question in enumerate(questions):
            formatted_prompt = self.format_gemma_prompt(neg_instruction, question)
            conversation = [{"role": "user", "content": formatted_prompt}]
            all_conversations.append(conversation)
            all_metadata.append({
                "system_prompt": neg_instruction,
                "label": "neg",
                "question_index": i,
                "question": question
            })
        
        # Default (no instruction) conversations
        for i, question in enumerate(questions):
            formatted_prompt = self.format_gemma_prompt(None, question)
            conversation = [{"role": "user", "content": formatted_prompt}]
            all_conversations.append(conversation)
            all_metadata.append({
                "system_prompt": None,
                "label": "default",
                "question_index": i,
                "question": question
            })
        
        logger.info(f"Generated {len(all_conversations)} conversation prompts for trait '{trait_name}'")
        
        # Run batch inference
        try:
            logger.info(f"Running batch inference for trait '{trait_name}'...")
            responses = batch_conversation_chat(
                model_wrapper=self.model_wrapper,
                conversations=all_conversations,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                progress=True
            )
            
            logger.info(f"Generated {len(responses)} responses for trait '{trait_name}'")
            
        except Exception as e:
            logger.error(f"Error during batch inference for trait '{trait_name}': {e}")
            return []
        
        # Combine responses with metadata
        result_objects = []
        for metadata, response in zip(all_metadata, responses):
            result_obj = {
                "system_prompt": metadata["system_prompt"],
                "label": metadata["label"],
                "conversation": [
                    {"role": "user", "content": all_conversations[len(result_objects)][0]["content"]},
                    {"role": "assistant", "content": response}
                ],
                "question_index": metadata["question_index"],
                "question": metadata["question"]
            }
            result_objects.append(result_obj)
        
        return result_objects
    
    def save_trait_responses(self, trait_name: str, responses: List[Dict]):
        """
        Save trait responses to a JSONL file.
        
        Args:
            trait_name: Name of the trait
            responses: List of response dictionaries
        """
        output_file = self.output_dir / f"{trait_name}.jsonl"
        
        try:
            with jsonlines.open(output_file, mode='w') as writer:
                for response in responses:
                    writer.write(response)
            
            logger.info(f"Saved {len(responses)} responses to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving responses for trait '{trait_name}': {e}")
    
    def should_skip_trait(self, trait_name: str) -> bool:
        """
        Check if trait should be skipped (already processed).
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            True if trait should be skipped
        """
        output_file = self.output_dir / f"{trait_name}.jsonl"
        return output_file.exists()
    
    def process_all_traits(self, skip_existing: bool = True):
        """
        Process all traits and generate responses.
        
        Args:
            skip_existing: Skip traits with existing output files
        """
        # Load model
        self.load_model()
        
        try:
            # Load trait files
            trait_files = self.load_trait_files()
            
            if not trait_files:
                logger.error("No trait files found to process")
                return
            
            # Filter traits if skipping existing
            if skip_existing:
                traits_to_process = {}
                for trait_name, trait_data in trait_files.items():
                    if self.should_skip_trait(trait_name):
                        logger.info(f"Skipping trait '{trait_name}' (already exists)")
                        continue
                    traits_to_process[trait_name] = trait_data
            else:
                traits_to_process = trait_files
            
            logger.info(f"Processing {len(traits_to_process)} traits")
            
            # Process each trait
            for i, (trait_name, trait_data) in enumerate(traits_to_process.items(), 1):
                logger.info(f"Processing trait {i}/{len(traits_to_process)}: {trait_name}")
                
                try:
                    # Generate responses
                    responses = self.generate_trait_responses(trait_name, trait_data)
                    
                    if responses:
                        # Save responses
                        self.save_trait_responses(trait_name, responses)
                        logger.info(f"Successfully processed trait '{trait_name}' ({len(responses)} responses)")
                    else:
                        logger.warning(f"No responses generated for trait '{trait_name}'")
                
                except Exception as e:
                    logger.error(f"Error processing trait '{trait_name}': {e}")
                    continue
            
            logger.info(f"Completed processing {len(traits_to_process)} traits")
            
        finally:
            # Clean up model
            self.close_model()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate trait responses using vLLM batch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python traits/2_generate_responses.py

    # Custom model and directories
    python traits/2_generate_responses.py \\
        --model-name google/gemma-2-27b-it \\
        --traits-dir /path/to/traits \\
        --output-dir /path/to/output

    # Custom generation parameters
    python traits/2_generate_responses.py \\
        --temperature 0.8 \\
        --max-tokens 1024 \\
        --question-count 10
        """
    )
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='google/gemma-2-27b-it',
                       help='HuggingFace model name (default: google/gemma-2-27b-it)')
    parser.add_argument('--traits-dir', type=str, default='/root/git/persona-subspace/traits/data',
                       help='Directory containing trait JSON files')
    parser.add_argument('--output-dir', type=str, default='/workspace/traits',
                       help='Output directory for JSONL files (default: /workspace/traits)')
    parser.add_argument('--max-model-len', type=int, default=4096,
                       help='Maximum model context length (default: 4096)')
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                       help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization ratio (default: 0.9)')
    
    # Generation parameters
    parser.add_argument('--question-count', type=int, default=20,
                       help='Number of questions to process per trait (default: 20)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter (default: 0.9)')
    
    # Optional flags
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process all traits, even if output files exist')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Traits directory: {args.traits_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Question count: {args.question_count}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Skip existing: {not args.no_skip_existing}")
    
    try:
        # Create generator
        generator = TraitResponseGenerator(
            model_name=args.model_name,
            traits_dir=args.traits_dir,
            output_dir=args.output_dir,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p
        )
        
        # Process all traits
        generator.process_all_traits(skip_existing=not args.no_skip_existing)
        
        logger.info("Trait response generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    finally:
        # Ensure cleanup
        logger.info("Performing final cleanup...")
        cleanup_all_models()
        
    return 0


if __name__ == "__main__":
    exit(main())