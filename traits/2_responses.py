#!/usr/bin/env python3
"""
Batch inference script for generating trait responses using vLLM.

This script processes trait files and generates model responses for positive, negative,
and default instructions using a specified vLLM model across multiple GPUs.

For each trait, it generates responses to the questions with 15 different
instruction types:
- pos (5 variants): Using positive trait instructions (all 5 pairs)
- neg (5 variants): Using negative trait instructions (all 5 pairs) 
- default (5 variants): Using default system prompts (5 different styles)

Questions can be sourced from either:
1. Individual trait files (default behavior) - each trait uses its own questions
2. Central questions file (--questions-file) - all traits use the same set of questions

Results are saved as JSONL files (one per trait) in the specified output directory.

Usage:
    # Using trait-specific questions (default)
    uv run traits/2_generate_responses.py \
        --model-name google/gemma-2-27b-it \
        --traits-dir /root/git/persona-subspace/traits/data/instructions \
        --output-dir /workspace/traits/responses
    
    # Using central questions file
    uv run traits/2_generate_responses.py \
        --model-name google/gemma-2-27b-it \
        --questions-file /root/git/persona-subspace/traits/data/questions_240.jsonl \
        --output-dir /workspace/traits/responses
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

from utils.inference_utils import load_vllm_model, batch_conversation, close_vllm_model, cleanup_all_models

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TraitResponseGenerator:
    """Generator for trait-based model responses using vLLM batch inference."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-27b-it",
        traits_dir: str = "/root/git/persona-subspace/traits/data/instructions",
        output_dir: str = "/workspace/traits/responses",
        max_model_len: int = 1024,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        question_count: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        prompt_indices: Optional[List[int]] = None,
        skip_default: bool = False,
        default_only: bool = False,
        append_mode: bool = False,
        questions_file: Optional[str] = None,
        traits_subset: Optional[Tuple[int, int]] = None
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
            prompt_indices: List of instruction prompt indices to process (None = all prompts)
            skip_default: Whether to skip default system prompts (only run pos/neg)
            default_only: Whether to skip pos/neg instructions (only run default prompts)
            append_mode: Whether to append to existing files instead of overwriting
            questions_file: Path to central questions JSONL file (if None, use trait-specific questions)
            traits_subset: Tuple of (start, end) indices to process subset of traits (if None, process all)
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
        self.prompt_indices = prompt_indices if prompt_indices is not None else list(range(5))
        self.skip_default = skip_default
        self.default_only = default_only
        self.append_mode = append_mode
        self.questions_file = questions_file
        self.traits_subset = traits_subset
        
        # Model wrapper (loaded lazily)
        self.model_wrapper = None
        
        # Central questions (loaded lazily)
        self.central_questions = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TraitResponseGenerator with model: {model_name}")
        logger.info(f"Traits directory: {self.traits_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Processing {self.question_count} questions per trait")
        if self.questions_file:
            logger.info(f"Using central questions file: {self.questions_file}")
        else:
            logger.info("Using trait-specific questions from individual files")
    
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
    
    def load_central_questions(self) -> List[str]:
        """
        Load questions from central JSONL file.
        
        Returns:
            List of question strings
        """
        if self.central_questions is not None:
            return self.central_questions
        
        if not self.questions_file:
            raise ValueError("No questions file specified")
        
        questions_path = Path(self.questions_file)
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
        
        logger.info(f"Loading questions from: {questions_path}")
        
        questions = []
        try:
            with jsonlines.open(questions_path, 'r') as reader:
                for entry in reader:
                    if 'question' not in entry:
                        logger.warning(f"Skipping entry missing 'question' field: {entry}")
                        continue
                    questions.append(entry['question'])
            
            logger.info(f"Loaded {len(questions)} questions from central file")
            self.central_questions = questions
            return questions
            
        except Exception as e:
            logger.error(f"Error loading central questions file: {e}")
            raise
    
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
        
        # Apply traits subset filtering if specified
        if self.traits_subset:
            start_idx, end_idx = self.traits_subset
            # Sort files to ensure consistent ordering
            json_files = sorted(json_files)
            original_count = len(json_files)
            json_files = json_files[start_idx:end_idx]
            logger.info(f"Applying traits subset [{start_idx}:{end_idx}] - processing {len(json_files)}/{original_count} trait files")
        
        for file_path in json_files:
            trait_name = file_path.stem
            try:
                with open(file_path, 'r') as f:
                    trait_data = json.load(f)
                
                # Validate required fields
                required_fields = ['instruction']
                if not self.questions_file:
                    required_fields.append('questions')
                
                if not all(key in trait_data for key in required_fields):
                    logger.warning(f"Skipping {trait_name}: missing required fields {required_fields}")
                    continue
                
                trait_files[trait_name] = trait_data
                logger.debug(f"Loaded trait: {trait_name}")
                
            except Exception as e:
                logger.error(f"Error loading trait file {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(trait_files)} trait files")
        return trait_files
    
    def supports_system_prompt(self) -> bool:
        """
        Check if the current model supports system prompts.
        
        Most models support system prompts by default. Gemma models are the exception.
        
        Returns:
            True if model supports system prompts, False if it requires concatenation
        """
        # Gemma models don't support system prompts properly
        if self.model_name.startswith("google/gemma-2"):
            return False
        # All other models (Qwen, Llama, etc.) support system prompts
        return True
    
    def format_conversation(self, instruction: Optional[str], question: str) -> List[Dict[str, str]]:
        """
        Format conversation for different model types.
        
        Default behavior: Use instruction as system prompt, question as user message
        Gemma models (google/*): Concatenate instruction and question as single user message
        
        Args:
            instruction: Optional instruction text (None for default/baseline)
            question: Question text
            
        Returns:
            List of message dictionaries for conversation
        """
        if self.supports_system_prompt():
            # Most models: use proper system prompt
            messages = []
            if instruction:
                messages.append({"role": "system", "content": instruction})
            messages.append({"role": "user", "content": question})
            return messages
        else:
            # Gemma models: concatenate instruction and question
            if instruction is None:
                # Default case: just the question
                formatted_prompt = question
            else:
                # Instruction + question
                formatted_prompt = f"{instruction}\n\n{question}"
            return [{"role": "user", "content": formatted_prompt}]
    
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
        
        # Extract all positive and negative instruction pairs
        all_pos_instructions = []
        all_neg_instructions = []
        
        for inst in instructions:
            if 'pos' in inst and 'neg' in inst:
                all_pos_instructions.append(inst['pos'])
                all_neg_instructions.append(inst['neg'])
        
        if not all_pos_instructions or not all_neg_instructions:
            logger.error(f"Missing pos/neg instructions for trait {trait_name}")
            return []
        
        logger.info(f"Found {len(all_pos_instructions)} positive and {len(all_neg_instructions)} negative instruction pairs")
        
        # Keep all instructions, filter during processing
        pos_instructions = all_pos_instructions
        neg_instructions = all_neg_instructions
        
        logger.info(f"Available {len(pos_instructions)} pos and {len(neg_instructions)} neg instruction prompts, will use indices: {self.prompt_indices}")

        # Define default system prompts
        model_name_short = ""
        if self.model_name.startswith("google/gemma"):
            model_name_short = "Gemma"
        elif self.model_name.startswith("Qwen"):
            model_name_short = "Qwen"
        elif self.model_name.startswith("meta-llama"):
            model_name_short = "Llama"
            
        default_instructions = [
            "",
            "You are an AI assistant.",
            "You are a large language model.",
            f"You are {model_name_short}.",
            "Respond as yourself."
        ] if not self.skip_default else []
        
        # Load existing responses if in append mode
        existing_responses = []
        existing_analysis = {"labels_done": set()}
        if self.append_mode:
            existing_responses = self.load_existing_responses(trait_name)
            
            # Truncate to first 60 responses (discard any invalid responses from buggy runs)
            if len(existing_responses) > 60:
                logger.info(f"Truncating existing responses from {len(existing_responses)} to 60 (discarding invalid responses)")
                existing_responses = existing_responses[:60]
                
                # Fix prompt_index for the first 60 responses (they should all be 0 from original correct run)
                for response in existing_responses:
                    response['prompt_index'] = 0
                logger.info("Reset prompt_index to 0 for all existing responses (they were from the original run)")
            
            existing_analysis = self.analyze_existing_responses(existing_responses)
            logger.info(f"Found {existing_analysis['total']} existing responses, prompts done: {existing_analysis['prompts_done']}")
        
        # Prepare conversation batches for each instruction type
        all_conversations = []
        all_metadata = []

        # Positive instruction conversations (selected variants) - skip if default_only
        if not self.default_only:
            for prompt_idx in self.prompt_indices:
                if prompt_idx < len(pos_instructions):
                    pos_instruction = pos_instructions[prompt_idx]
                    for q_idx, question in enumerate(questions):
                        # Skip if this combination already exists (check by question and prompt combination)
                        skip_key = ("pos", prompt_idx, q_idx)
                        if self.append_mode and skip_key in {(r['label'], r.get('prompt_index', 0), r['question_index']) for r in existing_responses}:
                            continue

                        conversation = self.format_conversation(pos_instruction, question)
                        all_conversations.append(conversation)
                        all_metadata.append({
                            "system_prompt": pos_instruction,
                            "label": "pos",
                            "prompt_index": prompt_idx,
                            "question_index": q_idx,
                            "question": question
                        })

        # Negative instruction conversations (selected variants) - skip if default_only
        if not self.default_only:
            for prompt_idx in self.prompt_indices:
                if prompt_idx < len(neg_instructions):
                    neg_instruction = neg_instructions[prompt_idx]
                    for q_idx, question in enumerate(questions):
                        # Skip if this combination already exists
                        skip_key = ("neg", prompt_idx, q_idx)
                        if self.append_mode and skip_key in {(r['label'], r.get('prompt_index', 0), r['question_index']) for r in existing_responses}:
                            continue

                        conversation = self.format_conversation(neg_instruction, question)
                        all_conversations.append(conversation)
                        all_metadata.append({
                            "system_prompt": neg_instruction,
                            "label": "neg",
                            "prompt_index": prompt_idx,
                            "question_index": q_idx,
                            "question": question
                        })
        
        # Default instruction conversations (selected variants)
        for prompt_idx in self.prompt_indices:
            if prompt_idx < len(default_instructions):
                default_instruction = default_instructions[prompt_idx]
                for q_idx, question in enumerate(questions):
                    # Skip if this combination already exists
                    skip_key = ("default", prompt_idx, q_idx)
                    if self.append_mode and skip_key in {(r['label'], r.get('prompt_index', 0), r['question_index']) for r in existing_responses}:
                        continue
                    
                    conversation = self.format_conversation(default_instruction, question)
                    all_conversations.append(conversation)
                    all_metadata.append({
                        "system_prompt": default_instruction,
                        "label": "default", 
                        "prompt_index": prompt_idx,
                        "question_index": q_idx,
                        "question": question
                    })
        
        logger.info(f"Generated {len(all_conversations)} conversation prompts for trait '{trait_name}'")
        
        # Run batch inference
        try:
            logger.info(f"Running batch inference for trait '{trait_name}'...")
            responses = batch_conversation(
                model_wrapper=self.model_wrapper,
                conversations=all_conversations,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                progress=True,
                enable_thinking=False
            )
            
            logger.info(f"Generated {len(responses)} responses for trait '{trait_name}'")
            
        except Exception as e:
            logger.error(f"Error during batch inference for trait '{trait_name}': {e}")
            return []
        
        # Combine responses with metadata
        new_result_objects = []
        for metadata, response in zip(all_metadata, responses):
            result_obj = {
                "system_prompt": metadata["system_prompt"],
                "label": metadata["label"],
                "prompt_index": metadata["prompt_index"],
                "conversation": all_conversations[len(new_result_objects)] + [
                    {"role": "assistant", "content": response}
                ],
                "question_index": metadata["question_index"],
                "question": metadata["question"]
            }
            new_result_objects.append(result_obj)
        
        # In append mode, combine with existing responses
        if self.append_mode and existing_responses:
            all_result_objects = existing_responses + new_result_objects
            logger.info(f"Combined {len(existing_responses)} existing + {len(new_result_objects)} new = {len(all_result_objects)} total responses")
            return all_result_objects
        else:
            return new_result_objects
    
    def generate_trait_responses_with_central_questions(self, trait_name: str, trait_data: Dict) -> List[Dict]:
        """
        Generate responses for a single trait using central questions file.
        
        Args:
            trait_name: Name of the trait
            trait_data: Trait data containing instructions (questions come from central file)
            
        Returns:
            List of response dictionaries for JSONL output
        """
        instructions = trait_data['instruction']
        
        # Load questions from central file
        central_questions = self.load_central_questions()
        questions = central_questions[:self.question_count]
        
        logger.info(f"Processing trait '{trait_name}' with {len(questions)} questions from central file")
        
        # Extract all positive and negative instruction pairs
        all_pos_instructions = []
        all_neg_instructions = []
        
        for inst in instructions:
            if 'pos' in inst and 'neg' in inst:
                all_pos_instructions.append(inst['pos'])
                all_neg_instructions.append(inst['neg'])
        
        if not all_pos_instructions or not all_neg_instructions:
            logger.error(f"Missing pos/neg instructions for trait {trait_name}")
            return []
        
        logger.info(f"Found {len(all_pos_instructions)} positive and {len(all_neg_instructions)} negative instruction pairs")
        
        # Keep all instructions, filter during processing
        pos_instructions = all_pos_instructions
        neg_instructions = all_neg_instructions
        
        logger.info(f"Available {len(pos_instructions)} pos and {len(neg_instructions)} neg instruction prompts, will use indices: {self.prompt_indices}")
        
        # Define default system prompts
        default_instructions = [
            "",
            "You are an AI assistant.",
            "You are a large language model.",
            "You are Llama.",
            "Respond as yourself."
        ] if not self.skip_default else []
        
        # Load existing responses if in append mode
        existing_responses = []
        existing_analysis = {"labels_done": set()}
        if self.append_mode:
            existing_responses = self.load_existing_responses(trait_name)
            
            # Truncate to first 60 responses (discard any invalid responses from buggy runs)
            if len(existing_responses) > 60:
                logger.info(f"Truncating existing responses from {len(existing_responses)} to 60 (discarding invalid responses)")
                existing_responses = existing_responses[:60]
                
                # Fix prompt_index for the first 60 responses (they should all be 0 from original correct run)
                for response in existing_responses:
                    response['prompt_index'] = 0
                logger.info("Reset prompt_index to 0 for all existing responses (they were from the original run)")
            
            existing_analysis = self.analyze_existing_responses(existing_responses)
            logger.info(f"Found {existing_analysis['total']} existing responses, prompts done: {existing_analysis['prompts_done']}")
        
        # Prepare conversation batches for each instruction type
        all_conversations = []
        all_metadata = []

        # Positive instruction conversations (selected variants) - skip if default_only
        if not self.default_only:
            for prompt_idx in self.prompt_indices:
                if prompt_idx < len(pos_instructions):
                    pos_instruction = pos_instructions[prompt_idx]
                    for q_idx, question in enumerate(questions):
                        # Skip if this combination already exists (check by question and prompt combination)
                        skip_key = ("pos", prompt_idx, q_idx)
                        if self.append_mode and skip_key in {(r['label'], r.get('prompt_index', 0), r['question_index']) for r in existing_responses}:
                            continue

                        conversation = self.format_conversation(pos_instruction, question)
                        all_conversations.append(conversation)
                        all_metadata.append({
                            "system_prompt": pos_instruction,
                            "label": "pos",
                            "prompt_index": prompt_idx,
                            "question_index": q_idx,
                            "question": question
                        })

        # Negative instruction conversations (selected variants) - skip if default_only
        if not self.default_only:
            for prompt_idx in self.prompt_indices:
                if prompt_idx < len(neg_instructions):
                    neg_instruction = neg_instructions[prompt_idx]
                    for q_idx, question in enumerate(questions):
                        # Skip if this combination already exists
                        skip_key = ("neg", prompt_idx, q_idx)
                        if self.append_mode and skip_key in {(r['label'], r.get('prompt_index', 0), r['question_index']) for r in existing_responses}:
                            continue

                        conversation = self.format_conversation(neg_instruction, question)
                        all_conversations.append(conversation)
                        all_metadata.append({
                            "system_prompt": neg_instruction,
                            "label": "neg",
                            "prompt_index": prompt_idx,
                            "question_index": q_idx,
                            "question": question
                        })
        
        # Default instruction conversations (selected variants)
        for prompt_idx in self.prompt_indices:
            if prompt_idx < len(default_instructions):
                default_instruction = default_instructions[prompt_idx]
                for q_idx, question in enumerate(questions):
                    # Skip if this combination already exists
                    skip_key = ("default", prompt_idx, q_idx)
                    if self.append_mode and skip_key in {(r['label'], r.get('prompt_index', 0), r['question_index']) for r in existing_responses}:
                        continue
                    
                    conversation = self.format_conversation(default_instruction, question)
                    all_conversations.append(conversation)
                    all_metadata.append({
                        "system_prompt": default_instruction,
                        "label": "default", 
                        "prompt_index": prompt_idx,
                        "question_index": q_idx,
                        "question": question
                    })
        
        logger.info(f"Generated {len(all_conversations)} conversation prompts for trait '{trait_name}'")
        
        # Run batch inference
        try:
            logger.info(f"Running batch inference for trait '{trait_name}'...")
            responses = batch_conversation(
                model_wrapper=self.model_wrapper,
                conversations=all_conversations,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                progress=True,
                enable_thinking=False
            )
            
            logger.info(f"Generated {len(responses)} responses for trait '{trait_name}'")
            
        except Exception as e:
            logger.error(f"Error during batch inference for trait '{trait_name}': {e}")
            return []
        
        # Combine responses with metadata
        new_result_objects = []
        for metadata, response in zip(all_metadata, responses):
            result_obj = {
                "system_prompt": metadata["system_prompt"],
                "label": metadata["label"],
                "prompt_index": metadata["prompt_index"],
                "conversation": all_conversations[len(new_result_objects)] + [
                    {"role": "assistant", "content": response}
                ],
                "question_index": metadata["question_index"],
                "question": metadata["question"]
            }
            new_result_objects.append(result_obj)
        
        # In append mode, combine with existing responses
        if self.append_mode and existing_responses:
            all_result_objects = existing_responses + new_result_objects
            logger.info(f"Combined {len(existing_responses)} existing + {len(new_result_objects)} new = {len(all_result_objects)} total responses")
            return all_result_objects
        else:
            return new_result_objects
    
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
    
    def load_existing_responses(self, trait_name: str) -> List[Dict]:
        """
        Load existing responses for a trait.
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            List of existing response dictionaries
        """
        output_file = self.output_dir / f"{trait_name}.jsonl"
        
        if not output_file.exists():
            return []
        
        existing_responses = []
        try:
            with jsonlines.open(output_file, 'r') as reader:
                for response in reader:
                    existing_responses.append(response)
            
            logger.debug(f"Loaded {len(existing_responses)} existing responses for trait '{trait_name}'")
            return existing_responses
            
        except Exception as e:
            logger.error(f"Error loading existing responses for trait '{trait_name}': {e}")
            return []
    
    def analyze_existing_responses(self, existing_responses: List[Dict]) -> Dict:
        """
        Analyze existing responses to determine what prompts have been processed.
        
        Args:
            existing_responses: List of existing response dictionaries
            
        Returns:
            Dictionary with analysis results
        """
        if not existing_responses:
            return {"prompts_done": set(), "labels_done": set(), "total": 0}
        
        # Check if responses have prompt_index field (new format)
        has_prompt_index = 'prompt_index' in existing_responses[0]
        
        if has_prompt_index:
            prompts_done = set(r['prompt_index'] for r in existing_responses)
            labels_done = set((r['label'], r['prompt_index']) for r in existing_responses)
        else:
            # Old format - assume prompt index 0 only
            prompts_done = {0}
            labels_done = set((r['label'], 0) for r in existing_responses)
        
        return {
            "prompts_done": prompts_done,
            "labels_done": labels_done,
            "total": len(existing_responses),
            "has_prompt_index": has_prompt_index
        }
    
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
                    # Generate responses using appropriate method
                    if self.questions_file:
                        responses = self.generate_trait_responses_with_central_questions(trait_name, trait_data)
                    else:
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
    
    def process_single_trait(self, trait_name: str):
        """
        Process a single trait for testing purposes.
        
        Args:
            trait_name: Name of the trait to process
        """
        # Load model
        self.load_model()
        
        try:
            # Load trait files
            trait_files = self.load_trait_files()
            
            if trait_name not in trait_files:
                logger.error(f"Trait '{trait_name}' not found in trait files")
                available_traits = list(trait_files.keys())
                logger.info(f"Available traits: {available_traits[:10]}...")  # Show first 10
                return
            
            trait_data = trait_files[trait_name]
            logger.info(f"Processing single trait: {trait_name}")
            
            try:
                # Generate responses using appropriate method
                if self.questions_file:
                    responses = self.generate_trait_responses_with_central_questions(trait_name, trait_data)
                else:
                    responses = self.generate_trait_responses(trait_name, trait_data)
                
                if responses:
                    # Save responses
                    self.save_trait_responses(trait_name, responses)
                    logger.info(f"Successfully processed trait '{trait_name}' ({len(responses)} responses)")
                else:
                    logger.warning(f"No responses generated for trait '{trait_name}'")
            
            except Exception as e:
                logger.error(f"Error processing trait '{trait_name}': {e}")
        
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
    parser.add_argument('--traits-dir', type=str, default='/root/git/persona-subspace/traits/data/instructions',
                       help='Directory containing trait JSON files')
    parser.add_argument('--output-dir', type=str, default='/workspace/traits/responses',
                       help='Output directory for JSONL files (default: /workspace/traits/responses)')
    parser.add_argument('--max-model-len', type=int, default=1024,
                       help='Maximum model context length (default: 1024)')
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                       help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.95,
                       help='GPU memory utilization ratio (default: 0.9)')
    
    # Generation parameters
    parser.add_argument('--question-count', type=int, default=240,
                       help='Number of questions to process per trait (default: 20)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter (default: 0.9)')
    
    # Instruction selection parameters
    parser.add_argument('--prompt-indices', type=str, default=None,
                       help='Comma-separated list of instruction prompt indices to process (e.g., "1,2,3,4")')
    parser.add_argument('--skip-default', action='store_true',
                       help='Skip default system prompts (only run pos/neg instructions)')
    parser.add_argument('--default-only', action='store_true',
                       help='Skip pos/neg instructions (only run default system prompts)')
    parser.add_argument('--append-mode', action='store_true',
                       help='Append responses to existing files instead of overwriting')
    
    # Questions source configuration
    parser.add_argument('--questions-file', type=str, default=None,
                       help='Path to central questions JSONL file (if not specified, uses trait-specific questions)')
    
    # Trait selection parameters
    parser.add_argument('--traits-subset', type=str, default=None,
                       help='Process subset of traits by index range (e.g., "0-120" or "121-240")')
    
    # Optional flags
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process all traits, even if output files exist')
    parser.add_argument('--single-trait', type=str, default=None,
                       help='Process only this specific trait (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
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
    
    # Parse traits subset
    traits_subset = None
    if args.traits_subset:
        try:
            if '-' in args.traits_subset:
                start, end = args.traits_subset.split('-')
                traits_subset = (int(start.strip()), int(end.strip()))
                logger.info(f"Using traits subset: {traits_subset[0]} to {traits_subset[1]}")
            else:
                logger.error(f"Invalid traits subset format: {args.traits_subset}. Use format like '0-120'")
                return 1
        except ValueError as e:
            logger.error(f"Invalid traits subset format: {args.traits_subset}")
            return 1

    # Validate mutually exclusive flags
    if args.skip_default and args.default_only:
        logger.error("Error: --skip-default and --default-only are mutually exclusive")
        return 1

    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Traits directory: {args.traits_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Question count: {args.question_count}")
    logger.info(f"  Questions file: {args.questions_file if args.questions_file else 'trait-specific'}")
    logger.info(f"  Traits subset: {traits_subset if traits_subset else 'all traits'}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Prompt indices: {prompt_indices if prompt_indices else 'all (0-4)'}")
    logger.info(f"  Skip default: {args.skip_default}")
    logger.info(f"  Default only: {args.default_only}")
    logger.info(f"  Append mode: {args.append_mode}")
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
            top_p=args.top_p,
            prompt_indices=prompt_indices,
            skip_default=args.skip_default,
            default_only=args.default_only,
            append_mode=args.append_mode,
            questions_file=args.questions_file,
            traits_subset=traits_subset
        )
        
        # Process traits
        if args.single_trait:
            # Process only the specified trait
            logger.info(f"Processing single trait: {args.single_trait}")
            generator.process_single_trait(args.single_trait)
        else:
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