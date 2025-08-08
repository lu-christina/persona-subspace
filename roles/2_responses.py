#!/usr/bin/env python3
"""
Batch inference script for generating role responses using vLLM.

This script processes role files and generates model responses for positive
and default instructions using a specified vLLM model across multiple GPUs.

For each role, it generates responses to the questions with 10 different
instruction types:
- pos (5 variants): Using positive role instructions
- default (5 variants): Using default system prompts

Questions can be sourced from either:
1. Individual role files (default behavior) - each role uses its own questions
2. Central questions file (--questions-file) - all roles use the same set of questions

Results are saved as JSONL files (one per role) in the specified output directory.

Usage:
    # Using role-specific questions (default)
    uv run roles/2_generate_responses.py \
        --model-name google/gemma-2-27b-it \
        --roles-dir /root/git/persona-subspace/roles/data/instructions \
        --output-dir /workspace/roles/responses
    
    # Using central questions file
    uv run roles/2_generate_responses.py \
        --model-name google/gemma-2-27b-it \
        --questions-file /root/git/persona-subspace/roles/data/questions_240.jsonl \
        --output-dir /workspace/roles/responses
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


class RoleResponseGenerator:
    """Generator for role-based model responses using vLLM batch inference."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-27b-it",
        roles_dir: str = "/root/git/persona-subspace/roles/data/instructions",
        output_dir: str = "/workspace/roles/responses",
        max_model_len: int = 1024,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        question_count: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        prompt_indices: Optional[List[int]] = None,
        include_default: bool = True,
        append_mode: bool = False,
        questions_file: Optional[str] = None,
        roles_subset: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the role response generator.
        
        Args:
            model_name: HuggingFace model identifier
            roles_dir: Directory containing role JSON files
            output_dir: Directory to save JSONL response files
            max_model_len: Maximum model context length
            tensor_parallel_size: Number of GPUs to use (auto-detect if None)
            gpu_memory_utilization: GPU memory utilization ratio
            question_count: Number of questions to process per role (default: 20)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            prompt_indices: List of instruction prompt indices to process (None = all prompts)
            include_default: Whether to include default system prompts
            append_mode: Whether to append to existing files instead of overwriting
            questions_file: Path to central questions JSONL file (if None, use role-specific questions)
            roles_subset: Tuple of (start, end) indices to process subset of roles (if None, process all)
        """
        self.model_name = model_name
        self.roles_dir = Path(roles_dir)
        self.output_dir = Path(output_dir)
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.question_count = question_count
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.prompt_indices = prompt_indices if prompt_indices is not None else list(range(5))
        self.include_default = include_default
        self.append_mode = append_mode
        self.questions_file = questions_file
        self.roles_subset = roles_subset
        
        # Model wrapper (loaded lazily)
        self.model_wrapper = None
        
        # Central questions (loaded lazily)
        self.central_questions = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized RoleResponseGenerator with model: {model_name}")
        logger.info(f"Roles directory: {self.roles_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Processing {self.question_count} questions per role")
        if self.questions_file:
            logger.info(f"Using central questions file: {self.questions_file}")
        else:
            logger.info("Using role-specific questions from individual files")
    
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
    
    def load_role_files(self) -> Dict[str, Dict]:
        """
        Load all role JSON files from the roles directory.
        
        Returns:
            Dict mapping role names to their data (instructions and questions)
        """
        role_files = {}
        
        # Find all JSON files (excluding subdirectories and special files)
        json_files = []
        for file_path in self.roles_dir.iterdir():
            if (file_path.is_file() and 
                file_path.suffix == '.json' and 
                not file_path.name.startswith('processing_summary') and
                not file_path.name.endswith('.backup') and
                not file_path.parent.name == 'descriptions'):
                json_files.append(file_path)
        
        logger.info(f"Found {len(json_files)} role files to process")
        
        # Apply roles subset filtering if specified
        if self.roles_subset:
            start_idx, end_idx = self.roles_subset
            # Sort files to ensure consistent ordering
            json_files = sorted(json_files)
            original_count = len(json_files)
            json_files = json_files[start_idx:end_idx]
            logger.info(f"Applying roles subset [{start_idx}:{end_idx}] - processing {len(json_files)}/{original_count} role files")
        
        for file_path in json_files:
            role_name = file_path.stem
            try:
                with open(file_path, 'r') as f:
                    role_data = json.load(f)
                
                # Validate required fields
                required_fields = ['instruction']
                if not self.questions_file:
                    required_fields.append('questions')
                
                if not all(key in role_data for key in required_fields):
                    logger.warning(f"Skipping {role_name}: missing required fields {required_fields}")
                    continue
                
                role_files[role_name] = role_data
                logger.debug(f"Loaded role: {role_name}")
                
            except Exception as e:
                logger.error(f"Error loading role file {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(role_files)} role files")
        return role_files
    
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
    
    def generate_role_responses(self, role_name: str, role_data: Dict) -> List[Dict]:
        """
        Generate responses for a single role across all instruction types.
        
        Args:
            role_name: Name of the role
            role_data: Role data containing instructions and questions
            
        Returns:
            List of response dictionaries for JSONL output
        """
        instructions = role_data['instruction']
        questions = role_data['questions'][:self.question_count]
        
        logger.info(f"Processing role '{role_name}' with {len(questions)} questions")
        
        # Extract all positive instructions
        all_pos_instructions = []
        
        for inst in instructions:
            if 'pos' in inst:
                all_pos_instructions.append(inst['pos'])
        
        if not all_pos_instructions:
            logger.error(f"Missing pos instructions for role {role_name}")
            return []
        
        logger.info(f"Found {len(all_pos_instructions)} positive instructions")
        
        # Keep all instructions, filter during processing
        pos_instructions = all_pos_instructions
        
        logger.info(f"Available {len(pos_instructions)} pos instruction prompts, will use indices: {self.prompt_indices}")
        
        # Define default system prompts
        default_instructions = [
            "",
            "You are an AI assistant.",
            "You are a large language model.",
            "You are Gemma.",
            "Respond as yourself."
        ] if self.include_default else []
        
        # Load existing responses if in append mode
        existing_responses = []
        existing_analysis = {"labels_done": set()}
        if self.append_mode:
            existing_responses = self.load_existing_responses(role_name)
            
            # Truncate responses if needed (discard any invalid responses from buggy runs)
            # With 10 instruction types (5 pos + 5 default) and default question_count, 
            # we expect a reasonable number of responses
            max_expected_responses = len(self.prompt_indices) * 2 * self.question_count  # pos + default
            if len(existing_responses) > max_expected_responses:
                logger.info(f"Truncating existing responses from {len(existing_responses)} to {max_expected_responses} (discarding invalid responses)")
                existing_responses = existing_responses[:max_expected_responses]
                
                # Fix prompt_index for existing responses (they should all be 0 from original correct run)
                for response in existing_responses:
                    response['prompt_index'] = 0
                logger.info("Reset prompt_index to 0 for all existing responses (they were from the original run)")
            
            existing_analysis = self.analyze_existing_responses(existing_responses)
            logger.info(f"Found {existing_analysis['total']} existing responses, prompts done: {existing_analysis['prompts_done']}")
        
        # Prepare conversation batches for each instruction type
        all_conversations = []
        all_metadata = []
        
        # Positive instruction conversations (selected variants)
        for prompt_idx in self.prompt_indices:
            if prompt_idx < len(pos_instructions):
                pos_instruction = pos_instructions[prompt_idx]
                for q_idx, question in enumerate(questions):
                    # Skip if this combination already exists (check by question and prompt combination)
                    skip_key = ("pos", prompt_idx, q_idx)
                    if self.append_mode and skip_key in {(r['label'], r.get('prompt_index', 0), r['question_index']) for r in existing_responses}:
                        continue
                    
                    formatted_prompt = self.format_gemma_prompt(pos_instruction, question)
                    conversation = [{"role": "user", "content": formatted_prompt}]
                    all_conversations.append(conversation)
                    all_metadata.append({
                        "system_prompt": pos_instruction,
                        "label": "pos",
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
                    
                    formatted_prompt = self.format_gemma_prompt(default_instruction, question)
                    conversation = [{"role": "user", "content": formatted_prompt}]
                    all_conversations.append(conversation)
                    all_metadata.append({
                        "system_prompt": default_instruction,
                        "label": "default", 
                        "prompt_index": prompt_idx,
                        "question_index": q_idx,
                        "question": question
                    })
        
        logger.info(f"Generated {len(all_conversations)} conversation prompts for role '{role_name}'")
        
        # Run batch inference
        try:
            logger.info(f"Running batch inference for role '{role_name}'...")
            responses = batch_conversation_chat(
                model_wrapper=self.model_wrapper,
                conversations=all_conversations,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                progress=True
            )
            
            logger.info(f"Generated {len(responses)} responses for role '{role_name}'")
            
        except Exception as e:
            logger.error(f"Error during batch inference for role '{role_name}': {e}")
            return []
        
        # Combine responses with metadata
        new_result_objects = []
        for metadata, response in zip(all_metadata, responses):
            result_obj = {
                "system_prompt": metadata["system_prompt"],
                "label": metadata["label"],
                "prompt_index": metadata["prompt_index"],
                "conversation": [
                    {"role": "user", "content": all_conversations[len(new_result_objects)][0]["content"]},
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
    
    def generate_role_responses_with_central_questions(self, role_name: str, role_data: Dict) -> List[Dict]:
        """
        Generate responses for a single role using central questions file.
        
        Args:
            role_name: Name of the role
            role_data: Role data containing instructions (questions come from central file)
            
        Returns:
            List of response dictionaries for JSONL output
        """
        instructions = role_data['instruction']
        
        # Load questions from central file
        central_questions = self.load_central_questions()
        questions = central_questions[:self.question_count]
        
        logger.info(f"Processing role '{role_name}' with {len(questions)} questions from central file")
        
        # Extract all positive instructions
        all_pos_instructions = []
        
        for inst in instructions:
            if 'pos' in inst:
                all_pos_instructions.append(inst['pos'])
        
        if not all_pos_instructions:
            logger.error(f"Missing pos instructions for role {role_name}")
            return []
        
        logger.info(f"Found {len(all_pos_instructions)} positive instructions")
        
        # Keep all instructions, filter during processing
        pos_instructions = all_pos_instructions
        
        logger.info(f"Available {len(pos_instructions)} pos instruction prompts, will use indices: {self.prompt_indices}")
        
        # Define default system prompts
        default_instructions = [
            "",
            "You are an AI assistant.",
            "You are a large language model.",
            "You are Gemma.",
            "Respond as yourself."
        ] if self.include_default else []
        
        # Load existing responses if in append mode
        existing_responses = []
        existing_analysis = {"labels_done": set()}
        if self.append_mode:
            existing_responses = self.load_existing_responses(role_name)
            
            # Truncate responses if needed (discard any invalid responses from buggy runs)
            # With 10 instruction types (5 pos + 5 default) and default question_count, 
            # we expect a reasonable number of responses
            max_expected_responses = len(self.prompt_indices) * 2 * self.question_count  # pos + default
            if len(existing_responses) > max_expected_responses:
                logger.info(f"Truncating existing responses from {len(existing_responses)} to {max_expected_responses} (discarding invalid responses)")
                existing_responses = existing_responses[:max_expected_responses]
                
                # Fix prompt_index for existing responses (they should all be 0 from original correct run)
                for response in existing_responses:
                    response['prompt_index'] = 0
                logger.info("Reset prompt_index to 0 for all existing responses (they were from the original run)")
            
            existing_analysis = self.analyze_existing_responses(existing_responses)
            logger.info(f"Found {existing_analysis['total']} existing responses, prompts done: {existing_analysis['prompts_done']}")
        
        # Prepare conversation batches for each instruction type
        all_conversations = []
        all_metadata = []
        
        # Positive instruction conversations (selected variants)
        for prompt_idx in self.prompt_indices:
            if prompt_idx < len(pos_instructions):
                pos_instruction = pos_instructions[prompt_idx]
                for q_idx, question in enumerate(questions):
                    # Skip if this combination already exists (check by question and prompt combination)
                    skip_key = ("pos", prompt_idx, q_idx)
                    if self.append_mode and skip_key in {(r['label'], r.get('prompt_index', 0), r['question_index']) for r in existing_responses}:
                        continue
                    
                    formatted_prompt = self.format_gemma_prompt(pos_instruction, question)
                    conversation = [{"role": "user", "content": formatted_prompt}]
                    all_conversations.append(conversation)
                    all_metadata.append({
                        "system_prompt": pos_instruction,
                        "label": "pos",
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
                    
                    formatted_prompt = self.format_gemma_prompt(default_instruction, question)
                    conversation = [{"role": "user", "content": formatted_prompt}]
                    all_conversations.append(conversation)
                    all_metadata.append({
                        "system_prompt": default_instruction,
                        "label": "default", 
                        "prompt_index": prompt_idx,
                        "question_index": q_idx,
                        "question": question
                    })
        
        logger.info(f"Generated {len(all_conversations)} conversation prompts for role '{role_name}'")
        
        # Run batch inference
        try:
            logger.info(f"Running batch inference for role '{role_name}'...")
            responses = batch_conversation_chat(
                model_wrapper=self.model_wrapper,
                conversations=all_conversations,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                progress=True
            )
            
            logger.info(f"Generated {len(responses)} responses for role '{role_name}'")
            
        except Exception as e:
            logger.error(f"Error during batch inference for role '{role_name}': {e}")
            return []
        
        # Combine responses with metadata
        new_result_objects = []
        for metadata, response in zip(all_metadata, responses):
            result_obj = {
                "system_prompt": metadata["system_prompt"],
                "label": metadata["label"],
                "prompt_index": metadata["prompt_index"],
                "conversation": [
                    {"role": "user", "content": all_conversations[len(new_result_objects)][0]["content"]},
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
    
    def save_role_responses(self, role_name: str, responses: List[Dict]):
        """
        Save role responses to a JSONL file.
        
        Args:
            role_name: Name of the role
            responses: List of response dictionaries
        """
        output_file = self.output_dir / f"{role_name}.jsonl"
        
        try:
            with jsonlines.open(output_file, mode='w') as writer:
                for response in responses:
                    writer.write(response)
            
            logger.info(f"Saved {len(responses)} responses to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving responses for role '{role_name}': {e}")
    
    def load_existing_responses(self, role_name: str) -> List[Dict]:
        """
        Load existing responses for a role.
        
        Args:
            role_name: Name of the role
            
        Returns:
            List of existing response dictionaries
        """
        output_file = self.output_dir / f"{role_name}.jsonl"
        
        if not output_file.exists():
            return []
        
        existing_responses = []
        try:
            with jsonlines.open(output_file, 'r') as reader:
                for response in reader:
                    existing_responses.append(response)
            
            logger.debug(f"Loaded {len(existing_responses)} existing responses for role '{role_name}'")
            return existing_responses
            
        except Exception as e:
            logger.error(f"Error loading existing responses for role '{role_name}': {e}")
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
    
    def should_skip_role(self, role_name: str) -> bool:
        """
        Check if role should be skipped (already processed).
        
        Args:
            role_name: Name of the role
            
        Returns:
            True if role should be skipped
        """
        if self.append_mode:
            return False  # Never skip in append mode
        
        output_file = self.output_dir / f"{role_name}.jsonl"
        return output_file.exists()
    
    def process_all_roles(self, skip_existing: bool = True):
        """
        Process all roles and generate responses.
        
        Args:
            skip_existing: Skip roles with existing output files
        """
        # Load model
        self.load_model()
        
        try:
            # Load role files
            role_files = self.load_role_files()
            
            if not role_files:
                logger.error("No role files found to process")
                return
            
            # Filter roles if skipping existing
            if skip_existing:
                roles_to_process = {}
                for role_name, role_data in role_files.items():
                    if self.should_skip_role(role_name):
                        logger.info(f"Skipping role '{role_name}' (already exists)")
                        continue
                    roles_to_process[role_name] = role_data
            else:
                roles_to_process = role_files
            
            logger.info(f"Processing {len(roles_to_process)} roles")
            
            # Process each role
            for i, (role_name, role_data) in enumerate(roles_to_process.items(), 1):
                logger.info(f"Processing role {i}/{len(roles_to_process)}: {role_name}")
                
                try:
                    # Generate responses using appropriate method
                    if self.questions_file:
                        responses = self.generate_role_responses_with_central_questions(role_name, role_data)
                    else:
                        responses = self.generate_role_responses(role_name, role_data)
                    
                    if responses:
                        # Save responses
                        self.save_role_responses(role_name, responses)
                        logger.info(f"Successfully processed role '{role_name}' ({len(responses)} responses)")
                    else:
                        logger.warning(f"No responses generated for role '{role_name}'")
                
                except Exception as e:
                    logger.error(f"Error processing role '{role_name}': {e}")
                    continue
            
            logger.info(f"Completed processing {len(roles_to_process)} roles")
            
        finally:
            # Clean up model
            self.close_model()
    
    def process_single_role(self, role_name: str):
        """
        Process a single role for testing purposes.
        
        Args:
            role_name: Name of the role to process
        """
        # Load model
        self.load_model()
        
        try:
            # Load role files
            role_files = self.load_role_files()
            
            if role_name not in role_files:
                logger.error(f"Role '{role_name}' not found in role files")
                available_roles = list(role_files.keys())
                logger.info(f"Available roles: {available_roles[:10]}...")  # Show first 10
                return
            
            role_data = role_files[role_name]
            logger.info(f"Processing single role: {role_name}")
            
            try:
                # Generate responses using appropriate method
                if self.questions_file:
                    responses = self.generate_role_responses_with_central_questions(role_name, role_data)
                else:
                    responses = self.generate_role_responses(role_name, role_data)
                
                if responses:
                    # Save responses
                    self.save_role_responses(role_name, responses)
                    logger.info(f"Successfully processed role '{role_name}' ({len(responses)} responses)")
                else:
                    logger.warning(f"No responses generated for role '{role_name}'")
            
            except Exception as e:
                logger.error(f"Error processing role '{role_name}': {e}")
        
        finally:
            # Clean up model
            self.close_model()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate role responses using vLLM batch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python roles/2_generate_responses.py

    # Custom model and directories
    python roles/2_generate_responses.py \\
        --model-name google/gemma-2-27b-it \\
        --roles-dir /path/to/roles \\
        --output-dir /path/to/output

    # Custom generation parameters
    python roles/2_generate_responses.py \\
        --temperature 0.8 \\
        --max-tokens 1024 \\
        --question-count 10
        """
    )
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='google/gemma-2-27b-it',
                       help='HuggingFace model name (default: google/gemma-2-27b-it)')
    parser.add_argument('--roles-dir', type=str, default='/root/git/persona-subspace/roles/data/instructions',
                       help='Directory containing role JSON files')
    parser.add_argument('--output-dir', type=str, default='/workspace/roles/responses',
                       help='Output directory for JSONL files (default: /workspace/roles/responses)')
    parser.add_argument('--max-model-len', type=int, default=1024,
                       help='Maximum model context length (default: 1024)')
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                       help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization ratio (default: 0.9)')
    
    # Generation parameters
    parser.add_argument('--question-count', type=int, default=30,
                       help='Number of questions to process per role (default: 20)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter (default: 0.9)')
    
    # Instruction selection parameters
    parser.add_argument('--prompt-indices', type=str, default=None,
                       help='Comma-separated list of instruction prompt indices to process (e.g., "1,2,3,4")')
    parser.add_argument('--include-default', action='store_true', default=True,
                       help='Include default system prompts in addition to positive prompts (default: True)')
    parser.add_argument('--no-default', action='store_false', dest='include_default',
                       help='Disable default system prompts')
    parser.add_argument('--append-mode', action='store_true',
                       help='Append responses to existing files instead of overwriting')
    
    # Questions source configuration
    parser.add_argument('--questions-file', type=str, default=None,
                       help='Path to central questions JSONL file (if not specified, uses role-specific questions)')
    
    # Role selection parameters
    parser.add_argument('--roles-subset', type=str, default=None,
                       help='Process subset of roles by index range (e.g., "0-120" or "121-240")')
    
    # Optional flags
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process all roles, even if output files exist')
    parser.add_argument('--single-role', type=str, default=None,
                       help='Process only this specific role (for testing)')
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
    
    # Parse roles subset
    roles_subset = None
    if args.roles_subset:
        try:
            if '-' in args.roles_subset:
                start, end = args.roles_subset.split('-')
                roles_subset = (int(start.strip()), int(end.strip()))
                logger.info(f"Using roles subset: {roles_subset[0]} to {roles_subset[1]}")
            else:
                logger.error(f"Invalid roles subset format: {args.roles_subset}. Use format like '0-120'")
                return 1
        except ValueError as e:
            logger.error(f"Invalid roles subset format: {args.roles_subset}")
            return 1
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Roles directory: {args.roles_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Question count: {args.question_count}")
    logger.info(f"  Questions file: {args.questions_file if args.questions_file else 'role-specific'}")
    logger.info(f"  Roles subset: {roles_subset if roles_subset else 'all roles'}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Prompt indices: {prompt_indices if prompt_indices else 'all (0-4)'}")
    logger.info(f"  Include default: {args.include_default}")
    logger.info(f"  Append mode: {args.append_mode}")
    logger.info(f"  Skip existing: {not args.no_skip_existing}")
    
    try:
        # Create generator
        generator = RoleResponseGenerator(
            model_name=args.model_name,
            roles_dir=args.roles_dir,
            output_dir=args.output_dir,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            prompt_indices=prompt_indices,
            include_default=args.include_default,
            append_mode=args.append_mode,
            questions_file=args.questions_file,
            roles_subset=roles_subset
        )
        
        # Process roles
        if args.single_role:
            # Process only the specified role
            logger.info(f"Processing single role: {args.single_role}")
            generator.process_single_role(args.single_role)
        else:
            # Process all roles
            generator.process_all_roles(skip_existing=not args.no_skip_existing)
        
        logger.info("Role response generation completed successfully!")
        
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