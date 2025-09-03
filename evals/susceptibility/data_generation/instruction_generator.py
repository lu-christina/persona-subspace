#!/usr/bin/env python3
"""
Multi-worker batch processing script for generating role data using Anthropic's batch API.

This script loads roles from a JSON file, formats them using the generate_trait prompt template,
and processes them in parallel using multiple workers with Anthropic's batch API.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from dotenv import load_dotenv
from prompts import PROMPTS

# Load environment variables from .env file
load_dotenv(Path.home() / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

class RoleBatchProcessor:
    """Multi-worker batch processor for role data generation."""
    
    def __init__(
        self,
        model_id: str = "claude-4-sonnet-20250514",
        num_workers: int = 3,
        max_tokens: int = 4000,
        anthropic_api_key: Optional[str] = None
    ):
        self.model_id = model_id
        self.num_workers = num_workers
        self.max_tokens = max_tokens
        
        # Initialize Anthropic client
        if anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            self.client = anthropic.Anthropic()
        
    def load_roles(self, roles_file_path: str) -> Dict[str, str]:
        """Load roles from JSON file."""
        with open(roles_file_path, 'r') as f:
            roles = json.load(f)
        LOGGER.info(f"Loaded {len(roles)} roles from {roles_file_path}")
        return roles
    
    def format_role_prompt(self, role_name: str, role_description: str) -> str:
        """Format a prompt for a specific role using the generate_role template."""
        prompt_template = PROMPTS["generate_role"]
        
        # Format the prompt with trait information
        formatted_prompt = prompt_template.format(
            ROLE=role_name,
            role_instruction=role_description,
            question_instruction=""  # Empty for now, can be customized if needed
        )
        
        return formatted_prompt
    
    def extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from markdown code blocks or other formats."""
        import re
        
        # Try to find JSON in markdown code blocks
        json_pattern = r'```json\s*\n(.*?)\n```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON without markdown formatting
        # Look for content between { and } that spans multiple lines
        json_pattern2 = r'\{[\s\S]*\}'
        matches2 = re.findall(json_pattern2, response_text)
        
        for match in matches2:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        return None
    
    def split_roles_for_workers(self, roles: Dict[str, str]) -> List[Dict[str, str]]:
        """Split roles dictionary into chunks for multiple workers."""
        trait_items = list(roles.items())
        chunk_size = len(trait_items) // self.num_workers + (1 if len(trait_items) % self.num_workers > 0 else 0)
        
        chunks = []
        for i in range(0, len(trait_items), chunk_size):
            chunk = dict(trait_items[i:i + chunk_size])
            chunks.append(chunk)
        
        LOGGER.info(f"Split {len(roles)} roles into {len(chunks)} chunks for {self.num_workers} workers")
        return chunks
    
    async def process_role_chunk(
        self, 
        worker_id: int, 
        role_chunk: Dict[str, str], 
        output_dir: Path
    ) -> Dict[str, str]:
        """Process a chunk of roles using batch API."""
        LOGGER.info(f"Worker {worker_id}: Processing {len(role_chunk)} roles")
        
        # Create batch requests for this chunk
        requests = []
        role_names = []
        for i, (role_name, role_description) in enumerate(role_chunk.items()):
            formatted_prompt = self.format_role_prompt(role_name, role_description)
            
            request = Request(
                custom_id=f"worker_{worker_id}_role_{i}_{role_name}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ]
                )
            )
            requests.append(request)
            role_names.append(role_name)
        
        # Create worker-specific log directory
        worker_log_dir = output_dir / f"worker_{worker_id}_logs"
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Submit batch request
            start_time = time.time()
            LOGGER.info(f"Worker {worker_id}: Submitting batch with {len(requests)} requests")
            
            batch_response = self.client.messages.batches.create(requests=requests)
            batch_id = batch_response.id
            
            # Log batch info
            with open(worker_log_dir / f"batch_{batch_id}.json", 'w') as f:
                json.dump(batch_response.model_dump(mode="json"), f, indent=2)
            
            LOGGER.info(f"Worker {worker_id}: Created batch {batch_id}, polling for completion...")
            
            # Poll for completion
            while True:
                batch_status = self.client.messages.batches.retrieve(batch_id)
                if batch_status.processing_status == "ended":
                    break
                LOGGER.info(f"Worker {worker_id}: Batch {batch_id} still processing...")
                await asyncio.sleep(30)  # Wait 30 seconds before checking again
            
            # Get results
            results_iter = self.client.messages.batches.results(batch_id)
            batch_results = list(results_iter)
            
            end_time = time.time()
            duration = end_time - start_time
            LOGGER.info(f"Worker {worker_id}: Completed batch {batch_id} in {duration:.2f}s")
            
            # Process responses
            results = {}
            custom_id_to_role = {f"worker_{worker_id}_role_{i}_{role_name}": role_name 
                                for i, role_name in enumerate(role_names)}
            
            for result in batch_results:
                role_name = custom_id_to_role.get(result.custom_id)
                if not role_name:
                    LOGGER.error(f"Worker {worker_id}: Unknown custom_id {result.custom_id}")
                    continue
                
                if result.result.type == "succeeded":
                    response_text = result.result.message.content[0].text if result.result.message.content else ""
                    try:
                        # First try to parse as direct JSON
                        response_data = json.loads(response_text)
                        results[role_name] = response_data
                    except json.JSONDecodeError:
                        # Try to extract JSON from markdown code blocks
                        extracted_json = self.extract_json_from_response(response_text)
                        if extracted_json:
                            results[role_name] = extracted_json
                        else:
                            # If no JSON found, store as text
                            LOGGER.warning(f"Worker {worker_id}: Response for {role_name} is not valid JSON, storing as text")
                            results[role_name] = {"raw_response": response_text}
                else:
                    LOGGER.error(f"Worker {worker_id}: Failed response for role {role_name}: {result.result}")
                    results[role_name] = {"error": f"Batch request failed: {result.result}"}
            
            return results
            
        except Exception as e:
            import traceback
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            LOGGER.error(f"Worker {worker_id}: Error processing batch - {error_details}")
            print(f"Full error details for Worker {worker_id}:")
            print(f"Error Type: {error_details['error_type']}")
            print(f"Error Message: {error_details['error_message']}")
            print(f"Traceback:\n{error_details['traceback']}")
            
            # Return error results for all roles in this chunk
            return {role_name: {"error": error_details} for role_name in role_names}
    
    async def save_individual_results(self, all_results: Dict[str, str], output_dir: Path):
        """Save individual role results to separate JSON files."""
        LOGGER.info(f"Saving {len(all_results)} individual role files to {output_dir}")
        
        for role_name, result in all_results.items():
            output_file = output_dir / f"{role_name}.json"
            try:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                LOGGER.debug(f"Saved result for role: {role_name}")
            except Exception as e:
                LOGGER.error(f"Error saving result for role {role_name}: {str(e)}")
    
    async def process_all_roles(
        self, 
        roles_file_path: str, 
        output_dir: Path,
        test_mode: bool = False,
        test_limit: int = 10
    ):
        """Main processing function that coordinates all workers."""
        # Load roles
        roles = self.load_roles(roles_file_path)
        
        # Test mode: limit number of roles
        if test_mode:
            trait_items = list(roles.items())[:test_limit]
            roles = dict(trait_items)
            LOGGER.info(f"Test mode: Processing only {len(roles)} roles")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split roles for workers
        role_chunks = self.split_roles_for_workers(roles)
        
        # Process chunks in parallel
        LOGGER.info(f"Starting {len(role_chunks)} workers...")
        start_time = time.time()
        
        worker_tasks = []
        for i, chunk in enumerate(role_chunks):
            task = self.process_role_chunk(i, chunk, output_dir)
            worker_tasks.append(task)
        
        # Wait for all workers to complete
        worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        # Combine results from all workers
        all_results = {}
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                LOGGER.error(f"Worker {i} failed with exception: {str(result)}")
            else:
                all_results.update(result)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        LOGGER.info(f"All workers completed in {total_duration:.2f}s")
        LOGGER.info(f"Successfully processed {len(all_results)} roles")
        
        # Save individual results
        await self.save_individual_results(all_results, output_dir)
        
        # Save summary
        summary = {
            "total_roles": len(roles),
            "processed_roles": len(all_results),
            "processing_time": total_duration,
            "model_id": self.model_id,
            "num_workers": self.num_workers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        LOGGER.info(f"Processing summary saved to {summary_file}")
        return all_results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process roles using Anthropic batch API with multiple workers")
    parser.add_argument(
        "--roles-file", 
        type=str, 
        default="/root/git/persona-subspace/evals/susceptibility/data_generation/roles_50.json",
        help="Path to roles JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/git/persona-subspace/evals/susceptibility/data_generation/instructions",
        help="Output directory for role response files"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="claude-4-sonnet-20250514",
        help="Claude model ID to use"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=5,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens per response"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with limited number of roles"
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=10,
        help="Number of roles to process in test mode"
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key (optional, will use environment if not provided)"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = RoleBatchProcessor(
        model_id=args.model_id,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        anthropic_api_key=args.anthropic_api_key
    )
    
    # Run processing
    output_dir = Path(args.output_dir)
    
    asyncio.run(processor.process_all_roles(
        roles_file_path=args.roles_file,
        output_dir=output_dir,
        test_mode=args.test_mode,
        test_limit=args.test_limit
    ))


if __name__ == "__main__":
    main()