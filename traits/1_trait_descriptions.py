#!/usr/bin/env python3
"""
Multi-worker batch processing script for generating trait data using Anthropic's batch API.

This script loads traits from a JSON file, formats them using the generate_trait prompt template,
and processes them in parallel using multiple workers with Anthropic's batch API.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from traits.prompts import PROMPTS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

class TraitBatchProcessor:
    """Multi-worker batch processor for trait data generation."""
    
    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-20241022",
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
        
    def load_traits(self, traits_file_path: str) -> Dict[str, str]:
        """Load traits from JSON file."""
        with open(traits_file_path, 'r') as f:
            traits = json.load(f)
        LOGGER.info(f"Loaded {len(traits)} traits from {traits_file_path}")
        return traits
    
    def format_trait_prompt(self, trait_name: str, trait_description: str) -> str:
        """Format a prompt for a specific trait using the generate_trait template."""
        prompt_template = PROMPTS["generate_trait"]
        
        # Format the prompt with trait information
        formatted_prompt = prompt_template.format(
            TRAIT=trait_name,
            trait_instruction=trait_description,
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
    
    def split_traits_for_workers(self, traits: Dict[str, str]) -> List[Dict[str, str]]:
        """Split traits dictionary into chunks for multiple workers."""
        trait_items = list(traits.items())
        chunk_size = len(trait_items) // self.num_workers + (1 if len(trait_items) % self.num_workers > 0 else 0)
        
        chunks = []
        for i in range(0, len(trait_items), chunk_size):
            chunk = dict(trait_items[i:i + chunk_size])
            chunks.append(chunk)
        
        LOGGER.info(f"Split {len(traits)} traits into {len(chunks)} chunks for {self.num_workers} workers")
        return chunks
    
    async def process_trait_chunk(
        self, 
        worker_id: int, 
        trait_chunk: Dict[str, str], 
        output_dir: Path
    ) -> Dict[str, str]:
        """Process a chunk of traits using batch API."""
        LOGGER.info(f"Worker {worker_id}: Processing {len(trait_chunk)} traits")
        
        # Create batch requests for this chunk
        requests = []
        trait_names = []
        for i, (trait_name, trait_description) in enumerate(trait_chunk.items()):
            formatted_prompt = self.format_trait_prompt(trait_name, trait_description)
            
            request = Request(
                custom_id=f"worker_{worker_id}_trait_{i}_{trait_name}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ]
                )
            )
            requests.append(request)
            trait_names.append(trait_name)
        
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
            custom_id_to_trait = {f"worker_{worker_id}_trait_{i}_{trait_name}": trait_name 
                                for i, trait_name in enumerate(trait_names)}
            
            for result in batch_results:
                trait_name = custom_id_to_trait.get(result.custom_id)
                if not trait_name:
                    LOGGER.error(f"Worker {worker_id}: Unknown custom_id {result.custom_id}")
                    continue
                
                if result.result.type == "succeeded":
                    response_text = result.result.message.content[0].text if result.result.message.content else ""
                    try:
                        # First try to parse as direct JSON
                        response_data = json.loads(response_text)
                        results[trait_name] = response_data
                    except json.JSONDecodeError:
                        # Try to extract JSON from markdown code blocks
                        extracted_json = self.extract_json_from_response(response_text)
                        if extracted_json:
                            results[trait_name] = extracted_json
                        else:
                            # If no JSON found, store as text
                            LOGGER.warning(f"Worker {worker_id}: Response for {trait_name} is not valid JSON, storing as text")
                            results[trait_name] = {"raw_response": response_text}
                else:
                    LOGGER.error(f"Worker {worker_id}: Failed response for trait {trait_name}: {result.result}")
                    results[trait_name] = {"error": f"Batch request failed: {result.result}"}
            
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
            
            # Return error results for all traits in this chunk
            return {trait_name: {"error": error_details} for trait_name in trait_names}
    
    async def save_individual_results(self, all_results: Dict[str, str], output_dir: Path):
        """Save individual trait results to separate JSON files."""
        LOGGER.info(f"Saving {len(all_results)} individual trait files to {output_dir}")
        
        for trait_name, result in all_results.items():
            output_file = output_dir / f"{trait_name}.json"
            try:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                LOGGER.debug(f"Saved result for trait: {trait_name}")
            except Exception as e:
                LOGGER.error(f"Error saving result for trait {trait_name}: {str(e)}")
    
    async def process_all_traits(
        self, 
        traits_file_path: str, 
        output_dir: Path,
        test_mode: bool = False,
        test_limit: int = 10
    ):
        """Main processing function that coordinates all workers."""
        # Load traits
        traits = self.load_traits(traits_file_path)
        
        # Test mode: limit number of traits
        if test_mode:
            trait_items = list(traits.items())[:test_limit]
            traits = dict(trait_items)
            LOGGER.info(f"Test mode: Processing only {len(traits)} traits")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split traits for workers
        trait_chunks = self.split_traits_for_workers(traits)
        
        # Process chunks in parallel
        LOGGER.info(f"Starting {len(trait_chunks)} workers...")
        start_time = time.time()
        
        worker_tasks = []
        for i, chunk in enumerate(trait_chunks):
            task = self.process_trait_chunk(i, chunk, output_dir)
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
        LOGGER.info(f"Successfully processed {len(all_results)} traits")
        
        # Save individual results
        await self.save_individual_results(all_results, output_dir)
        
        # Save summary
        summary = {
            "total_traits": len(traits),
            "processed_traits": len(all_results),
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
    parser = argparse.ArgumentParser(description="Process traits using Anthropic batch API with multiple workers")
    parser.add_argument(
        "--traits-file", 
        type=str, 
        default="/root/git/persona-subspace/traits/data/descriptions/traits_200.json",
        help="Path to traits JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/git/persona-subspace/traits/data",
        help="Output directory for trait response files"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Claude model ID to use"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
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
        help="Run in test mode with limited number of traits"
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=10,
        help="Number of traits to process in test mode"
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        help="Anthropic API key (optional, will use environment if not provided)"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = TraitBatchProcessor(
        model_id=args.model_id,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        anthropic_api_key=args.anthropic_api_key
    )
    
    # Run processing
    output_dir = Path(args.output_dir)
    
    asyncio.run(processor.process_all_traits(
        traits_file_path=args.traits_file,
        output_dir=output_dir,
        test_mode=args.test_mode,
        test_limit=args.test_limit
    ))


if __name__ == "__main__":
    main()