#!/usr/bin/env python3
"""
Script to fetch Claude explanations from Neuronpedia API and add them to the CSV file.
"""

import pandas as pd
import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = "gemma_trainer131k-l0-114_layer20/1000_prompts/explanations_1percent.csv"
OUTPUT_FILE = "gemma_trainer131k-l0-114_layer20/explanations_with_claude.csv"

# INPUT_FILE = "llama_trainer32x_layer15/10000_prompts/explanations_1percent.csv"
# OUTPUT_FILE = "llama_trainer32x_layer15/explanations_with_claude.csv"


# # Neuronpedia API parameters
MODEL_ID = "gemma-2-9b"
LAYER = "20-gemmascope-res-131k"
# MODEL_ID = "llama3.1-8b"
# LAYER = "15-llamascope-res-131k"


EXPLANATION_TYPE = "eleuther_acts_top20"
EXPLANATION_MODEL_NAME = "claude-3-7-sonnet-20250219"

class ThreadSafeCSVWriter:
    """Thread-safe CSV writer that appends results incrementally."""
    
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.write_lock = threading.Lock()
        
        # Create header if file doesn't exist
        if not self.output_file.exists():
            with open(self.output_file, 'w') as f:
                f.write("feature_id,link,claude_desc\n")
        
    def append_feature(self, feature_id: int, link: str, claude_desc: Optional[str]):
        """Append a feature's data to the CSV file."""
        with self.write_lock:
            # Escape any quotes in the description and wrap in quotes if it contains commas
            if claude_desc is not None:
                claude_desc = claude_desc.replace('"', '""')
                if ',' in claude_desc or '"' in claude_desc or '\n' in claude_desc:
                    claude_desc = f'"{claude_desc}"'
            else:
                claude_desc = ""
            
            with open(self.output_file, 'a') as f:
                f.write(f"{feature_id},{link},{claude_desc}\n")
                
    def sort_and_deduplicate(self):
        """Sort the final CSV file by feature_id and remove duplicates."""
        with self.write_lock:
            df = pd.read_csv(self.output_file)
            # Remove duplicates, keeping the last occurrence (most recent)
            df = df.drop_duplicates(subset=['feature_id'], keep='last')
            # Sort by feature_id
            df = df.sort_values('feature_id')
            # Save back to file
            df.to_csv(self.output_file, index=False)

class NeuronpediaClient:
    """Client for interacting with Neuronpedia API with rate limiting."""
    
    def __init__(self, api_key: str, base_url: str = "https://neuronpedia.org", rate_limit_delay: float = 0.1):
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key})
        self.lock = threading.Lock()
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            self.last_request_time = time.time()
    
    def generate_explanation(self, model_id: str, layer: str, index: int, 
                           explanation_type: str = EXPLANATION_TYPE,
                           explanation_model_name: str = EXPLANATION_MODEL_NAME) -> Optional[Dict[str, Any]]:
        """Generate a new explanation via POST request."""
        self._rate_limit()
        
        url = f"{self.base_url}/api/explanation/generate"
        payload = {
            "modelId": model_id,
            "layer": layer,
            "index": index,
            "explanationType": explanation_type,
            "explanationModelName": explanation_model_name
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Generate failed for feature {index}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error generating explanation for feature {index}: {e}")
            return None
    
    def get_feature(self, model_id: str, layer: str, index: int) -> Optional[Dict[str, Any]]:
        """Get existing feature data via GET request."""
        self._rate_limit()
        
        url = f"{self.base_url}/api/feature/{model_id}/{layer}/{index}"
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Get feature failed for {index}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting feature {index}: {e}")
            return None

def extract_claude_description(feature_data: Dict[str, Any], 
                              explanation_model_name: str = "claude-3-7-sonnet-20250219",
                              explanation_type: str = "eleuther_acts_top20") -> Optional[str]:
    """Extract Claude description from feature data."""
    if not feature_data or 'explanations' not in feature_data:
        return None
    
    explanations = feature_data['explanations']
    for explanation in explanations:
        if (explanation.get('explanationModelName') == explanation_model_name and 
            explanation.get('typeName') == explanation_type):
            return explanation.get('description')
    
    return None

def process_feature(client: NeuronpediaClient, writer: ThreadSafeCSVWriter, feature_id: int, 
                   model_id: str = MODEL_ID, 
                   layer: str = LAYER) -> tuple[int, Optional[str]]:
    """Process a single feature to get Claude description."""
    
    # Generate the link for this feature
    link = f"https://www.neuronpedia.org/{model_id}/{layer}/{feature_id}"
    
    # First try to generate a new explanation
    result = client.generate_explanation(model_id, layer, feature_id)
    
    description = None
    if result and 'description' in result:
        logger.info(f"Generated new explanation for feature {feature_id}")
        description = result['description']
    else:
        # If generation failed, try to get existing feature data
        feature_data = client.get_feature(model_id, layer, feature_id)
        
        if feature_data:
            description = extract_claude_description(feature_data)
            if description:
                logger.info(f"Found existing explanation for feature {feature_id}")
            else:
                logger.warning(f"No Claude explanation found for feature {feature_id}")
        else:
            logger.error(f"Failed to get any data for feature {feature_id}")
    
    # Save result immediately by appending to file
    writer.append_feature(feature_id, link, description)
    return feature_id, description

def main():
    """Main function to process the CSV file."""
    # Load environment variables
    load_dotenv(os.path.expanduser("~/.env"))
    api_key = os.getenv("NEURONPEDIA_API_KEY")
    
    if not api_key:
        logger.error("NEURONPEDIA_API_KEY not found in ~/.env file")
        return

    input_file = Path(INPUT_FILE)
    output_file = Path(OUTPUT_FILE)
    
    if not input_file.exists():
        logger.error(f"Input file {input_file} not found")
        return
    
    # Load the input CSV file
    logger.info(f"Loading input CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Add claude_desc column if it doesn't exist
    if 'claude_desc' not in df.columns:
        df['claude_desc'] = None
    
    # Load existing feature_ids from output file if it exists
    existing_features = set()
    if output_file.exists():
        logger.info(f"Loading existing features from output file: {output_file}")
        try:
            output_df = pd.read_csv(output_file)
            if 'feature_id' in output_df.columns and 'claude_desc' in output_df.columns:
                # Get feature_ids that already have Claude descriptions
                existing_features = set(output_df[
                    (output_df['claude_desc'].notna()) & 
                    (output_df['claude_desc'] != '')
                ]['feature_id'].tolist())
                logger.info(f"Found {len(existing_features)} features with existing explanations")
            else:
                logger.warning("Output file exists but missing required columns")
        except Exception as e:
            logger.warning(f"Error reading output file: {e}")
    
    # Get feature IDs that don't have Claude descriptions yet
    all_feature_ids = set(df['feature_id'].tolist())
    features_to_process = list(all_feature_ids - existing_features)
    
    if not features_to_process:
        logger.info("All features already have Claude descriptions")
        return
    
    logger.info(f"Processing {len(features_to_process)} features (skipping {len(existing_features)} already populated)")
    
    # Initialize client and writer
    client = NeuronpediaClient(api_key)
    writer = ThreadSafeCSVWriter(output_file)
    
    # Process features with threading
    completed_count = 0
    total_count = len(features_to_process)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_feature = {
            executor.submit(process_feature, client, writer, feature_id): feature_id 
            for feature_id in features_to_process
        }
        
        for future in as_completed(future_to_feature):
            feature_id = future_to_feature[future]
            try:
                result_feature_id, description = future.result()
                completed_count += 1
                logger.info(f"Completed processing feature {result_feature_id} ({completed_count}/{total_count})")
            except Exception as e:
                logger.error(f"Error processing feature {feature_id}: {e}")
                completed_count += 1
    
    # Sort and deduplicate the final output
    logger.info("Sorting and deduplicating final output...")
    writer.sort_and_deduplicate()
    
    # Print final summary
    final_df = pd.read_csv(output_file)
    total_features = len(final_df)
    features_with_claude = final_df['claude_desc'].notna().sum()
    logger.info(f"Final summary: {features_with_claude}/{total_features} features have Claude descriptions")
    logger.info(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()