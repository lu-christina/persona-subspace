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

class ThreadSafeCSVWriter:
    """Thread-safe CSV writer that saves results incrementally."""
    
    def __init__(self, output_file: Path, df: pd.DataFrame):
        self.output_file = output_file
        self.df = df.copy()
        self.write_lock = threading.Lock()
        
    def update_feature(self, feature_id: int, claude_desc: Optional[str]):
        """Update a feature's claude_desc and save to file."""
        with self.write_lock:
            self.df.loc[self.df['feature_id'] == feature_id, 'claude_desc'] = claude_desc
            self.df.to_csv(self.output_file, index=False)
            
    def get_dataframe(self) -> pd.DataFrame:
        """Get current dataframe state."""
        with self.write_lock:
            return self.df.copy()

class NeuronpediaClient:
    """Client for interacting with Neuronpedia API with rate limiting."""
    
    def __init__(self, api_key: str, base_url: str = "https://neuronpedia.org", rate_limit_delay: float = 0.5):
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
                           explanation_type: str = "eleuther_acts_top20",
                           explanation_model_name: str = "claude-3-7-sonnet-20250219") -> Optional[Dict[str, Any]]:
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
                   model_id: str = "llama3.1-8b", 
                   layer: str = "15-llamascope-res-131k") -> tuple[int, Optional[str]]:
    """Process a single feature to get Claude description."""
    
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
    
    # Save result immediately
    writer.update_feature(feature_id, description)
    return feature_id, description

def main():
    """Main function to process the CSV file."""
    # Load environment variables
    load_dotenv(os.path.expanduser("~/.env"))
    api_key = os.getenv("NEURONPEDIA_API_KEY")
    
    if not api_key:
        logger.error("NEURONPEDIA_API_KEY not found in ~/.env file")
        return
    
    # File paths
    input_file = Path("llama_trainer32x_layer15/missing_explanations_with_claude.csv")
    output_file = Path("llama_trainer32x_layer15/missing_explanations_with_claude.csv")
    
    if not input_file.exists():
        logger.error(f"Input file {input_file} not found")
        return
    
    # Load the CSV file
    logger.info(f"Loading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Add claude_desc column if it doesn't exist
    if 'claude_desc' not in df.columns:
        df['claude_desc'] = None
    
    # Get feature IDs that don't have Claude descriptions yet (skip populated ones)
    features_to_process = df[(df['claude_desc'].isna()) | (df['claude_desc'] == '')]['feature_id'].tolist()
    
    if not features_to_process:
        logger.info("All features already have Claude descriptions")
        return
    
    logger.info(f"Processing {len(features_to_process)} features (skipping {len(df) - len(features_to_process)} already populated)")
    
    # Initialize client and writer
    client = NeuronpediaClient(api_key)
    writer = ThreadSafeCSVWriter(output_file, df)
    
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
    
    # Print final summary
    final_df = writer.get_dataframe()
    total_features = len(final_df)
    features_with_claude = final_df['claude_desc'].notna().sum()
    logger.info(f"Final summary: {features_with_claude}/{total_features} features have Claude descriptions")
    logger.info(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()