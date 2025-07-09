#!/usr/bin/env python3
"""
Script to download feature explanations from Neuronpedia API and merge them with CSV files.

This script:
1. Downloads all feature explanations from Neuronpedia for a specific model/SAE
2. Saves the explanations to disk as JSON
3. Reads CSV files from llama_trainer32x_layer15 directory
4. Adds explanation columns to the CSV files based on feature_id matches
"""

import requests
import json
import pandas as pd
import os
from pathlib import Path
import time
from typing import Dict, List, Optional
import argparse


class NeuronpediaFeatureUpdater:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Neuronpedia feature updater.
        
        Args:
            api_key: Neuronpedia API key. If None, will look for NEURONPEDIA_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('NEURONPEDIA_API_KEY')
        self.base_url = "https://www.neuronpedia.org/api"
        self.headers = {'x-api-key': self.api_key} if self.api_key else {}
        
        # Based on the CSV structure, we're working with llama3.1-8b model, layer 15
        # Note: Need to verify correct model/SAE IDs with Neuronpedia
        self.model_id = "llama3.1-8b"
        self.sae_id = "15-llamascope-res-131k"
        
    def download_explanations(self, output_file: str = "neuronpedia_explanations.json") -> Dict:
        """
        Download all feature explanations from Neuronpedia for the specified model/SAE.
        
        Args:
            output_file: Path to save the downloaded explanations
            
        Returns:
            Dictionary containing explanations indexed by feature_id
        """
        print(f"Downloading explanations for {self.model_id}/{self.sae_id}...")
        
        # Use the export endpoint to get all explanations for the SAE
        url = f"{self.base_url}/explanation/export"
        params = {
            'modelId': self.model_id,
            'saeId': self.sae_id
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            explanations_data = response.json()
            print(f"Downloaded {len(explanations_data)} explanations")
            
            # Index explanations by feature_id for easy lookup
            explanations_by_id = {}
            for explanation in explanations_data:
                feature_id = explanation.get('index')  # 'index' is the feature_id
                if feature_id is not None:
                    # Convert to string for consistent lookup
                    explanations_by_id[str(feature_id)] = explanation
            
            # Save to disk
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(explanations_by_id, f, indent=2)
            
            print(f"Saved explanations to {output_path}")
            return explanations_by_id
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading explanations: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
                if e.response.status_code == 404:
                    print("\nHint: The model/SAE ID combination might not exist in Neuronpedia.")
                    print("Check the Neuronpedia website for the correct model and SAE IDs.")
                    print("You can also try using individual feature lookup via search endpoints.")
            return {}
    
    def load_explanations(self, file_path: str) -> Dict:
        """
        Load explanations from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing explanations
            
        Returns:
            Dictionary containing explanations indexed by feature_id
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Explanations file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            return {}
    
    def update_csv_with_explanations(self, csv_path: str, explanations: Dict, 
                                   output_path: Optional[str] = None) -> bool:
        """
        Update a CSV file by adding explanation columns based on feature_id matches.
        
        Args:
            csv_path: Path to the input CSV file
            explanations: Dictionary of explanations indexed by feature_id
            output_path: Path for output CSV. If None, overwrites input file.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            if 'feature_id' not in df.columns:
                print(f"Error: 'feature_id' column not found in {csv_path}")
                return False
            
            # Add explanation columns - convert feature_id to string for lookup
            df['explanation'] = df['feature_id'].apply(
                lambda fid: explanations.get(str(fid), {}).get('description', '')
            )
            
            # Add additional explanation metadata
            df['explanation_score'] = df['feature_id'].apply(
                lambda fid: explanations.get(str(fid), {}).get('score', None)
            )
            
            # Save the updated CSV
            output_file = output_path or csv_path
            df.to_csv(output_file, index=False)
            
            matched_count = (df['explanation'] != '').sum()
            print(f"Updated {csv_path}: {matched_count}/{len(df)} features have explanations")
            
            return True
            
        except Exception as e:
            print(f"Error updating CSV {csv_path}: {e}")
            return False
    
    def update_all_csvs(self, csv_directory: str, explanations: Dict) -> None:
        """
        Update all CSV files in the specified directory with explanations.
        
        Args:
            csv_directory: Directory containing CSV files to update
            explanations: Dictionary of explanations indexed by feature_id
        """
        csv_dir = Path(csv_directory)
        
        if not csv_dir.exists():
            print(f"Error: Directory {csv_directory} does not exist")
            return
        
        csv_files = list(csv_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {csv_directory}")
            return
        
        print(f"Found {len(csv_files)} CSV files to update")
        
        for csv_file in csv_files:
            print(f"Processing {csv_file.name}...")
            self.update_csv_with_explanations(str(csv_file), explanations)
    
    def run(self, csv_directory: str = "llama_trainer32x_layer15", 
            explanations_file: str = "neuronpedia_explanations.json",
            download_fresh: bool = False) -> None:
        """
        Main execution method.
        
        Args:
            csv_directory: Directory containing CSV files to update
            explanations_file: Path to explanations JSON file
            download_fresh: Whether to download fresh explanations or use cached file
        """
        # Download or load explanations
        if download_fresh or not os.path.exists(explanations_file):
            explanations = self.download_explanations(explanations_file)
        else:
            print(f"Loading cached explanations from {explanations_file}")
            explanations = self.load_explanations(explanations_file)
        
        if not explanations:
            print("No explanations available. Exiting.")
            return
        
        # Update CSV files
        self.update_all_csvs(csv_directory, explanations)
        
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Download Neuronpedia explanations and update CSV files')
    parser.add_argument('--api-key', help='Neuronpedia API key (or set NEURONPEDIA_API_KEY env var)')
    parser.add_argument('--csv-dir', default='llama_trainer32x_layer15', 
                       help='Directory containing CSV files to update')
    parser.add_argument('--explanations-file', default='neuronpedia_explanations.json',
                       help='Path to explanations JSON file')
    parser.add_argument('--download-fresh', action='store_true',
                       help='Download fresh explanations instead of using cached file')
    
    args = parser.parse_args()
    
    # Initialize the updater
    updater = NeuronpediaFeatureUpdater(api_key=args.api_key)
    
    # Run the update process
    updater.run(
        csv_directory=args.csv_dir,
        explanations_file=args.explanations_file,
        download_fresh=args.download_fresh
    )


if __name__ == "__main__":
    main()