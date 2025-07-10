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
from typing import Dict, List, Optional, Set
import argparse
import pickle
from collections import defaultdict


class NeuronpediaFeatureUpdater:
    def __init__(self, api_key: Optional[str] = None, cache_file: str = "feature_cache.pkl"):
        """
        Initialize the Neuronpedia feature updater.
        
        Args:
            api_key: Neuronpedia API key. If None, will look for NEURONPEDIA_API_KEY env var.
            cache_file: Path to pickle file for caching feature explanations.
        """
        self.api_key = api_key or os.getenv('NEURONPEDIA_API_KEY')
        self.base_url = "https://www.neuronpedia.org/api"
        self.headers = {'x-api-key': self.api_key} if self.api_key else {}
        
        # Based on the CSV structure, we're working with llama3.1-8b model, layer 15
        # Note: Need to verify correct model/SAE IDs with Neuronpedia
        self.model_id = "llama3.1-8b"
        self.sae_id = "15-llamascope-res-131k"
        
        # Caching and rate limiting
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests (10 requests/second)
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cached feature explanations from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"Loaded {len(cache)} cached explanations from {self.cache_file}")
                return cache
            except Exception as e:
                print(f"Error loading cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_feature_explanation(self, feature_id: str) -> Optional[Dict]:
        """
        Get explanation for a single feature, using cache if available.
        
        Args:
            feature_id: The feature ID to get explanation for
            
        Returns:
            Feature explanation dictionary or None if not found
        """
        # Check cache first
        cache_key = f"{self.model_id}_{self.sae_id}_{feature_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Make API request
        self._rate_limit()
        url = f"{self.base_url}/feature/{self.model_id}/{self.sae_id}/{feature_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            feature_data = response.json()
            
            # Cache the result
            self.cache[cache_key] = feature_data
            
            if self.request_count % 50 == 0:  # Save cache every 50 requests
                self._save_cache()
                print(f"Made {self.request_count} API requests so far...")
            
            return feature_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching feature {feature_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:  # Rate limited
                    print("Rate limited - increasing delay...")
                    self.min_request_interval *= 1.5
                    time.sleep(1)  # Extra pause
            return None
        
    def download_explanations(self, feature_ids: List[str], output_file: str = "neuronpedia_explanations.json") -> Dict:
        """
        Download feature explanations individually from Neuronpedia for the specified feature IDs.
        
        Args:
            feature_ids: List of feature IDs to download explanations for
            output_file: Path to save the downloaded explanations
            
        Returns:
            Dictionary containing explanations indexed by feature_id
        """
        # Deduplicate feature IDs while preserving order
        unique_feature_ids = list(dict.fromkeys(str(fid) for fid in feature_ids))
        total_features = len(unique_feature_ids)
        
        print(f"Downloading explanations for {total_features} unique features (from {len(feature_ids)} total)")
        print(f"Model: {self.model_id}, SAE: {self.sae_id}")
        
        explanations_by_id = {}
        cached_count = 0
        
        for i, feature_id in enumerate(unique_feature_ids):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{total_features} features processed")
            
            cache_key = f"{self.model_id}_{self.sae_id}_{feature_id}"
            if cache_key in self.cache:
                cached_count += 1
                feature_data = self.cache[cache_key]
            else:
                feature_data = self.get_feature_explanation(feature_id)
            
            if feature_data:
                explanations_by_id[feature_id] = feature_data
        
        # Save final cache
        self._save_cache()
        
        print(f"Downloaded {len(explanations_by_id)} explanations")
        print(f"Used cache for {cached_count} features")
        print(f"Made {self.request_count} new API requests")
        
        # Save explanations to JSON file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(explanations_by_id, f, indent=2)
        
        print(f"Saved explanations to {output_path}")
        return explanations_by_id
    
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
            
            # Helper function to safely extract explanation data
            def get_explanation_text(fid):
                feature_data = explanations.get(str(fid))
                if not feature_data:
                    return ''
                explanations_data = feature_data.get('explanations')
                if isinstance(explanations_data, list) and explanations_data:
                    # Take first explanation from list
                    explanations_data = explanations_data[0]
                if not isinstance(explanations_data, dict):
                    return ''
                return explanations_data.get('description', '').strip()
            
            def get_explanation_score(fid):
                feature_data = explanations.get(str(fid))
                if not feature_data:
                    return None
                explanations_data = feature_data.get('explanations')
                if isinstance(explanations_data, list) and explanations_data:
                    # Take first explanation from list
                    explanations_data = explanations_data[0]
                if not isinstance(explanations_data, dict):
                    return None
                return explanations_data.get('score', None)
            
            # Add explanation columns
            df['explanation'] = df['feature_id'].apply(get_explanation_text)
            df['explanation_score'] = df['feature_id'].apply(get_explanation_score)
            
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
    
    def get_feature_ids_from_csvs(self, csv_directory: str) -> List[str]:
        """
        Extract all unique feature IDs from CSV files in the directory.
        
        Args:
            csv_directory: Directory containing CSV files
            
        Returns:
            List of unique feature IDs found in the CSV files
        """
        csv_dir = Path(csv_directory)
        if not csv_dir.exists():
            print(f"Error: Directory {csv_directory} does not exist")
            return []
        
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {csv_directory}")
            return []
        
        all_feature_ids = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'feature_id' in df.columns:
                    feature_ids = df['feature_id'].astype(str).tolist()
                    all_feature_ids.extend(feature_ids)
                    print(f"Found {len(feature_ids)} feature IDs in {csv_file.name}")
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        return all_feature_ids

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
        # Get all feature IDs from CSV files first
        feature_ids = self.get_feature_ids_from_csvs(csv_directory)
        if not feature_ids:
            print("No feature IDs found in CSV files. Exiting.")
            return
        
        print(f"Found {len(feature_ids)} total feature IDs across all CSV files")
        
        # Download or load explanations
        if download_fresh or not os.path.exists(explanations_file):
            explanations = self.download_explanations(feature_ids, explanations_file)
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
    parser.add_argument('--explanations-file', default='/workspace/sae/llama-3.1-8b/saes/resid_post_layer_15/trainer_32x/explanations.json',
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