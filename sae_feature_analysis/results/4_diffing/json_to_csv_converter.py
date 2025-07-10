#!/usr/bin/env python3
"""
Script to convert explanations.json to CSV format with feature_id, explanation, and link columns.
"""

import json
import pandas as pd
import argparse

def json_to_csv(json_file_path, csv_file_path):
    """
    Convert explanations JSON file to CSV format.
    
    Args:
        json_file_path: Path to the input JSON file
        csv_file_path: Path to the output CSV file
    """
    print(f"Loading explanations from {json_file_path}")
    
    try:
        with open(json_file_path, 'r') as f:
            explanations_data = json.load(f)
        
        print(f"Loaded {len(explanations_data)} explanations")
        
        # Prepare data for CSV
        csv_data = []
        
        for feature_id, feature_data in explanations_data.items():
            # Extract explanation text
            explanation_text = ""
            if isinstance(feature_data, dict):
                explanations = feature_data.get('explanations', {})
                if isinstance(explanations, list) and explanations:
                    # Take first explanation from list
                    explanations = explanations[0]
                if isinstance(explanations, dict):
                    explanation_text = explanations.get('description', '').strip()
            
            # Create Neuronpedia link
            link = f"https://www.neuronpedia.org/llama3.1-8b/15-llamascope-res-131k/{feature_id}"
            
            csv_data.append({
                'feature_id': feature_id,
                'explanation': explanation_text,
                'link': link
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df = df.sort_values('feature_id', key=lambda x: x.astype(int))  # Sort by feature_id numerically
        df.to_csv(csv_file_path, index=False)
        
        print(f"Saved {len(df)} entries to {csv_file_path}")
        
        # Print some statistics
        non_empty_explanations = (df['explanation'] != '').sum()
        print(f"Features with explanations: {non_empty_explanations}/{len(df)}")
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert explanations JSON to CSV format')
    parser.add_argument('--input', '-i', 
                       default='/workspace/sae/llama-3.1-8b/saes/resid_post_layer_15/trainer_32x/explanations.json',
                       help='Input JSON file path')
    parser.add_argument('--output', '-o', 
                       default='explanations.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    json_to_csv(args.input, args.output)

if __name__ == "__main__":
    main()