#!/usr/bin/env python3

import pandas as pd
import os
from pathlib import Path

def process_csv_files():
    """Process all CSV files in ./results/universal/, count features, and combine them."""
    
    results_dir = Path("./results/universal")
    output_file = "combined.csv"
    
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist")
        return
    
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in ./results/universal/")
        return
    
    print("Feature counts per file:")
    print("-" * 40)
    
    combined_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            feature_count = len(df)
            print(f"{csv_file.name}: {feature_count} features")
            
            # Add source column
            df['source'] = csv_file.name
            combined_data.append(df)
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    if combined_data:
        # Combine all dataframes
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Save to combined.csv
        combined_df.to_csv(output_file, index=False)
        print(f"\nCombined {len(combined_data)} files into {output_file}")
        print(f"Total features: {len(combined_df)}")
    else:
        print("No data to combine")

if __name__ == "__main__":
    process_csv_files()