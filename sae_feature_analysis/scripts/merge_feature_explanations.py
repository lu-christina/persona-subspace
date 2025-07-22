#!/usr/bin/env python3
"""
Script to merge two CSV files based on feature_id column.

Takes a main CSV file with feature_id column and merges it with an explanations CSV file
that has feature_id and claude_desc columns, adding the claude_desc to the main file
where feature_ids match. Modifies the main CSV file in place.

Usage:
    python merge_csv.py <main_csv> <explanations_csv>
"""

import pandas as pd
import argparse
import sys
from pathlib import Path


def merge_csv_files(main_csv_path, explanations_csv_path):
    """
    Merge two CSV files based on feature_id column, modifying main CSV in place.
    
    Args:
        main_csv_path: Path to main CSV file with feature_id column (modified in place)
        explanations_csv_path: Path to explanations CSV with feature_id and claude_desc
    """
    try:
        # Read the CSV files
        main_df = pd.read_csv(main_csv_path)
        explanations_df = pd.read_csv(explanations_csv_path)
        
        # Validate required columns
        if 'feature_id' not in main_df.columns:
            raise ValueError(f"Main CSV {main_csv_path} missing 'feature_id' column")
        
        if 'feature_id' not in explanations_df.columns:
            raise ValueError(f"Explanations CSV {explanations_csv_path} missing 'feature_id' column")
            
        if 'claude_desc' not in explanations_df.columns:
            raise ValueError(f"Explanations CSV {explanations_csv_path} missing 'claude_desc' column")
        
        # Merge on feature_id, keeping all rows from main_df (left join)
        merged_df = main_df.merge(
            explanations_df[['feature_id', 'claude_desc']], 
            on='feature_id', 
            how='left'
        )
        
        # Save the merged result back to the main CSV file
        merged_df.to_csv(main_csv_path, index=False)
        
        # Print summary
        total_rows = len(main_df)
        matched_rows = len(merged_df[merged_df['claude_desc'].notna()])
        
        print(f"Merge completed successfully!")
        print(f"Total rows in main CSV: {total_rows}")
        print(f"Rows with matching explanations: {matched_rows}")
        print(f"Rows without explanations: {total_rows - matched_rows}")
        print(f"Main CSV file updated: {main_csv_path}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge CSV files based on feature_id column",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_csv.py features.csv explanations.csv
  python merge_csv.py data/main.csv data/explanations.csv
        """
    )
    
    parser.add_argument('main_csv', help='Main CSV file with feature_id column (modified in place)')
    parser.add_argument('explanations_csv', help='Explanations CSV with feature_id and claude_desc columns')
    
    args = parser.parse_args()
    
    # Validate input files exist
    main_path = Path(args.main_csv)
    explanations_path = Path(args.explanations_csv)
    
    if not main_path.exists():
        print(f"Error: Main CSV file does not exist: {args.main_csv}")
        sys.exit(1)
        
    if not explanations_path.exists():
        print(f"Error: Explanations CSV file does not exist: {args.explanations_csv}")
        sys.exit(1)
    
    merge_csv_files(args.main_csv, args.explanations_csv)


if __name__ == "__main__":
    main()