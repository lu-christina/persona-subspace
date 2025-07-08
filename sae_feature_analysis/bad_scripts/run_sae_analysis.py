#!/usr/bin/env python3
"""
Script to run SAE feature analysis across all model combinations.
Supports multiple analysis types: universal, specific, or both.
"""

import os
import argparse
from sae_feature_analysis import UniversalFeatureAnalyzer, SpecificFeatureAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run SAE feature analysis')
    parser.add_argument('--analysis', choices=['universal', 'specific', 'both'], 
                       default='universal', help='Type of analysis to run')
    
    # Universal analysis options
    parser.add_argument('--universal-output', type=str,
                       help='Exact output file path for universal analysis results (including filename)')
    parser.add_argument('--threshold', type=float, default=0.3, 
                       help='Prompt threshold for universal analysis (default: 0.3)')
    parser.add_argument('--activation-threshold', type=float, default=0.01,
                       help='Activation threshold (default: 0.01)')
    
    # Specific analysis options
    parser.add_argument('--features-csv', type=str, 
                       help='Path to CSV file with feature_id and source columns for specific analysis')
    parser.add_argument('--specific-output', type=str,
                       help='Exact output file path for specific analysis results (including filename)')
    parser.add_argument('--record-prompts', action='store_true',
                       help='Record prompts that activate features above threshold to JSONL files')
    
    # Common options
    parser.add_argument('--prompts-path', type=str, default='./prompts',
                       help='Path to prompts directory or specific .jsonl file (default: ./prompts)')
    
    return parser.parse_args()


def main():
    """Run SAE feature analysis."""
    args = parse_args()
    
    # Validate arguments
    if args.analysis in ['universal', 'both']:
        if not args.universal_output:
            print("Error: --universal-output required for universal analysis")
            return
    
    if args.analysis in ['specific', 'both']:
        if not args.features_csv:
            print("Error: --features-csv required for specific analysis")
            return
        if not args.specific_output:
            print("Error: --specific-output required for specific analysis")
            return
    
    # Run universal analysis
    if args.analysis in ['universal', 'both']:
        print("Starting universal feature analysis...")
        
        analyzer = UniversalFeatureAnalyzer()
        analyzer.run_analysis(
            output_path=args.universal_output,
            prompts_path=args.prompts_path,
            activation_threshold=args.activation_threshold,
            prompt_threshold=args.threshold
        )
    
    # Run specific analysis  
    if args.analysis in ['specific', 'both']:
        print("Starting specific feature analysis...")
        
        analyzer = SpecificFeatureAnalyzer(features_csv_path=args.features_csv)
        analyzer.run_analysis(
            output_path=args.specific_output,
            prompts_path=args.prompts_path,
            features_csv_path=args.features_csv,
            record_prompts=args.record_prompts,
            activation_threshold=args.activation_threshold
        )


if __name__ == "__main__":
    main()