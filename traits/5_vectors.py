#!/usr/bin/env python3
"""
5_vectors.py - Generate trait vectors from response activations and scores

Creates 4 types of vectors for each trait:
1. pos_neg: mean(pos) - mean(neg) for all pairs
2. pos_neg_{threshold}: mean(pos) - mean(neg) for pairs with score difference > threshold  
3. pos_default: mean(pos) - mean(default) for all pairs
4. pos_default_{threshold}: mean(pos) - mean(default) for pairs with score difference > threshold

Each vector has shape (n_layers, hidden_dim) = (46, 4608)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import torch
from tqdm import tqdm


def load_data(trait: str) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, Union[int, str]]]]:
    """Load activation tensors and scores for a given trait."""
    activations_path = f"/workspace/traits/response_activations/{trait}.pt"
    scores_path = f"/workspace/traits/extract_scores/{trait}.json"
    
    try:
        activations = torch.load(activations_path, map_location='cpu')
        with open(scores_path, 'r') as f:
            scores = json.load(f)
        return activations, scores
    except FileNotFoundError as e:
        print(f"Warning: Missing file for {trait}: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading data for {trait}: {e}")
        return None, None


def compute_vectors(
    activations: Dict[str, torch.Tensor], 
    scores: Dict[str, Union[int, str]],
    pos_neg_threshold: int,
    pos_default_threshold: int
) -> Dict[str, torch.Tensor]:
    """Compute the 4 vector types from activations and scores."""
    
    # Collect all pairs for different conditions
    pos_neg_all_pairs = []
    pos_neg_filtered_pairs = []
    pos_default_all_pairs = []
    pos_default_filtered_pairs = []
    
    # Iterate through all prompt and question combinations
    for prompt_idx in range(5):  # p0 to p4
        for question_idx in range(240):  # q0 to q19
            pos_key = f"pos_p{prompt_idx}_q{question_idx}"
            neg_key = f"neg_p{prompt_idx}_q{question_idx}"
            default_key = f"default_p{prompt_idx}_q{question_idx}"
            
            # Check if all required keys exist
            if pos_key in activations and pos_key in scores:
                pos_activation = activations[pos_key]
                pos_score = scores[pos_key]
                
                # Skip if pos_score is REFUSAL
                if pos_score == "REFUSAL":
                    continue
                
                # Process pos-neg pairs
                if neg_key in activations and neg_key in scores:
                    neg_activation = activations[neg_key]
                    neg_score = scores[neg_key]
                    
                    # Skip if neg_score is REFUSAL
                    if neg_score == "REFUSAL":
                        continue
                    
                    # Add to all pairs
                    pos_neg_all_pairs.append((pos_activation, neg_activation))
                    
                    # Add to filtered pairs if score difference exceeds threshold
                    if (pos_score - neg_score) > pos_neg_threshold:
                        pos_neg_filtered_pairs.append((pos_activation, neg_activation))
                
                # Process pos-default pairs  
                if default_key in activations and default_key in scores:
                    default_activation = activations[default_key]
                    default_score = scores[default_key]
                    
                    # Skip if default_score is REFUSAL
                    if default_score == "REFUSAL":
                        continue
                    
                    # Add to all pairs
                    pos_default_all_pairs.append((pos_activation, default_activation))
                    
                    # Add to filtered pairs if score difference exceeds threshold
                    if (pos_score - default_score) > pos_default_threshold:
                        pos_default_filtered_pairs.append((pos_activation, default_activation))
    
    # Compute mean differences for each vector type
    def compute_mean_difference(pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not pairs:
            # Return zeros tensor with correct shape (46, 4608)
            return torch.zeros(46, 4608)
        
        pos_tensors = [pair[0] for pair in pairs]
        comparison_tensors = [pair[1] for pair in pairs]
        
        pos_mean = torch.stack(pos_tensors).mean(dim=0)
        comparison_mean = torch.stack(comparison_tensors).mean(dim=0)
        
        return pos_mean - comparison_mean
    
    # Build result dictionary with dynamic keys
    result = {
        'pos_neg': compute_mean_difference(pos_neg_all_pairs),
        f'pos_neg_{pos_neg_threshold}': compute_mean_difference(pos_neg_filtered_pairs),
        'pos_default': compute_mean_difference(pos_default_all_pairs),
        f'pos_default_{pos_default_threshold}': compute_mean_difference(pos_default_filtered_pairs)
    }
    
    return result


def process_trait(trait: str, pos_neg_threshold: int, pos_default_threshold: int) -> bool:
    """Process a single trait and save vectors."""
    activations, scores = load_data(trait)
    
    if activations is None or scores is None:
        return False
    
    vectors = compute_vectors(activations, scores, pos_neg_threshold, pos_default_threshold)
    
    # Save vectors
    output_path = f"/workspace/traits/vectors/{trait}.pt"
    torch.save(vectors, output_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate trait vectors from activations and scores")
    parser.add_argument("--pos_neg_threshold", type=int, default=50,
                       help="Score difference threshold for pos-neg filtering (default: 50)")
    parser.add_argument("--pos_default_threshold", type=int, default=50,
                       help="Score difference threshold for pos-default filtering (default: 50)")
    parser.add_argument("--traits", nargs="+", 
                       help="Specific traits to process (default: all traits)")
    parser.add_argument("--list_traits", action="store_true",
                       help="List available traits and exit")
    
    args = parser.parse_args()
    
    # Get list of available traits from activations directory
    activations_dir = Path("/workspace/traits/response_activations")
    available_traits = [f.stem for f in activations_dir.glob("*.pt")]
    available_traits.sort()
    
    if args.list_traits:
        print(f"Available traits ({len(available_traits)}):")
        for trait in available_traits:
            print(f"  {trait}")
        return
    
    # Determine which traits to process
    if args.traits:
        traits_to_process = [t for t in args.traits if t in available_traits]
        if len(traits_to_process) != len(args.traits):
            missing = set(args.traits) - set(traits_to_process)
            print(f"Warning: Traits not found: {missing}")
    else:
        traits_to_process = available_traits
    
    print(f"Processing {len(traits_to_process)} traits with thresholds:")
    print(f"  pos_neg_threshold: {args.pos_neg_threshold}")
    print(f"  pos_default_threshold: {args.pos_default_threshold}")
    
    # Ensure output directory exists
    os.makedirs("/workspace/traits/vectors", exist_ok=True)
    
    # Process traits with progress bar
    successful = 0
    failed = 0
    
    for trait in tqdm(traits_to_process, desc="Processing traits"):
        if process_trait(trait, args.pos_neg_threshold, args.pos_default_threshold):
            successful += 1
        else:
            failed += 1
    
    print(f"\nCompleted: {successful} successful, {failed} failed")
    print(f"Vectors saved to data/vectors/")


if __name__ == "__main__":
    main()