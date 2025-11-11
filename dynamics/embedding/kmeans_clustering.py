#!/usr/bin/env python3
"""
K-means clustering script for embedding analysis with PC1 statistics.

Loads embeddings from parquet files, joins with PC1 data, applies L2
normalization, and performs k-means clustering with per-cluster PC1 analysis.
Supports analyzing different PC1 metrics (pc1_delta, next_pc1, prev_pc1).
"""

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nInterrupted by user. Exiting...")
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def load_embeddings(input_dir: Path) -> pd.DataFrame:
    """Load all parquet shards from input directory."""
    parquet_files = sorted(input_dir.glob("shard-*.parquet"))

    if not parquet_files:
        parquet_files = sorted(input_dir.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_dir}")

    print(f"Loading {len(parquet_files)} parquet file(s)...")
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} total rows")

    return combined_df


def load_pc1_deltas(pc1_file: Path) -> pd.DataFrame:
    """Load PC1 delta data from parquet file."""
    print(f"Loading PC1 delta data from {pc1_file}...")
    pc1_df = pd.read_parquet(pc1_file)
    print(f"Loaded {len(pc1_df)} PC1 delta rows")
    return pc1_df


def join_with_pc1(embeddings_df: pd.DataFrame, pc1_df: pd.DataFrame) -> pd.DataFrame:
    """Join embeddings with PC1 delta data on common keys."""
    # Common keys for joining
    join_keys = ['short_model', 'short_auditor_model', 'domain', 'persona_id', 'topic_id', 'response_id']

    # Check if all join keys exist in both dataframes
    missing_in_embeddings = set(join_keys) - set(embeddings_df.columns)
    missing_in_pc1 = set(join_keys) - set(pc1_df.columns)

    if missing_in_embeddings:
        raise ValueError(f"Missing columns in embeddings: {missing_in_embeddings}")
    if missing_in_pc1:
        raise ValueError(f"Missing columns in PC1 data: {missing_in_pc1}")

    print(f"Joining on keys: {join_keys}")

    # Perform left join to keep all embedding rows
    joined_df = embeddings_df.merge(
        pc1_df[join_keys + ['pc1_delta', 'prev_pc1', 'next_pc1']],
        on=join_keys,
        how='left'
    )

    n_with_pc1 = joined_df['pc1_delta'].notna().sum()
    print(f"Joined: {n_with_pc1}/{len(joined_df)} rows have PC1 delta data")

    return joined_df


def extract_and_normalize_embeddings(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extract embeddings from dataframe and apply L2 normalization."""
    print("Extracting embeddings...")

    # Convert embedding column to numpy array
    embeddings = np.stack(df['embedding'].values)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Applying L2 normalization...")

    # L2 normalize the embeddings
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    return embeddings_normalized, df


def compute_cluster_pc1_stats(df: pd.DataFrame, labels: np.ndarray, k: int, target_metric: str = 'pc1_delta') -> Dict:
    """Compute per-cluster PC1 statistics for specified target metric."""
    df_with_labels = df.copy()
    df_with_labels['cluster_label'] = labels

    metrics = {
        'k': int(k),
        'target_metric': target_metric,
        'n_total_points': int(len(df)),
        'clusters': {}
    }

    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_df = df_with_labels[cluster_mask]

        # Get rows with target metric data
        cluster_with_metric = cluster_df[cluster_df[target_metric].notna()]

        cluster_stats = {
            'n_points': int(cluster_mask.sum()),
            'n_with_metric': int(len(cluster_with_metric))
        }

        # Compute stats if we have data
        if len(cluster_with_metric) > 0:
            cluster_stats[f'{target_metric}_mean'] = float(cluster_with_metric[target_metric].mean())
            cluster_stats[f'{target_metric}_std'] = float(cluster_with_metric[target_metric].std())
        else:
            cluster_stats[f'{target_metric}_mean'] = None
            cluster_stats[f'{target_metric}_std'] = None

        metrics['clusters'][str(cluster_id)] = cluster_stats

    return metrics


def run_kmeans(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10
) -> np.ndarray:
    """Run k-means clustering.

    Note: Using euclidean metric on L2-normalized embeddings is equivalent
    to using cosine distance.
    """
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=300
    )

    labels = kmeans.fit_predict(embeddings)

    return labels


def save_results(
    df: pd.DataFrame,
    labels: np.ndarray,
    metrics: dict,
    output_dir: Path,
    k: int
):
    """Save clustering results to output directory."""
    # Create subdirectory for this k value
    subdir = output_dir / f"k{k}"
    subdir.mkdir(parents=True, exist_ok=True)

    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster_label'] = labels

    # Save clusters.parquet
    clusters_path = subdir / "clusters.parquet"
    df_with_clusters.to_parquet(clusters_path, index=False)
    print(f"  Saved clusters to {clusters_path}")

    # Save metrics.json
    metrics_path = subdir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run k-means clustering on embeddings with PC1 metric analysis"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing parquet files with embeddings'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for clustering results'
    )
    parser.add_argument(
        '--pc1-file',
        type=Path,
        required=True,
        help='Path to PC1 deltas parquet file'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        required=True,
        help='List of k values to sweep (e.g., 5 10 20 50)'
    )
    parser.add_argument(
        '--target-metric',
        type=str,
        default='pc1_delta',
        choices=['pc1_delta', 'next_pc1', 'prev_pc1'],
        help='Target metric to analyze per cluster (default: pc1_delta)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--n-init',
        type=int,
        default=10,
        help='Number of k-means initializations (default: 10)'
    )

    args = parser.parse_args()

    # Load data
    embeddings_df = load_embeddings(args.input_dir)
    pc1_df = load_pc1_deltas(args.pc1_file)

    # Join with PC1 data
    df = join_with_pc1(embeddings_df, pc1_df)

    # Extract and normalize embeddings
    embeddings_normalized, df = extract_and_normalize_embeddings(df)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # K-means sweep
    print(f"\nRunning k-means for {len(args.k_values)} values of k...")
    print(f"Target metric: {args.target_metric}")

    try:
        with tqdm(total=len(args.k_values), desc="K-means sweep") as pbar:
            for k in args.k_values:
                pbar.set_description(f"k={k}")

                # Run k-means
                labels = run_kmeans(
                    embeddings_normalized,
                    k=k,
                    random_state=args.random_state,
                    n_init=args.n_init
                )

                # Compute per-cluster stats for target metric
                metrics = compute_cluster_pc1_stats(df, labels, k, target_metric=args.target_metric)

                # Save results
                save_results(
                    df,
                    labels,
                    metrics,
                    args.output_dir,
                    k
                )

                pbar.update(1)

                # Print summary
                n_with_metric = sum(
                    c['n_with_metric']
                    for c in metrics['clusters'].values()
                )
                print(f"  k={k}: {n_with_metric} points with {args.target_metric} data across {k} clusters")

        print(f"\nDone! Results saved to {args.output_dir}")
    except KeyboardInterrupt:
        print("\n\nClustering interrupted by user. Partial results may have been saved.")
        sys.exit(0)


if __name__ == '__main__':
    main()
