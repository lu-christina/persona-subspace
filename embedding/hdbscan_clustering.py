#!/usr/bin/env python3
"""
HDBSCAN clustering script for embedding analysis.

Loads embeddings from parquet files, applies L2 normalization, and performs
HDBSCAN clustering with hyperparameter sweeps.
"""

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
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


def compute_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute clustering quality metrics."""
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    n_total = len(labels)
    noise_percentage = (n_noise / n_total) * 100

    metrics = {
        'n_clusters': int(n_clusters),
        'n_noise_points': int(n_noise),
        'n_total_points': int(n_total),
        'noise_percentage': float(noise_percentage),
    }

    # Compute cluster size distribution
    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[int(label)] = int(np.sum(labels == label))

    metrics['cluster_sizes'] = cluster_sizes

    # Compute silhouette score (only if we have clusters and non-noise points)
    if n_clusters > 1 and n_noise < n_total:
        # Only compute on non-noise points
        mask = labels != -1
        if np.sum(mask) > 0:
            try:
                silhouette = silhouette_score(
                    embeddings[mask],
                    labels[mask],
                    metric='euclidean'  # Using euclidean on L2-normalized = cosine
                )
                metrics['silhouette_score'] = float(silhouette)
            except Exception as e:
                print(f"Warning: Could not compute silhouette score: {e}")
                metrics['silhouette_score'] = None
        else:
            metrics['silhouette_score'] = None
    else:
        metrics['silhouette_score'] = None

    # Compute Davies-Bouldin score (only if we have clusters)
    if n_clusters > 1:
        mask = labels != -1
        if np.sum(mask) > 0:
            try:
                db_score = davies_bouldin_score(embeddings[mask], labels[mask])
                metrics['davies_bouldin_score'] = float(db_score)
            except Exception as e:
                print(f"Warning: Could not compute Davies-Bouldin score: {e}")
                metrics['davies_bouldin_score'] = None
        else:
            metrics['davies_bouldin_score'] = None
    else:
        metrics['davies_bouldin_score'] = None

    return metrics


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Run HDBSCAN clustering.

    Note: Using euclidean metric on L2-normalized embeddings is equivalent
    to using cosine distance, since ||a - b||^2 = 2(1 - cos(a,b)) for normalized vectors.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # Equivalent to cosine for L2-normalized vectors
        cluster_selection_epsilon=0.0,  # default, but explicit
        cluster_selection_method='eom',  # 'excess of mass', good for variable density
        core_dist_n_jobs=-1  # Use all available cores
    )

    labels = clusterer.fit_predict(embeddings)

    return labels


def save_results(
    df: pd.DataFrame,
    labels: np.ndarray,
    metrics: dict,
    output_dir: Path,
    min_cluster_size: int,
    min_samples: int
):
    """Save clustering results to output directory."""
    # Create subdirectory for this parameter combination
    subdir = output_dir / f"size{min_cluster_size}_samples{min_samples}"
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
        description="Run HDBSCAN clustering on embeddings with hyperparameter sweep"
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
        '--min-cluster-sizes',
        type=int,
        nargs='+',
        required=True,
        help='List of min_cluster_size values to sweep (e.g., 10 20 50)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        nargs='+',
        required=True,
        help='List of min_samples values to sweep (e.g., 1 5 10)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        help='Distance metric for HDBSCAN (default: euclidean, equivalent to cosine for L2-normalized vectors)'
    )

    args = parser.parse_args()

    # Load and prepare data (only once)
    df = load_embeddings(args.input_dir)
    embeddings_normalized, df = extract_and_normalize_embeddings(df)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameter sweep
    total_combinations = len(args.min_cluster_sizes) * len(args.min_samples)
    print(f"\nRunning {total_combinations} clustering combinations...")

    try:
        with tqdm(total=total_combinations, desc="Clustering sweep") as pbar:
            for min_cluster_size in args.min_cluster_sizes:
                for min_samples in args.min_samples:
                    pbar.set_description(
                        f"size={min_cluster_size}, samples={min_samples}"
                    )

                    # Run clustering
                    labels = run_hdbscan(
                        embeddings_normalized,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric=args.metric
                    )

                    # Compute metrics
                    metrics = compute_clustering_metrics(embeddings_normalized, labels)
                    metrics['min_cluster_size'] = min_cluster_size
                    metrics['min_samples'] = min_samples
                    metrics['metric'] = args.metric

                    # Save results
                    save_results(
                        df,
                        labels,
                        metrics,
                        args.output_dir,
                        min_cluster_size,
                        min_samples
                    )

                    pbar.update(1)

                    # Print summary
                    print(f"  Clusters: {metrics['n_clusters']}, "
                          f"Noise: {metrics['noise_percentage']:.1f}%")

        print(f"\nDone! Results saved to {args.output_dir}")
    except KeyboardInterrupt:
        print("\n\nClustering interrupted by user. Partial results may have been saved.")
        sys.exit(0)


if __name__ == '__main__':
    main()
