#!/usr/bin/env python3
"""
UMAP dimensionality reduction script for embedding analysis.

Loads embeddings from parquet files, joins with PC1 data, applies L2
normalization, and performs UMAP reduction to 2D and 3D with hyperparameter sweeps.
"""

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.manifold import trustworthiness
import umap
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


def load_pc1_data(pc1_file: Path) -> pd.DataFrame:
    """Load PC1 data from parquet file."""
    print(f"Loading PC1 data from {pc1_file}...")
    pc1_df = pd.read_parquet(pc1_file)
    print(f"Loaded {len(pc1_df)} PC1 rows")
    return pc1_df


def join_with_pc1(embeddings_df: pd.DataFrame, pc1_df: pd.DataFrame) -> pd.DataFrame:
    """Join embeddings with PC1 data on common keys."""
    # Common keys for joining
    join_keys = ['short_model', 'short_auditor_model', 'domain', 'persona_id', 'topic_id', 'response_id']

    print(f"Joining on keys: {join_keys}")

    # Perform left join to keep all embedding rows
    joined_df = embeddings_df.merge(
        pc1_df[join_keys + ['pc1_delta', 'prev_pc1', 'next_pc1']],
        on=join_keys,
        how='left'
    )

    n_with_pc1 = joined_df['pc1_delta'].notna().sum()
    print(f"Joined: {n_with_pc1}/{len(joined_df)} rows have PC1 data")

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


def compute_metrics(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    n_components: int
) -> Dict:
    """Compute metrics for UMAP embedding."""
    metrics = {
        'n_neighbors': int(n_neighbors),
        'min_dist': float(min_dist),
        'n_components': int(n_components),
        'n_samples': int(X_high.shape[0])
    }

    # Compute trustworthiness
    print(f"    Computing trustworthiness...")
    trust = trustworthiness(X_high, X_low, n_neighbors=n_neighbors)
    metrics['trustworthiness'] = float(trust)

    # Compute basic stats for each dimension
    print(f"    Computing basic statistics...")
    stats = {}
    for i in range(n_components):
        dim_name = f'umap_{i}'
        stats[dim_name] = {
            'mean': float(X_low[:, i].mean()),
            'std': float(X_low[:, i].std()),
            'min': float(X_low[:, i].min()),
            'max': float(X_low[:, i].max())
        }
    metrics['stats'] = stats

    return metrics


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    random_state: int = 42
) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    print(f"    Running UMAP (n_components={n_components})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric='euclidean',  # Using euclidean on L2-normalized = cosine distance
        n_jobs=-1
    )

    embedding_low = reducer.fit_transform(embeddings)

    return embedding_low


def save_results(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    coords_3d: np.ndarray,
    metrics_2d: Dict,
    metrics_3d: Dict,
    output_dir: Path,
    n_neighbors: int,
    min_dist: float
):
    """Save UMAP results to output directory."""
    # Create subdirectory for this parameter combination
    # Format min_dist to use dot instead of underscore
    dist_str = f"{min_dist:.10g}".replace('.', '.')  # Keep the dot
    subdir = output_dir / f"neighbors{n_neighbors}_dist{dist_str}"
    subdir.mkdir(parents=True, exist_ok=True)

    # Save 2D coordinates
    df_2d = df.copy()
    df_2d['umap_0'] = coords_2d[:, 0]
    df_2d['umap_1'] = coords_2d[:, 1]
    coords_2d_path = subdir / "coords_2d.parquet"
    df_2d.to_parquet(coords_2d_path, index=False)
    print(f"    Saved 2D coordinates to {coords_2d_path}")

    # Save 3D coordinates
    df_3d = df.copy()
    df_3d['umap_0'] = coords_3d[:, 0]
    df_3d['umap_1'] = coords_3d[:, 1]
    df_3d['umap_2'] = coords_3d[:, 2]
    coords_3d_path = subdir / "coords_3d.parquet"
    df_3d.to_parquet(coords_3d_path, index=False)
    print(f"    Saved 3D coordinates to {coords_3d_path}")

    # Save metrics (combined for both 2D and 3D)
    metrics = {
        'n_neighbors': int(n_neighbors),
        'min_dist': float(min_dist),
        '2d': metrics_2d,
        '3d': metrics_3d
    }
    metrics_path = subdir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run UMAP dimensionality reduction on embeddings with hyperparameter sweep"
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
        help='Output directory for UMAP results'
    )
    parser.add_argument(
        '--pc1-file',
        type=Path,
        required=True,
        help='Path to PC1 data parquet file'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        nargs='+',
        required=True,
        help='List of n_neighbors values to sweep (e.g., 5 15 30 50)'
    )
    parser.add_argument(
        '--min-dists',
        type=float,
        nargs='+',
        required=True,
        help='List of min_dist values to sweep (e.g., 0.0 0.1 0.5)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Load data
    embeddings_df = load_embeddings(args.input_dir)
    pc1_df = load_pc1_data(args.pc1_file)

    # Join with PC1 data
    df = join_with_pc1(embeddings_df, pc1_df)

    # Extract and normalize embeddings
    embeddings_normalized, df = extract_and_normalize_embeddings(df)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameter sweep
    total_combinations = len(args.n_neighbors) * len(args.min_dists)
    print(f"\nRunning UMAP for {total_combinations} parameter combinations...")

    try:
        with tqdm(total=total_combinations, desc="UMAP sweep") as pbar:
            for n_neighbors in args.n_neighbors:
                for min_dist in args.min_dists:
                    pbar.set_description(f"neighbors={n_neighbors}, dist={min_dist}")

                    # Run UMAP for 2D
                    coords_2d = run_umap(
                        embeddings_normalized,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                        random_state=args.random_state
                    )

                    # Compute metrics for 2D
                    metrics_2d = compute_metrics(
                        embeddings_normalized,
                        coords_2d,
                        n_neighbors,
                        min_dist,
                        n_components=2
                    )

                    # Run UMAP for 3D
                    coords_3d = run_umap(
                        embeddings_normalized,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=3,
                        random_state=args.random_state
                    )

                    # Compute metrics for 3D
                    metrics_3d = compute_metrics(
                        embeddings_normalized,
                        coords_3d,
                        n_neighbors,
                        min_dist,
                        n_components=3
                    )

                    # Save results
                    save_results(
                        df,
                        coords_2d,
                        coords_3d,
                        metrics_2d,
                        metrics_3d,
                        args.output_dir,
                        n_neighbors,
                        min_dist
                    )

                    pbar.update(1)

                    # Print summary
                    print(f"  Trustworthiness (2D): {metrics_2d['trustworthiness']:.3f}, "
                          f"(3D): {metrics_3d['trustworthiness']:.3f}")

        print(f"\nDone! Results saved to {args.output_dir}")
    except KeyboardInterrupt:
        print("\n\nUMAP interrupted by user. Partial results may have been saved.")
        sys.exit(0)


if __name__ == '__main__':
    main()
