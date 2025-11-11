#!/usr/bin/env python3
"""
Ridge regression script for predicting PC1 deltas from embeddings.

Loads embeddings and PC1 delta data, performs L2 normalization, and trains
ridge regression using RidgeCV which automatically selects the best alpha
via cross-validation.
"""

import argparse
import json
import pickle
import signal
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def join_and_prepare_data(
    embeddings_df: pd.DataFrame,
    pc1_df: pd.DataFrame,
    target_metric: str = 'pc1_delta'
) -> Tuple[np.ndarray, np.ndarray]:
    """Join embeddings with PC1 data and prepare for regression."""
    # Common keys for joining
    join_keys = ['short_model', 'short_auditor_model', 'domain', 'persona_id', 'topic_id', 'response_id']

    print(f"Joining on keys: {join_keys}")
    print(f"Target metric: {target_metric}")

    # Perform inner join to keep only rows with both embedding and PC1 data
    joined_df = embeddings_df.merge(
        pc1_df[join_keys + ['pc1_delta', 'prev_pc1', 'next_pc1']],
        on=join_keys,
        how='inner'
    )

    print(f"After join: {len(joined_df)} rows with both embeddings and PC1 data")

    # Extract embeddings
    embeddings = np.stack(joined_df['embedding'].values)
    print(f"Embeddings shape: {embeddings.shape}")

    # L2 normalize embeddings
    print(f"Applying L2 normalization...")
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    # Extract target variable
    y = joined_df[target_metric].values
    print(f"Target ({target_metric}) shape: {y.shape}")

    return embeddings_normalized, y


def train_and_evaluate_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list,
    n_folds: int,
    target_metric: str = 'pc1_delta'
) -> Tuple[RidgeCV, Dict]:
    """Train ridge regression with automatic alpha selection via cross-validation."""
    # Create RidgeCV model (automatically selects best alpha)
    print(f"  Running {n_folds}-fold cross-validation with {len(alphas)} alpha values...")
    ridge_cv = RidgeCV(
        alphas=alphas,
        cv=n_folds,
        scoring='neg_mean_squared_error'
    )

    # Fit the model (this performs CV and selects best alpha automatically)
    ridge_cv.fit(X, y)

    # Get the best alpha
    best_alpha = ridge_cv.alpha_
    print(f"  Best alpha selected: {best_alpha}")

    # Compute metrics on the full dataset
    y_pred = ridge_cv.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)

    metrics = {
        'target_metric': target_metric,
        'best_alpha': float(best_alpha),
        'alphas_tested': [float(a) for a in alphas],
        'n_samples': int(len(y)),
        'n_features': int(X.shape[1]),
        'cv_folds': int(n_folds),
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae)
    }

    return ridge_cv, metrics


def save_results(
    model: RidgeCV,
    metrics: Dict,
    output_dir: Path
):
    """Save model and metrics to output directory."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model to {model_path}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ridge regression to predict PC1 deltas from embeddings"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing parquet files with embeddings'
    )
    parser.add_argument(
        '--pc1-file',
        type=Path,
        required=True,
        help='Path to PC1 deltas parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for regression results'
    )
    parser.add_argument(
        '--alphas',
        type=float,
        nargs='+',
        required=True,
        help='List of alpha values for RidgeCV to test (e.g., 0.1 1.0 10.0 100.0). Best will be selected automatically.'
    )
    parser.add_argument(
        '--target-metric',
        type=str,
        default='pc1_delta',
        choices=['pc1_delta', 'next_pc1', 'prev_pc1'],
        help='Target metric to predict (default: pc1_delta)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )

    args = parser.parse_args()

    # Load data
    embeddings_df = load_embeddings(args.input_dir)
    pc1_df = load_pc1_deltas(args.pc1_file)

    # Join and prepare data
    X, y = join_and_prepare_data(embeddings_df, pc1_df, target_metric=args.target_metric)

    # Train ridge regression with automatic alpha selection
    print(f"\nTraining RidgeCV with {len(args.alphas)} alpha value(s)...")

    try:
        # Train and evaluate (RidgeCV automatically selects best alpha)
        model, metrics = train_and_evaluate_ridge(
            X,
            y,
            alphas=args.alphas,
            n_folds=args.n_folds,
            target_metric=args.target_metric
        )

        # Save results
        save_results(model, metrics, args.output_dir)

        # Print summary
        print(f"\nResults:")
        print("-" * 60)
        print(f"Target metric: {metrics['target_metric']}")
        print(f"Best alpha selected: {metrics['best_alpha']:.3f}")
        print(f"Alphas tested: {metrics['alphas_tested']}")
        print(f"R² score: {metrics['r2']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"MAE: {metrics['mae']:.3f}")
        print("-" * 60)
        print(f"\nDone! Results saved to {args.output_dir}")

    except KeyboardInterrupt:
        print("\n\nRegression interrupted by user. Partial results may have been saved.")
        sys.exit(0)


if __name__ == '__main__':
    main()
