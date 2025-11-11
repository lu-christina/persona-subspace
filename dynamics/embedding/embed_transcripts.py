#!/usr/bin/env python3
"""
Embed user turns from transcript JSON files using the chatspace embedding pipeline.

Usage:
    python embed_transcripts.py \\
        --input-root /workspace/qwen-3-32b/dynamics \\
        --auditor-models gpt-5,sonnet-4.5 \\
        --embedding-model Qwen/Qwen3-Embedding-0.6B \\
        --device cuda \\
        --dtype bfloat16
"""

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

# Import chatspace components
from chatspace.hf_embed.config import SentenceTransformerConfig
from chatspace.hf_embed.pipeline import run_sentence_transformer

# Import our custom transcript loader
from transcript_loader import load_transcript_dataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Embed user turns from transcript JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output paths
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Base path like /workspace/qwen-3-32b/dynamics",
    )
    parser.add_argument(
        "--auditor-models",
        type=str,
        default="gpt-5,sonnet-4.5,kimi-k2",
        help="Comma-separated list of auditor model short names",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: {input-root}/embedding/user_turns)",
    )

    # Embedding model configuration
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer model name from HuggingFace",
    )

    # Pipeline configuration
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=8192,
        help="Number of rows per Parquet shard",
    )
    parser.add_argument(
        "--tokens-per-batch",
        type=int,
        default=131072,
        help="Total tokens per batch (adaptive batching)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Maximum number of rows to process (for testing)",
    )

    # Hardware configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for embeddings",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for faster inference",
    )

    # Misc
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of field containing text to embed",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # Validate input path
    if not args.input_root.exists():
        logging.error(f"Input root does not exist: {args.input_root}")
        sys.exit(1)

    # Parse auditor models
    auditor_models = [name.strip() for name in args.auditor_models.split(",")]
    logging.info(f"Processing auditor models: {auditor_models}")

    # Extract short model name from input_root path
    # E.g., /workspace/qwen-3-32b/dynamics -> qwen-3-32b
    short_model = args.input_root.parent.name

    # Determine output directory
    # Chatspace flat mode (dataset="."): {output_root}/{split}/ and {output_root}/indexes/
    # Result:
    #   - Shards: /workspace/qwen-3-32b/dynamics/embedding/user_turns/
    #   - Indexes: /workspace/qwen-3-32b/dynamics/embedding/indexes/
    # We set output_root to: /workspace/qwen-3-32b/dynamics/embedding
    if args.output_dir:
        # User specified custom output
        output_root = args.output_dir.parent
        split_name = args.output_dir.name
    else:
        # Default structure
        output_root = args.input_root / "embedding"
        split_name = "user_turns"

    logging.info(f"Output shards: {output_root}/{split_name}/")
    logging.info(f"Output indexes: {output_root}/indexes/")
    logging.info(f"Loading transcripts from: {args.input_root}")

    # Create a factory function using partial - this is picklable for multiprocessing
    dataset_factory = partial(
        load_transcript_dataset,
        input_root=args.input_root,
        auditor_models=auditor_models,
        text_field=args.text_field,
        max_rows=args.max_rows,
    )

    # Create chatspace config with custom dataset factory
    config = SentenceTransformerConfig(
        # Model configuration - use full model name for loading
        model_name=args.embedding_model,

        # Path configuration - use simple names to minimize directory nesting
        # Result: {output_root}/embeddings/default/./user_turns/
        dataset=".",  # Minimal dataset name
        output_root=output_root,
        split=split_name,

        # Custom dataset factory (returns an iterator when called)
        custom_dataset_factory=dataset_factory,

        # Text field
        text_field=args.text_field,

        # Pipeline configuration
        rows_per_shard=args.rows_per_shard,
        tokens_per_batch=args.tokens_per_batch,
        max_rows=args.max_rows,

        # Hardware configuration
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile_model,

        # Disable features we don't need
        extract_first_assistant=False,
        subset=None,
    )

    try:
        # Run the embedding pipeline
        logging.info(f"Starting embedding pipeline for model: {short_model}")
        logging.info(f"Embedding model: {args.embedding_model}")

        result = run_sentence_transformer(config)

        logging.info("Pipeline completed successfully!")
        logging.info(f"Manifest: {result.get('manifest_path')}")
        logging.info(f"Rows processed: {result.get('rows_total', 0)}")
        logging.info(f"Shards created: {result.get('num_shards', 0)}")

        return 0

    except Exception as e:
        logging.exception("Pipeline failed with error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
