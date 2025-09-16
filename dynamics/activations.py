#!/usr/bin/env python3
"""
Activation extraction script for dynamics conversation transcripts.

This script processes JSON transcript files from dynamics experiments and extracts
mean activations for each conversation turn. The output preserves the original
JSON structure while adding an 'activations' tensor.

Usage:
    uv run dynamics/activations.py \
        --model-name Qwen/Qwen3-32B \
        --target-dir /root/git/persona-subspace/dynamics/results/qwen-3-32b/transcripts \
        --output-dir /root/git/persona-subspace/dynamics/results/qwen-3-32b/activations
"""

import argparse
import gc
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from tqdm import tqdm

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.probing_utils import load_model, process_batch_conversations
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




class DynamicsActivationExtractor:
    """Extractor for dynamics conversation transcript activations."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-32B",
        target_dir: str = "/root/git/persona-subspace/dynamics/results/qwen-3-32b/transcripts",
        output_dir: str = "/root/git/persona-subspace/dynamics/results/qwen-3-32b/activations",
        batch_size: int = 4,  # Process multiple conversations per batch
        max_length: int = 4096,
        chat_model_name: Optional[str] = None,
        thinking: bool = False,
        target_files: Optional[List[str]] = None,
    ):
        """
        Initialize the dynamics activation extractor.

        Args:
            model_name: HuggingFace model identifier
            target_dir: Directory containing JSON transcript files
            output_dir: Directory to save enhanced .pt files
            batch_size: Batch size for processing conversations
            max_length: Maximum sequence length
            chat_model_name: Optional HuggingFace model identifier for tokenizer
            thinking: Enable thinking mode for chat templates
            target_files: Optional list of specific filenames to process (without .json extension)
        """
        self.model_name = model_name
        self.chat_model_name = chat_model_name
        self.target_dir = Path(target_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.target_files = target_files

        # Chat template configuration
        self.chat_kwargs = {}
        if thinking is not None and model_name:
            # Only Qwen models support enable_thinking parameter
            if 'qwen' in model_name.lower():
                self.chat_kwargs['enable_thinking'] = thinking

        # Model and tokenizer (loaded once)
        self.model = None
        self.tokenizer = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized DynamicsActivationExtractor with model: {model_name}")
        logger.info(f"Target directory: {self.target_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_model(self, device=None):
        """Load model and tokenizer."""
        if self.model is None:
            if self.chat_model_name is not None:
                # Load tokenizer from chat model, model from activation model
                logger.info(f"Loading model from: {self.model_name}")
                logger.info(f"Loading tokenizer from: {self.chat_model_name}")
                self.model, _ = load_model(self.model_name, device=device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.chat_model_name)
                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
            else:
                # Load both from same model
                logger.info(f"Loading model: {self.model_name}")
                self.model, self.tokenizer = load_model(self.model_name, device=device)
            logger.info("Model loaded successfully")

    def close_model(self):
        """Clean up model resources."""
        if self.model is not None:
            logger.info("Cleaning up model resources")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Force garbage collection
            gc.collect()

    def get_transcript_files(self) -> List[Path]:
        """Get JSON transcript files in the target directory, optionally filtered by target_files."""
        if self.target_files is not None:
            # Process specific files
            json_files = []
            missing_files = []

            for filename in self.target_files:
                # Add .json extension if not present
                if not filename.endswith('.json'):
                    filename = f"{filename}.json"

                file_path = self.target_dir / filename
                if file_path.exists():
                    json_files.append(file_path)
                else:
                    missing_files.append(filename)

            if missing_files:
                logger.error(f"The following files were not found in {self.target_dir}:")
                for missing in missing_files:
                    logger.error(f"  - {missing}")
                raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

            logger.info(f"Processing {len(json_files)} specific files: {[f.stem for f in json_files]}")
            return sorted(json_files)
        else:
            # Process all JSON files in directory
            json_files = list(self.target_dir.glob("*.json"))
            logger.info(f"Found {len(json_files)} JSON files in {self.target_dir}")
            return sorted(json_files)

    def bucket_conversations_by_length(self, json_files: List[Path]) -> List[Path]:
        """
        Sort conversation files by tokenized length for efficient batching.

        Args:
            json_files: List of JSON file paths

        Returns:
            List of file paths sorted by conversation length (shortest first)
        """
        logger.info("Bucketing conversations by length...")

        file_lengths = []
        for json_file in json_files:
            try:
                transcript_data = self.load_transcript(json_file)
                if transcript_data is None or 'conversation' not in transcript_data:
                    # Put invalid files at the end with a large length
                    file_lengths.append((json_file, 999999))
                    continue

                conversation = transcript_data['conversation']
                if not conversation:
                    file_lengths.append((json_file, 0))
                    continue

                # Calculate tokenized length using chat template
                try:
                    tokenized = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        add_generation_prompt=False,
                        **self.chat_kwargs
                    )
                    length = len(tokenized)
                except Exception as e:
                    logger.warning(f"Error tokenizing {json_file.name}: {e}")
                    length = 999999  # Put at end if tokenization fails

                file_lengths.append((json_file, length))

            except Exception as e:
                logger.warning(f"Error processing {json_file.name} for length bucketing: {e}")
                file_lengths.append((json_file, 999999))

        # Sort by length (shortest first)
        file_lengths.sort(key=lambda x: x[1])
        sorted_files = [f for f, _ in file_lengths]

        # Log length distribution
        lengths = [length for _, length in file_lengths if length < 999999]
        if lengths:
            logger.info(f"Length bucketing complete: min={min(lengths)}, max={max(lengths)}, "
                       f"mean={sum(lengths)/len(lengths):.1f} tokens")

        return sorted_files

    def load_transcript(self, json_file: Path) -> Dict:
        """Load a transcript JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            return None

    def process_transcript_batch(self, json_files: List[Path]) -> Tuple[int, int]:
        """Process a batch of transcript files and save enhanced versions.

        Returns:
            tuple: (completed_count, failed_count)
        """
        try:
            logger.info(f"Processing batch of {len(json_files)} files")

            # Load all transcripts
            conversations = []
            transcript_data_list = []
            valid_files = []

            for json_file in json_files:
                transcript_data = self.load_transcript(json_file)
                if transcript_data is None:
                    logger.warning(f"Failed to load {json_file.name}")
                    continue

                conversation = transcript_data.get('conversation', [])
                if not conversation:
                    logger.warning(f"No conversation found in {json_file.name}")
                    continue

                conversations.append(conversation)
                transcript_data_list.append(transcript_data)
                valid_files.append(json_file)

            if not conversations:
                logger.error("No valid conversations in batch")
                return 0, len(json_files)

            logger.info(f"Processing {len(conversations)} valid conversations")

            # Calculate batch statistics before processing
            seq_lengths = []
            turn_counts = []
            for conversation in conversations:
                try:
                    # Calculate tokenized length
                    tokenized = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=True,
                        add_generation_prompt=False,
                        **self.chat_kwargs
                    )
                    seq_lengths.append(len(tokenized))
                    turn_counts.append(len(conversation))
                except Exception as e:
                    logger.warning(f"Error calculating stats for conversation: {e}")
                    seq_lengths.append(0)
                    turn_counts.append(0)

            # Log batch statistics
            if seq_lengths:
                avg_seq_len = sum(seq_lengths) / len(seq_lengths)
                max_seq_len = max(seq_lengths)
                avg_turns = sum(turn_counts) / len(turn_counts)
                logger.info(f"Batch stats: avg_seq_len={avg_seq_len:.1f}, max_seq_len={max_seq_len}, "
                           f"avg_turns={avg_turns:.1f}, effective_batch_size={len(conversations)}")

            # Extract batch activations using the new span-based approach
            batch_activations = process_batch_conversations(
                model=self.model,
                tokenizer=self.tokenizer,
                conversations=conversations,
                max_length=self.max_length,
                **self.chat_kwargs
            )

            logger.info(f"Extracted activations for {len(batch_activations)} conversations")

            # Save enhanced data for each conversation
            completed_count = 0
            failed_count = 0

            for i, (json_file, transcript_data, activations) in enumerate(zip(valid_files, transcript_data_list, batch_activations)):
                try:
                    if activations.numel() == 0:
                        logger.warning(f"Empty activations for {json_file.name}")
                        failed_count += 1
                        continue

                    logger.info(f"Activations for {json_file.name}: shape {activations.shape}")

                    # Create enhanced data with activations
                    enhanced_data = transcript_data.copy()
                    enhanced_data['activations'] = activations

                    # Save to output directory
                    output_file = self.output_dir / f"{json_file.stem}.pt"
                    torch.save(enhanced_data, output_file)

                    # Log file size
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    logger.info(f"Saved enhanced transcript to {output_file} ({file_size_mb:.1f} MB)")

                    completed_count += 1

                except Exception as e:
                    logger.error(f"Error saving {json_file.name}: {e}")
                    failed_count += 1

            return completed_count, failed_count

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0, len(json_files)

    def process_all_transcripts(self, skip_existing: bool = True):
        """Process all transcript files in the target directory using batching."""
        try:
            # Load model once
            self.load_model()

            # Get transcript files
            transcript_files = self.get_transcript_files()

            if not transcript_files:
                logger.error("No transcript files found")
                return

            # Filter existing files if needed
            if skip_existing:
                files_to_process = []
                for json_file in transcript_files:
                    output_file = self.output_dir / f"{json_file.stem}.pt"
                    if not output_file.exists():
                        files_to_process.append(json_file)
                    else:
                        logger.info(f"Skipping {json_file.name} (output exists)")
            else:
                files_to_process = transcript_files

            # Sort files by conversation length for efficient batching
            files_to_process = self.bucket_conversations_by_length(files_to_process)

            logger.info(f"Processing {len(files_to_process)} transcript files with batch size {self.batch_size}")

            # Process in batches
            completed_count = 0
            failed_count = 0

            # Create batches
            batches = [files_to_process[i:i + self.batch_size] for i in range(0, len(files_to_process), self.batch_size)]

            with tqdm(batches, desc="Processing batches", unit="batch") as pbar:
                for batch_idx, batch_files in enumerate(pbar):
                    pbar.set_postfix(
                        batch=f"{batch_idx+1}/{len(batches)}",
                        files=f"{len(batch_files)}",
                        refresh=True
                    )

                    try:
                        batch_completed, batch_failed = self.process_transcript_batch(batch_files)
                        completed_count += batch_completed
                        failed_count += batch_failed

                        pbar.set_postfix(
                            batch=f"{batch_idx+1}/{len(batches)}",
                            completed=completed_count,
                            failed=failed_count,
                            refresh=True
                        )

                    except Exception as e:
                        failed_count += len(batch_files)
                        logger.error(f"Exception processing batch {batch_idx+1}: {e}")
                        pbar.set_postfix(
                            batch=f"{batch_idx+1}/{len(batches)}",
                            status="✗",
                            refresh=True
                        )

                    # Periodic cleanup after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            logger.info(f"Processing summary: {completed_count} completed, {failed_count} failed")

        finally:
            # Final cleanup
            self.close_model()
            logger.info("Final cleanup completed")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Extract per-turn activations from dynamics conversation transcripts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - process all files
    python dynamics/activations.py \\
        --model-name Qwen/Qwen3-32B \\
        --target-dir dynamics/results/qwen-3-32b/transcripts \\
        --output-dir dynamics/results/qwen-3-32b/activations

    # Process specific files only
    python dynamics/activations.py \\
        --model-name Qwen/Qwen3-32B \\
        --target-dir dynamics/results/qwen-3-32b/transcripts \\
        --output-dir dynamics/results/qwen-3-32b/activations \\
        --files writing_persona9_topic1 philosophy_persona15_topic3

    # Custom batch size and thinking disabled
    python dynamics/activations.py \\
        --model-name Qwen/Qwen3-32B \\
        --target-dir dynamics/results/qwen-3-32b/transcripts \\
        --output-dir dynamics/results/qwen-3-32b/activations \\
        --batch-size 8 \\
        --no-thinking
        """
    )

    # Model configuration
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-32B',
                       help='HuggingFace model name')
    parser.add_argument('--chat-model', type=str, default=None,
                       help='Optional HuggingFace model name for tokenizer')
    parser.add_argument('--target-dir', type=str,
                       default='/root/git/persona-subspace/dynamics/results/qwen-3-32b/transcripts',
                       help='Directory containing transcript JSON files')
    parser.add_argument('--output-dir', type=str,
                       default='/root/git/persona-subspace/dynamics/results/qwen-3-32b/activations',
                       help='Output directory for enhanced .pt files')

    # Processing options
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for processing conversations (default: 4)')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process all files, even if output files exist')
    parser.add_argument('--max-length', type=int, default=4096,
                       help='Maximum sequence length (default: 4096)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--files', nargs='*', default=None,
                       help='Specific files to process (without .json extension). '
                            'Example: --files writing_persona9_topic1 philosophy_persona15_topic3')

    parser.add_argument(
        "--thinking",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=False,
        help="Enable thinking mode for chat templates (default: False)"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Target directory: {args.target_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Thinking enabled: {args.thinking}")
    if args.files:
        logger.info(f"  Specific files: {args.files}")
    else:
        logger.info(f"  Processing: All files in directory")

    try:
        # Create extractor
        extractor = DynamicsActivationExtractor(
            model_name=args.model_name,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            chat_model_name=args.chat_model,
            thinking=args.thinking,
            target_files=args.files
        )

        # Process all transcripts
        extractor.process_all_transcripts(
            skip_existing=not args.no_skip_existing
        )

        logger.info("Dynamics activation extraction completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())