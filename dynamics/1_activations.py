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
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Add utils to path for imports
sys.path.append('.')
sys.path.append('..')

from utils.probing_utils import load_model, process_batch_conversations, process_batch_conversations_no_code
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
        interval: Optional[Tuple[int, int]] = None,
        no_code: bool = False,
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
            interval: Optional tuple (start, end) for processing files in range [start:end]
            no_code: Exclude code blocks from activation averaging
        """
        self.model_name = model_name
        self.chat_model_name = chat_model_name
        self.target_dir = Path(target_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.target_files = target_files
        self.interval = interval
        self.no_code = no_code

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

    def load_tokenizer(self):
        """Load only the tokenizer (for bucketing without loading the full model)."""
        if self.tokenizer is None:
            tokenizer_name = self.chat_model_name if self.chat_model_name else self.model_name
            logger.info(f"Loading tokenizer from: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            logger.info("Tokenizer loaded successfully")

    def load_model(self, device=None):
        """Load model and tokenizer."""
        if self.model is None:
            # Load tokenizer first if not already loaded
            if self.tokenizer is None:
                self.load_tokenizer()
            
            if self.chat_model_name is not None:
                # Load model from activation model (tokenizer already loaded)
                logger.info(f"Loading model from: {self.model_name}")
                self.model, _ = load_model(self.model_name, device=device)
            else:
                # Load model (tokenizer already loaded)
                logger.info(f"Loading model: {self.model_name}")
                self.model, _ = load_model(self.model_name, device=device)
            logger.info("Model loaded successfully")

    def close_model(self):
        """Clean up model resources (keeps tokenizer loaded)."""
        if self.model is not None:
            logger.info("Cleaning up model resources")
            del self.model
            self.model = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Force garbage collection
            gc.collect()
    
    def close_all(self):
        """Clean up all resources including tokenizer."""
        self.close_model()
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

    def get_transcript_files(self) -> List[Path]:
        """Get JSON transcript files in the target directory, optionally filtered by target_files and interval."""
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
            json_files = sorted(json_files)
        else:
            # Process all JSON files in directory
            json_files = list(self.target_dir.glob("*.json"))
            logger.info(f"Found {len(json_files)} JSON files in {self.target_dir}")
            json_files = sorted(json_files)

        # Apply interval filtering if specified
        if self.interval is not None:
            start, end = self.interval
            original_count = len(json_files)

            # Sort files alphabetically for consistent interval behavior
            json_files = sorted(json_files, key=lambda x: x.name)

            # Apply interval slicing [start:end] (start inclusive, end exclusive)
            if start < 0 or start >= len(json_files):
                logger.warning(f"Interval start {start} is out of range [0, {len(json_files)})")
                start = max(0, min(start, len(json_files)))

            if end < start or end > len(json_files):
                logger.warning(f"Interval end {end} is out of range [{start}, {len(json_files)}]")
                end = max(start, min(end, len(json_files)))

            json_files = json_files[start:end]
            logger.info(f"Applied interval [{start}:{end}] - filtered from {original_count} to {len(json_files)} files")

            if json_files:
                logger.info(f"Processing files from '{json_files[0].name}' to '{json_files[-1].name}'")

        return json_files

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
            if self.no_code:
                batch_activations = process_batch_conversations_no_code(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    conversations=conversations,
                    max_length=self.max_length,
                    **self.chat_kwargs
                )
                logger.info("Using no-code processing (excluding code blocks from averaging)")
            else:
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

        if not files_to_process:
            logger.info("No files to process")
            return

        # Sort files by conversation length for efficient batching (preserving this critical optimization)
        logger.info("Preparing files for processing with length bucketing...")
        self.load_tokenizer()  # Only need tokenizer for length bucketing
        try:
            files_to_process = self.bucket_conversations_by_length(files_to_process)
        finally:
            # Tokenizer is lightweight, keep it loaded for later use
            pass

        # Check if we can use multi-GPU processing
        # NOTE: For very large models (like 70B+), multi-GPU file distribution doesn't work well
        # because each process tries to load a full copy of the model on each GPU.
        # For such models, use single-process mode with device_map="auto" to shard across GPUs.
        model_is_large = is_large_model(self.model_name)
        if model_is_large:
            logger.info(f"Model {self.model_name} detected as large model (70B+) - using single-process mode with GPU sharding")
        
        use_multi_gpu_file_processing = not model_is_large
        
        if use_multi_gpu_file_processing and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"Multi-GPU processing available with {torch.cuda.device_count()} GPUs")

            # Create args object for multi-GPU processing
            args = ExtractorArgs(
                model_name=self.model_name,
                target_dir=str(self.target_dir),
                output_dir=str(self.output_dir),
                batch_size=self.batch_size,
                max_length=self.max_length,
                chat_model=self.chat_model_name,
                thinking=self.chat_kwargs.get('enable_thinking', False),
                no_code=self.no_code,
                interval=self.interval
            )

            # Try multi-GPU processing
            if run_multi_gpu_processing(args, files_to_process):
                logger.info("Multi-GPU processing completed successfully")
                return
            else:
                logger.info("Falling back to single-GPU processing")

        # Single-GPU processing (original logic)
        try:
            # Load model once
            self.load_model()

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


def is_large_model(model_name: str) -> bool:
    """
    Detect if a model is too large to fit on a single GPU.
    
    Large models (70B+) should use device_map="auto" sharding instead of 
    multi-GPU file distribution where each worker loads a full model copy.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        True if the model is large (70B+), False otherwise
    """
    model_lower = model_name.lower()
    
    # Check for explicit size indicators in model name
    large_size_patterns = [
        '70b', '72b', '90b', '100b', '110b', '120b', '140b', '180b', '200b',
        '300b', '400b', '500b', '600b', '700b', '800b', '900b', '1000b',
        '405b',  # Llama 3.1 405B
    ]
    
    for pattern in large_size_patterns:
        if pattern in model_lower:
            return True
    
    return False


class ExtractorArgs:
    """Args object for multi-GPU processing."""
    def __init__(self, model_name, target_dir, output_dir, batch_size, max_length, chat_model, thinking, no_code=False, interval=None):
        self.model_name = model_name
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.chat_model = chat_model
        self.thinking = thinking
        self.no_code = no_code
        self.interval = interval


def process_files_on_gpu(gpu_id, files, args):
    """Process a subset of transcript files on a specific GPU."""
    # Set the GPU for this process
    torch.cuda.set_device(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Set up logging for this process
    logger = logging.getLogger(f"GPU-{gpu_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - GPU-{gpu_id} - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info(f"Starting processing on GPU {gpu_id} with {len(files)} files")

    try:
        # Create extractor for this GPU
        extractor = DynamicsActivationExtractor(
            model_name=args.model_name,
            target_dir=str(Path(args.target_dir)),  # Convert to string for compatibility
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            chat_model_name=args.chat_model,
            thinking=args.thinking,
            target_files=[f.stem for f in files],  # Pass just the filenames without .json
            interval=None,  # Interval filtering already applied before GPU distribution
            no_code=args.no_code
        )

        # Load model on this specific GPU
        extractor.load_model(device=f"cuda:{gpu_id}")

        # Process assigned files
        completed_count = 0
        failed_count = 0

        with tqdm(files, desc=f"GPU-{gpu_id} files", unit="file", position=gpu_id) as file_pbar:
            for file_path in file_pbar:
                file_pbar.set_postfix(file=file_path.name[:20], refresh=True)

                try:
                    # Process this file by calling the batch processing with single file
                    batch_completed, batch_failed = extractor.process_transcript_batch([file_path])
                    completed_count += batch_completed
                    failed_count += batch_failed

                    if batch_completed > 0:
                        file_pbar.set_postfix(file=file_path.name[:20], status="✓", refresh=True)
                    else:
                        file_pbar.set_postfix(file=file_path.name[:20], status="✗", refresh=True)

                except Exception as e:
                    failed_count += 1
                    file_pbar.set_postfix(file=file_path.name[:20], status="✗", refresh=True)
                    logger.error(f"Exception processing file {file_path.name}: {e}")

        logger.info(f"GPU {gpu_id} completed: {completed_count} successful, {failed_count} failed")

    except Exception as e:
        logger.error(f"Fatal error on GPU {gpu_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    finally:
        # Cleanup
        if 'extractor' in locals():
            extractor.close_model()
        logger.info(f"GPU {gpu_id} cleanup completed")


def run_multi_gpu_processing(args, files_to_process):
    """Run multi-GPU processing with file distribution."""
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        logger.info(f"Only {num_gpus} GPU(s) available, using single-GPU processing")
        return False

    logger.info(f"Using {num_gpus} GPUs for processing {len(files_to_process)} files")

    # Distribute files across GPUs
    files_per_gpu = len(files_to_process) // num_gpus
    remainder = len(files_to_process) % num_gpus

    file_chunks = []
    start_idx = 0

    for gpu_id in range(num_gpus):
        # Give extra files to first few GPUs if there's a remainder
        chunk_size = files_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + chunk_size

        chunk = files_to_process[start_idx:end_idx]
        file_chunks.append((gpu_id, chunk))

        if chunk:
            logger.info(f"GPU {gpu_id}: {len(chunk)} files ({chunk[0].name} to {chunk[-1].name})")
        start_idx = end_idx

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Launch processes
    processes = []
    for gpu_id, chunk in file_chunks:
        if chunk:  # Only launch if there are files to process
            p = mp.Process(
                target=process_files_on_gpu,
                args=(gpu_id, chunk, args)
            )
            p.start()
            processes.append(p)

    # Wait for all processes to complete
    logger.info(f"Launched {len(processes)} GPU processes")
    for p in processes:
        p.join()

    logger.info("Multi-GPU processing completed!")
    return True


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

    # Exclude code blocks from activation averaging
    python dynamics/activations.py \\
        --model-name Qwen/Qwen3-32B \\
        --target-dir dynamics/results/qwen-3-32b/transcripts \\
        --output-dir dynamics/results/qwen-3-32b/activations \\
        --no-code
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
    parser.add_argument('--max-length', type=int, default=40960,
                       help='Maximum sequence length (default: 40960)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--files', nargs='*', default=None,
                       help='Specific files to process (without .json extension). '
                            'Example: --files writing_persona9_topic1 philosophy_persona15_topic3')

    parser.add_argument(
        "--interval",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Run activations in range [START:END] (start inclusive, end exclusive)"
    )

    parser.add_argument(
        "--thinking",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=False,
        help="Enable thinking mode for chat templates (default: False)"
    )

    parser.add_argument(
        "--no-code",
        action="store_true",
        default=False,
        help="Exclude code blocks (` and ```) from activation averaging (default: False)"
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
    logger.info(f"  No-code processing: {args.no_code}")
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
            target_files=args.files,
            interval=tuple(args.interval) if args.interval else None,
            no_code=args.no_code
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