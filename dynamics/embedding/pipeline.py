"""
Custom pipeline that integrates transcript loading with chatspace embedding pipeline.

This module provides a custom dataset loader worker that works with multiprocessing.
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

from chatspace.hf_embed import pipeline
from chatspace.hf_embed.config import SentenceTransformerConfig

from transcript_loader import load_transcript_dataset


# Sentinel for signaling process shutdown (matching chatspace convention)
_STOP = object()


def _transcript_loader_worker(
    cfg: SentenceTransformerConfig,
    input_root: Path,
    auditor_models: list[str],
    row_queue: mp.Queue[Any],
    error_queue: mp.Queue[tuple[str, Exception]],
    shutdown_event: mp.Event,
) -> None:
    """
    Background process for loading transcript rows.

    This replaces chatspace's _dataset_loader_worker to use our transcript loader.
    """
    try:
        # Load transcripts and yield rows
        dataset_iter = load_transcript_dataset(
            input_root=input_root,
            auditor_models=auditor_models,
            text_field=cfg.text_field,
            max_rows=cfg.max_rows,
        )

        for row in dataset_iter:
            if shutdown_event.is_set():
                break
            row_queue.put(row)

    except Exception as exc:
        logging.exception("Transcript loader process error")
        error_queue.put(("loader", exc))
    finally:
        row_queue.put(_STOP)


def run_transcript_embedding(
    cfg: SentenceTransformerConfig,
    input_root: Path,
    auditor_models: list[str],
) -> dict[str, Any]:
    """
    Run embedding pipeline with custom transcript loader.

    This function wraps the chatspace pipeline but uses our custom transcript
    loader worker instead of the HuggingFace dataset loader.

    Args:
        cfg: SentenceTransformer configuration
        input_root: Base path like /workspace/{model}/dynamics
        auditor_models: List of auditor model short names

    Returns:
        Dictionary with manifest, run summary, and paths
    """
    import time

    from chatspace.hf_embed.pipeline import (
        _STOP,
        _encode_and_dispatch_batch,
        _apply_stats_update,
    )
    from chatspace.hf_embed.bucketing import (
        _BucketBuffer,
        _effective_batch_size,
        _enumerate_bucket_sizes,
        _select_bucket_size,
        _token_sequence_length,
    )
    from chatspace.hf_embed.model import _load_model
    from chatspace.hf_embed.metrics import PipelineStats, StatsUpdate
    from chatspace.hf_embed.orchestrator import ProcessController, ProgressUpdate
    from chatspace.hf_embed.utils import (
        _iso_now,
        _observe_git_sha,
        _prepare_paths,
    )
    from chatspace.hf_embed.writer import _ShardWriter, write_manifest

    start_time = time.time()
    paths = _prepare_paths(
        cfg.output_root,
        cfg.model_name,
        cfg.dataset,
        cfg.subset,
        cfg.split,
        cfg.manifest_relpath,
    )

    git_sha = _observe_git_sha()
    created_at = _iso_now()
    run_id = cfg.run_id or created_at.replace(":", "").replace("-", "")

    # Use spawn method for multiprocessing (works well with CUDA)
    ctx = mp.get_context("spawn")

    # Shared state and queues
    max_bucket_batch = _effective_batch_size(cfg.bucket_min_tokens, cfg)
    row_queue: mp.Queue[Any] = ctx.Queue(maxsize=max(1, max_bucket_batch * cfg.prefetch_batches))
    batch_queue: mp.Queue[Any] = ctx.Queue(maxsize=max(1, cfg.prefetch_batches))
    progress_queue: mp.Queue[Any] = ctx.Queue(maxsize=100)
    stats_queue = ctx.Queue(maxsize=100)
    error_queue: mp.Queue[tuple[str, Exception]] = ctx.Queue()
    shard_metadata_queue: mp.Queue[Any] = ctx.Queue(maxsize=1)
    shutdown_event = ctx.Event()

    # Parent-side stats
    stats = PipelineStats()

    # Create process controller
    controller = ProcessController()
    controller.set_shutdown_event(shutdown_event)

    # Create loader process with our custom transcript loader
    loader_process = ctx.Process(
        target=_transcript_loader_worker,
        args=(cfg, input_root, auditor_models, row_queue, error_queue, shutdown_event),
        name="transcript-loader",
    )
    controller.register_process("loader", loader_process)

    # Create writer process
    writer_process = ctx.Process(
        target=_ShardWriter(
            paths["shard_dir"],
            cfg.rows_per_shard,
            batch_queue,
            error_queue,
            shard_metadata_queue,
            shutdown_event,
            progress_queue,
        ),
        name="shard-writer",
    )
    controller.register_process("writer", writer_process)

    # Start processes
    controller.start_processes()

    # Encoder runs in main process
    try:
        logging.info(f"Loading embedding model: {cfg.model_name}")
        model_runner = _load_model(cfg)

        # Warmup if compilation is enabled
        if cfg.compile_model:
            logging.info("Warming up compiled model...")
            bucket_sizes = list(_enumerate_bucket_sizes(
                cfg.bucket_min_tokens,
                cfg.bucket_max_tokens,
            ))
            model_runner.warmup(bucket_sizes, cfg)

        # Main encoding loop
        buckets = _BucketBuffer()
        rows_processed = 0
        encoder_idle = False

        logging.info("Starting encoding loop...")

        while True:
            # Check for errors
            if not error_queue.empty():
                source, exc = error_queue.get_nowait()
                raise RuntimeError(f"Error in {source} process: {exc}") from exc

            # Collect stats updates
            while not stats_queue.empty():
                try:
                    update = stats_queue.get_nowait()
                    if isinstance(update, StatsUpdate):
                        _apply_stats_update(stats, update)
                except:
                    break

            # Try to get a row
            try:
                row = row_queue.get(timeout=0.1)
            except:
                encoder_idle = True
                continue

            encoder_idle = False

            # Check for stop sentinel
            if row is _STOP:
                logging.info("Received stop sentinel, flushing buckets...")
                break

            # Check max_rows limit
            if cfg.max_rows and rows_processed >= cfg.max_rows:
                logging.info(f"Reached max_rows limit: {cfg.max_rows}")
                break

            # Tokenize and add to bucket
            text = row.get(cfg.text_field)
            if not text:
                continue

            tokens = model_runner.tokenize(text)
            seq_len = _token_sequence_length(tokens)
            bucket_size = _select_bucket_size(seq_len, cfg.bucket_min_tokens, cfg.bucket_max_tokens)

            buckets.add(bucket_size, row, tokens)

            # Check if bucket is ready to encode
            batch_size = _effective_batch_size(bucket_size, cfg)
            if buckets.size(bucket_size) >= batch_size:
                token_batch = buckets.pop_batch(bucket_size, batch_size)
                reached_limit, num_rows = _encode_and_dispatch_batch(
                    token_batch,
                    model_runner=model_runner,
                    cfg=cfg,
                    created_at=created_at,
                    run_id=run_id,
                    batch_queue=batch_queue,
                    progress_queue=progress_queue,
                    stats_queue=stats_queue,
                    rows_processed_so_far=rows_processed,
                )
                rows_processed += num_rows

                if reached_limit:
                    break

        # Flush remaining buckets
        logging.info("Flushing remaining buckets...")
        for bucket_size in buckets.nonempty_buckets():
            while buckets.size(bucket_size) > 0:
                batch_size = _effective_batch_size(bucket_size, cfg)
                token_batch = buckets.pop_batch(bucket_size, batch_size)
                reached_limit, num_rows = _encode_and_dispatch_batch(
                    token_batch,
                    model_runner=model_runner,
                    cfg=cfg,
                    created_at=created_at,
                    run_id=run_id,
                    batch_queue=batch_queue,
                    progress_queue=progress_queue,
                    stats_queue=stats_queue,
                    rows_processed_so_far=rows_processed,
                )
                rows_processed += num_rows

        # Signal writer to finish
        batch_queue.put(_STOP)

        # Wait for writer to finish
        logging.info("Waiting for writer to finish...")
        shard_metadata = shard_metadata_queue.get(timeout=300)

        # Shutdown
        controller.shutdown(timeout=10)

        # Generate manifest
        duration = time.time() - start_time

        manifest = write_manifest(
            paths=paths,
            cfg=cfg,
            shard_metadata=shard_metadata,
            stats=stats,
            created_at=created_at,
            run_id=run_id,
            git_sha=git_sha,
            duration=duration,
        )

        logging.info(f"Pipeline completed in {duration:.1f}s")
        logging.info(f"Manifest: {paths['manifest_path']}")

        return {
            "manifest": manifest,
            "manifest_path": paths["manifest_path"],
            "run_path": paths["run_path"],
            "shard_dir": paths["shard_dir"],
            "rows_total": stats.rows_total,
            "rows_skipped": stats.rows_skipped,
            "num_shards": len(shard_metadata),
            "duration_seconds": duration,
        }

    except Exception as exc:
        logging.exception("Encoder process error")
        controller.shutdown(timeout=5)
        raise
    finally:
        controller.cleanup()
