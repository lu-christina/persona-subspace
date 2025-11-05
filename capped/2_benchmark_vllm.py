#!/usr/bin/env python3
"""
Benchmark Evaluation with vLLM Steering (Fast on Big GPUs)

This script evaluates (optionally steered) vLLM models on benchmark suites using
lm-evaluation-harness. It uses the chatspace VLLMSteerModel for efficient steering
with vLLM's optimized inference engine.

Key differences from 2_benchmark_eval.py:
- Uses vLLM instead of HuggingFace transformers
- Applies steering via VLLMSteerModel.steering() context manager
- Custom lm-eval wrapper (VLLMSteeringLM) to bridge vLLM with lm-eval-harness
- vLLM-specific optimizations (tensor parallelism, memory utilization)
"""

import argparse
from datetime import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch

# -----------------------------
# Env toggles before imports
# -----------------------------
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -----------------------------
# Project paths
# -----------------------------
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import chatspace vLLM steering components
from chatspace.generation.vllm_steer_model import (
    VLLMSteerModel,
    VLLMSteeringConfig,
)
from chatspace.generation.compat import load_legacy_role_trait_config, LegacyExperiment

# -----------------------------
# lm-eval imports
# -----------------------------
import lm_eval
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import Collator, configure_pad_token
from vllm import SamplingParams


# -----------------------------
# VLLMSteeringLM wrapper for lm-eval
# -----------------------------

class VLLMSteeringLM(TemplateLM):
    """
    Wrapper around VLLMSteerModel to make it compatible with lm-eval-harness.

    This class bridges the VLLMSteerModel interface with the TemplateLM interface
    expected by lm-eval, implementing the required methods for evaluation.
    """

    def __init__(self, vllm_steer_model: VLLMSteerModel, batch_size: int | str = 1):
        super().__init__()
        self.model = vllm_steer_model
        self.tokenizer = vllm_steer_model.tokenizer
        self.tokenizer = configure_pad_token(self.tokenizer)
        # Handle "auto" batch size
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            self._batch_size = "auto"
        else:
            self._batch_size = int(batch_size)
        self._max_gen_toks = None

    def _ensure_engine_initialized(self):
        """Synchronous wrapper to ensure async engine is initialized."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we can't use run()
            raise RuntimeError("Cannot initialize engine from within running event loop")
        except RuntimeError:
            # No running loop, use asyncio.run() which handles loop creation
            asyncio.run(self.model._ensure_engine_initialized())

    def _apply_steering_sync(self, steering_spec):
        """Synchronously apply steering spec."""
        import asyncio
        asyncio.run(self.model.push_steering_spec(steering_spec))

    def _remove_steering_sync(self):
        """Synchronously remove steering spec."""
        import asyncio
        asyncio.run(self.model.pop_steering_spec())

    def _generate_sync(self, prompts=None, prompt_token_ids=None, sampling_params=None, use_tqdm=False):
        """Synchronous wrapper for async generate.

        Args:
            prompts: String prompt(s) or None if using prompt_token_ids
            prompt_token_ids: Token ID list(s) or None if using prompts
            sampling_params: SamplingParams instance
            use_tqdm: Whether to show progress bar
        """
        import asyncio
        import uuid
        from vllm.inputs import TokensPrompt
        from tqdm import tqdm

        async def _async_generate():
            await self.model._ensure_engine_initialized()
            results = []

            # Determine input type and normalize to list
            if prompt_token_ids is not None:
                # Using token IDs
                if isinstance(prompt_token_ids[0], int):
                    # Single list of token IDs
                    inputs_list = [TokensPrompt(prompt_token_ids=prompt_token_ids)]
                else:
                    # List of lists of token IDs
                    inputs_list = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
            elif prompts is not None:
                # Using string prompts
                if isinstance(prompts, str):
                    inputs_list = [prompts]
                else:
                    inputs_list = prompts
            else:
                raise ValueError("Must provide either prompts or prompt_token_ids")

            # Wrap with tqdm if requested
            iterator = tqdm(inputs_list, desc="Generating") if use_tqdm else inputs_list

            for inp in iterator:
                request_id = f"gen_{uuid.uuid4().hex}"
                final_output = None
                async for output in self.model._engine.generate(inp, sampling_params, request_id=request_id):
                    final_output = output
                if final_output is None:
                    raise RuntimeError(f"No output for prompt")
                results.append(final_output)
            return results

        return asyncio.run(_async_generate())

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Try config first, otherwise load from transformers
        if self.model.cfg.max_model_len is not None:
            return self.model.cfg.max_model_len
        # Load from transformers config as fallback
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model.cfg.model_name, trust_remote_code=True)
        return getattr(config, 'max_position_embeddings', 4096)

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def tokenizer_name(self) -> str:
        """Return the tokenizer name for cache fingerprinting."""
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(self, string: str, add_special_tokens: bool = False) -> List[int]:
        """Tokenize a string."""
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode tokens to string."""
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts (pre-tokenized).

        This is the method required by TemplateLM. It receives requests in the format:
        [((context, continuation), context_tokens, continuation_tokens), ...]

        For vLLM, we use prompt_logprobs to get token-level log probabilities.
        """
        results = []

        # Handle batch_size auto - process all at once
        batch_size = len(requests) if self.batch_size == "auto" else int(self.batch_size)

        # Process in batches
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]

            # Extract tokenized data from requests
            prompt_token_ids = []
            ctx_cont_pairs = []

            for req in batch:
                # req format: ((context, continuation), context_tokens, continuation_tokens)
                _, ctx_tokens, cont_tokens = req
                prompt_token_ids.append(ctx_tokens + cont_tokens)
                ctx_cont_pairs.append((ctx_tokens, cont_tokens))

            # Use vLLM to get logprobs
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                prompt_logprobs=1,
                logprobs=1,
            )

            outputs = self._generate_sync(
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )

            for output, (ctx_tokens, cont_tokens) in zip(outputs, ctx_cont_pairs):
                # Get logprobs for continuation tokens
                prompt_logprobs = output.prompt_logprobs
                if prompt_logprobs is None:
                    results.append((0.0, False))
                    continue

                # Sum logprobs over continuation tokens
                cont_start = len(ctx_tokens)
                logprob_sum = 0.0
                is_greedy = True

                for idx in range(cont_start, len(ctx_tokens) + len(cont_tokens)):
                    if idx >= len(prompt_logprobs) or prompt_logprobs[idx] is None:
                        continue

                    token_id = (ctx_tokens + cont_tokens)[idx]
                    token_logprobs = prompt_logprobs[idx]

                    if token_id in token_logprobs:
                        logprob_sum += token_logprobs[token_id].logprob

                        # Check if this token is the greedy choice
                        max_logprob_token = max(token_logprobs.items(), key=lambda x: x[1].logprob)[0]
                        if token_id != max_logprob_token:
                            is_greedy = False
                    else:
                        is_greedy = False

                results.append((logprob_sum, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """
        Compute rolling log-likelihood for perplexity evaluation.
        """
        results = []

        for req in requests:
            string = req.args[0]
            tokens = self.tok_encode(string)

            # Use vLLM to get logprobs for the entire sequence
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                prompt_logprobs=1,
            )

            outputs = self._generate_sync(
                prompt_token_ids=[tokens],
                sampling_params=sampling_params,
            )

            output = outputs[0]
            prompt_logprobs = output.prompt_logprobs

            if prompt_logprobs is None:
                results.append(0.0)
                continue

            # Sum all token logprobs
            logprob_sum = 0.0
            for idx, token_logprobs in enumerate(prompt_logprobs):
                if token_logprobs is None or idx >= len(tokens):
                    continue
                token_id = tokens[idx]
                if token_id in token_logprobs:
                    logprob_sum += token_logprobs[token_id].logprob

            results.append(logprob_sum)

        return results

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        Generate text until stopping criteria are met.
        """
        results = []

        # Handle batch_size auto - process all at once
        batch_size = len(requests) if self.batch_size == "auto" else int(self.batch_size)

        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]

            contexts = []
            gen_kwargs_list = []

            for req in batch:
                context, gen_kwargs = req.args
                contexts.append(context)
                gen_kwargs_list.append(gen_kwargs)

            # Prepare sampling params from first request (assume uniform)
            gen_kwargs = gen_kwargs_list[0] if gen_kwargs_list else {}
            max_tokens = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            stop_sequences = gen_kwargs.get("until", [])

            # Filter out empty strings from stop sequences (used to signal EOS-only for eq_bench)
            stop_sequences = [s for s in stop_sequences if s] if stop_sequences else []

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences if stop_sequences else None,
            )

            # Generate using the underlying vLLM engine directly for better control
            outputs = self._generate_sync(
                prompts=contexts,
                sampling_params=sampling_params,
            )

            # Extract text from vLLM outputs
            texts = [output.outputs[0].text for output in outputs]
            results.extend(texts)

        return results

    def apply_chat_template(self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Apply chat template using the tokenizer."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


# -----------------------------
# Config loading (uses compat module for legacy format)
# -----------------------------
# The compat module handles loading legacy .pt configs and converting them
# to SteeringSpec objects with projection caps (no additive steering)


# -----------------------------
# Result save/merge helpers (same as original)
# -----------------------------

def clean_for_serialization(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: clean_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_serialization(x) for x in obj]
    if isinstance(obj, set):
        return sorted(clean_for_serialization(x) for x in obj)
    if callable(obj):
        return f"<function {getattr(obj, '__name__', 'unknown')}>"
    if hasattr(obj, "__dict__"):
        try:
            return clean_for_serialization(obj.__dict__)
        except Exception:
            return str(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def save_results_with_fallback(results: Dict[str, Any], filepath: str) -> None:
    # Try JSON first
    cleaned = clean_for_serialization(results)
    path = filepath if filepath.endswith('.json') else f"{filepath}.json"
    with open(path, 'w') as f:
        json.dump(cleaned, f, indent=2)
    print(f"[save] Wrote cleaned JSON {path}")


# -----------------------------
# Core evaluation
# -----------------------------

def run_evaluation(
    lm: VLLMSteeringLM,
    model_name: str,
    tasks: List[str],
    experiment_id: str,
    num_fewshot: int = 0,
    limit: Optional[int] = 1000,
    random_seed: int = 42,
    numpy_random_seed: int = 42,
    torch_random_seed: int = 42,
    fewshot_random_seed: int = 42,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    max_gen_toks: Optional[int] = None,
    thinking: Optional[bool] = None,
    apply_chat_template: bool = True,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"Experiment: {experiment_id}")
    print(f"Model:      {model_name}")
    print(f"Tasks:      {tasks}\n")

    # Auto-detect Qwen models and set thinking default via system instruction
    system_instruction = None
    if thinking is None and "qwen" in model_name.lower():
        system_instruction = "You are a helpful assistant. /no_think"
        print(f"[config] Detected Qwen model, using system_instruction to disable thinking")
    elif thinking is False:
        system_instruction = "You are a helpful assistant. /no_think"
        print(f"[config] Using system_instruction to disable thinking")
    elif thinking is True:
        system_instruction = ""
        print(f"[config] Using system_instruction with thinking enabled")

    # Run eval with a timer
    t0 = time.time()
    gen_kwargs = {}
    if max_gen_toks is not None:
        gen_kwargs["max_gen_toks"] = max_gen_toks
        lm._max_gen_toks = max_gen_toks

    # Fix eq_bench: Set until to empty string to signal EOS-only stopping.
    # The default fewshot_delimiter of "\n\n" prevents eq_bench from generating proper responses.
    # Using [""] (list with empty string) signals to lm-eval that until IS specified but with no stop strings.
    if any("eq_bench" in task.lower() for task in tasks):
        gen_kwargs["until"] = [""]
        print("[config] Detected eq_bench task, setting until=[''] to use model's EOS token")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
        confirm_run_unsafe_code=True,
        use_cache=use_cache,
        cache_requests=cache_requests,
        apply_chat_template=apply_chat_template,
        system_instruction=system_instruction,
        gen_kwargs=gen_kwargs,
    )
    elapsed = time.time() - t0

    # Attach metadata + timing
    results['experiment_id'] = experiment_id
    results['model_name'] = model_name
    results.setdefault('meta', {})['elapsed_seconds'] = elapsed

    # Quick throughput print (approx; uses provided limit if set)
    try:
        # If limit is None, try to infer n from configs; otherwise use limit
        total_items = 0
        if limit is not None:
            total_items = int(limit) * len(tasks)
        else:
            for t in tasks:
                cfg = results.get('configs', {}).get(t, {})
                n = int(cfg.get('count', 0))
                total_items += n
        if total_items > 0 and elapsed > 0:
            print(f"[perf] ~{total_items/elapsed:.2f} samples/sec over {total_items} items ({elapsed:.1f}s)")
    except Exception:
        pass

    return results


# -----------------------------
# CLI
# -----------------------------

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate vLLM steered models on benchmarks (fast path)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config_filepath", type=str, required=True,
                   help="Path to multi-capping config .pt")
    p.add_argument("--cap_from", type=str, default="",
                   help="Folder for the cap experiment")
    p.add_argument("--experiment_ids", type=str, nargs="+", required=True,
                   help="Experiment IDs to run; use 'baseline' for no steering; 'all' to expand all in config")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tasks", type=str, default="mmlu_pro",
                   help="Comma-separated task list (e.g., mmlu_pro,gsm8k,ifeval)")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=1000,
                   help="Limit number of examples per task (None/<=0 for full)")
    p.add_argument("--max_gen_toks", type=int, default=None,
                   help="Cap generation length (tokens) for generation tasks")

    # vLLM-specific parameters
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Number of GPUs for tensor parallelism")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                   help="GPU memory utilization (0.0-1.0)")
    p.add_argument("--max_model_len", type=int, default=None,
                   help="Maximum model context length")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="auto",
                   help="Model dtype")

    # Caching
    p.add_argument("--use_cache", type=str, default=None,
                   help="Path to lm-eval cache directory (enables skipping repeat requests)")
    p.add_argument("--cache_requests", action="store_true",
                   help="Enable request-level cache in lm-eval")

    # Seeds
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--numpy_random_seed", type=int, default=42)
    p.add_argument("--torch_random_seed", type=int, default=42)
    p.add_argument("--fewshot_random_seed", type=int, default=42)

    # Thinking mode (for Qwen models)
    p.add_argument("--thinking", type=lambda x: x.lower() in ['true', '1', 'yes'], default=None,
                   help="Enable thinking mode for chat templates. If not specified, automatically set to False for Qwen models, unset for other models.")

    # Chat template (default True)
    p.add_argument("--no_chat_template", dest="apply_chat_template", action="store_false", default=True,
                   help="Disable chat template application (for multiple-choice tasks like MMLU-Pro). Default is to apply chat templates.")

    return p.parse_args()


def cleanup_vllm_model(vllm_model: VLLMSteerModel) -> None:
    """
    Clean up vLLM resources to prevent GPU memory leaks.

    vLLM doesn't provide an official cleanup API, so we manually destroy
    parallel state and delete internal components based on community workarounds.
    See: https://github.com/vllm-project/vllm/issues/1908

    This cleanup only terminates child processes spawned by this script,
    not all VLLM processes on the system.
    """
    import multiprocessing

    try:
        print("[cleanup] Destroying vLLM model and releasing GPU memory...")

        # Step 1: Shutdown model executor FIRST to gracefully terminate workers
        # This allows workers to clean up their NCCL resources before exit
        try:
            if hasattr(vllm_model, 'llm') and hasattr(vllm_model.llm, 'llm_engine'):
                engine = vllm_model.llm.llm_engine
                if hasattr(engine, 'model_executor'):
                    print("[cleanup] Shutting down model executor...")
                    engine.model_executor.shutdown()
                    print("[cleanup] Model executor shutdown complete")
        except Exception as e:
            print(f"[cleanup] Warning: Could not shutdown model executor: {e}")

        # Step 2: Destroy distributed parallel state
        try:
            print("[cleanup] Destroying distributed parallel state...")
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            print("[cleanup] Destroyed model parallel state")
        except ImportError:
            print("[cleanup] Warning: Could not import destroy_model_parallel")
        except Exception as e:
            print(f"[cleanup] Warning: destroy_model_parallel failed: {e}")

        # Step 3: Delete internal engine components
        try:
            if hasattr(vllm_model, 'llm') and hasattr(vllm_model.llm, 'llm_engine'):
                engine = vllm_model.llm.llm_engine
                # Delete model executor (already shut down)
                if hasattr(engine, 'model_executor'):
                    del engine.model_executor
                    print("[cleanup] Deleted model executor")
        except Exception as e:
            print(f"[cleanup] Warning: Could not delete model executor: {e}")

        # Step 4: Delete the LLM engine reference
        try:
            if hasattr(vllm_model, 'llm'):
                del vllm_model.llm
                print("[cleanup] Deleted LLM engine")
        except Exception as e:
            print(f"[cleanup] Warning: Could not delete LLM engine: {e}")

        # Step 5: Run garbage collection to trigger cleanup
        gc.collect()
        print("[cleanup] Ran garbage collection")

        # Step 6: Final GPU cleanup
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[cleanup] Cleared CUDA cache and synchronized")

        print("[cleanup] Cleanup complete!")

    except Exception as e:
        print(f"[cleanup] Error during cleanup: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    args = parse_arguments()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]

    # Load experiments using compat module
    print(f"[config] Loading legacy config: {args.config_filepath}")
    all_experiments = load_legacy_role_trait_config(args.config_filepath)
    print(f"[config] Loaded {len(all_experiments)} experiments")

    # Build lookup dict for experiments
    experiments_by_id = {exp.id: exp for exp in all_experiments}

    # Expand 'all' and validate experiment IDs
    exp_ids: List[str] = []
    for exp_id in args.experiment_ids:
        if exp_id.lower() == 'all':
            all_ids = [exp.id for exp in all_experiments]
            exp_ids.extend(all_ids)
            print(f"[cli] Expanded 'all' → {len(all_ids)} experiments")
        else:
            exp_ids.append(exp_id)

    print(f"[cli] Total experiments to run: {len(exp_ids)} -> {exp_ids}")

    # === Load model once for all experiments ===
    print("\n" + "=" * 70)
    print("[init] Loading vLLM model once for all experiments")
    print("=" * 70)

    # Determine bootstrap layers from all experiments
    bootstrap_layers = set()
    for exp_id in exp_ids:
        if exp_id.lower() not in {"baseline", "unsteered", "control"}:
            if exp_id in experiments_by_id:
                exp = experiments_by_id[exp_id]
                bootstrap_layers.update(exp.spec.layers.keys())

    bootstrap_layers = tuple(sorted(bootstrap_layers))
    print(f"[init] Bootstrap layers for steering: {bootstrap_layers}")

    # Create VLLMSteeringConfig
    vllm_cfg = VLLMSteeringConfig(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        bootstrap_layers=bootstrap_layers,
    )

    # Initialize VLLMSteerModel
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True)
    print(f"[init] Model loaded: {args.model_name}")
    print(f"[init] Hidden size: {vllm_model.hidden_size}")
    print(f"[init] Layer count: {vllm_model.layer_count}\n")

    # Wrap in lm-eval compatible interface with auto batch size
    lm = VLLMSteeringLM(vllm_model, batch_size="auto")

    # Use try-finally to ensure cleanup happens even on errors or Ctrl+C
    try:
        for exp_id in exp_ids:
            try:
                # === Create structured directories ===
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_suffix = timestamp

                run_dirs = {}
                is_baseline = exp_id.lower() in {"baseline", "unsteered", "control"}
                for task in tasks:
                    if is_baseline:
                        # Baseline: out_dir/task/baseline/run_suffix (no exp_id)
                        task_dir = out_dir / task / "baseline" / run_suffix
                    else:
                        # Steered: out_dir/task/cap_from/exp_id/run_suffix
                        task_dir = out_dir / task / args.cap_from / exp_id / run_suffix
                    task_dir.mkdir(parents=True, exist_ok=True)
                    run_dirs[task] = task_dir

                print(f"\n[run] experiment={exp_id}\n[run] dirs={run_dirs}\n")

                # === Run evaluation with steering context manager ===
                if not is_baseline:
                    # Get experiment and apply steering
                    if exp_id not in experiments_by_id:
                        raise ValueError(f"Experiment '{exp_id}' not found in config")

                    experiment = experiments_by_id[exp_id]
                    steering_spec = experiment.spec
                    unique_layers = sorted(steering_spec.layers.keys())
                    num_interventions = len(steering_spec.layers)
                    print(f"[steer] Applying {num_interventions} projection cap(s) on layers {unique_layers}")

                    # Apply steering manually since we're in sync context
                    import asyncio
                    asyncio.run(vllm_model.push_steering_spec(steering_spec))
                    try:
                        print("[steer] Steering active (projection capping only)")
                        results = run_evaluation(
                            lm=lm,
                            model_name=args.model_name,
                            tasks=tasks,
                            experiment_id=exp_id,
                            num_fewshot=args.num_fewshot,
                            limit=(None if args.limit is None or int(args.limit) <= 0 else int(args.limit)),
                            random_seed=args.random_seed,
                            numpy_random_seed=args.numpy_random_seed,
                            torch_random_seed=args.torch_random_seed,
                            fewshot_random_seed=args.fewshot_random_seed,
                            use_cache=args.use_cache,
                            cache_requests=args.cache_requests,
                            max_gen_toks=args.max_gen_toks,
                            thinking=args.thinking,
                            apply_chat_template=args.apply_chat_template,
                        )
                    finally:
                        asyncio.run(vllm_model.pop_steering_spec())
                        print("[steer] Steering removed")
                else:
                    # Baseline: no steering
                    print("[baseline] Running without steering")
                    results = run_evaluation(
                        lm=lm,
                        model_name=args.model_name,
                        tasks=tasks,
                        experiment_id=exp_id,
                        num_fewshot=args.num_fewshot,
                        limit=(None if args.limit is None or int(args.limit) <= 0 else int(args.limit)),
                        random_seed=args.random_seed,
                        numpy_random_seed=args.numpy_random_seed,
                        torch_random_seed=args.torch_random_seed,
                        fewshot_random_seed=args.fewshot_random_seed,
                        use_cache=args.use_cache,
                        cache_requests=args.cache_requests,
                        max_gen_toks=args.max_gen_toks,
                        thinking=args.thinking,
                        apply_chat_template=args.apply_chat_template,
                    )

                # === Save per-task results & create symlink ===
                for task, task_dir in run_dirs.items():
                    result_path = task_dir / "results.json"
                    save_results_with_fallback(results, str(result_path))

                    # Write manifest (all CLI args + timestamp)
                    manifest = vars(args).copy()
                    manifest.update({
                        "timestamp": timestamp,
                        "experiment_id": exp_id,
                        "task": task,
                        "run_dir": str(task_dir),
                        "vllm": True,
                    })
                    with open(task_dir / "manifest.json", "w") as f:
                        json.dump(manifest, f, indent=2)

                    # Update latest symlink
                    latest_link = task_dir.parent / "latest"
                    try:
                        if latest_link.exists() or latest_link.is_symlink():
                            latest_link.unlink()
                        os.symlink(task_dir.name, latest_link)
                    except Exception as e:
                        print(f"[warn] could not create symlink {latest_link}: {e}")

                    print(f"[save] results → {result_path}")
                    print(f"[save] manifest → {task_dir / 'manifest.json'}")
                    print(f"[link] latest → {latest_link}")

            except Exception as e:
                print(f"[error] Experiment {exp_id} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    finally:
        # Always clean up vLLM resources, even if interrupted with Ctrl+C
        cleanup_vllm_model(vllm_model)


if __name__ == "__main__":
    main()
