#!/usr/bin/env python3
"""
Benchmark Evaluation with Steering (Fast on Big GPUs)

This script evaluates (optionally steered) HF transformers models on benchmark
suites using lm-evaluation-harness. It keeps your original multi-vector
projection-cap capping logic, and adds practical speed knobs for large GPUs
(H200/A100/H100):

- Choice of replication (data-parallel via concurrent processes) vs sharding
  (model-parallel via device_map='auto').
- bf16 + optional FlashAttention-2, TF32 toggles.
- Request-level caching in lm-eval.
- Large-batch friendly defaults for MC ranking tasks like MMLU-Pro.
- Result merging and robust JSON/PT saving.
- Tiny live throughput report.

Example (baseline, fast MMLU-Pro on a single H200 with a big batch):

  uv run 2_benchmark_eval.py \
    --config_filepath /workspace/qwen-3-32b/evals/multi_capping_config.pt \
    --experiment_ids baseline \
    --model_name google/gemma-2-27b-it \
    --tasks mmlu_pro \
    --output_dir /root/git/persona-subspace/evals/benchmark_results \
    --batch_size 256 \
    --device_strategy replicate \
    --limit 1000 \
    --use_cache .lm_eval_cache --cache_requests

Example (shard the model when needed, e.g., very large checkpoints or long ctx):

  uv run 2_benchmark_eval.py \
    --config_filepath /workspace/qwen-3-32b/evals/multi_capping_config.pt \
    --experiment_ids baseline cap_1.0 \
    --model_name google/gemma-2-27b-it \
    --tasks mmlu_pro,gsm8k \
    --output_dir /root/git/persona-subspace/evals/benchmark_results \
    --batch_size 64 \
    --device_strategy shard \
    --limit 1000

"""

import argparse
from datetime import datetime
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
os.environ.setdefault("TORCH_ALLOW_TF32", "1")    # allow TF32 on Ampere+/Hopper
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------
# Project paths
# -----------------------------
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))

from utils.steering_utils import create_projection_cap_steerer  # type: ignore

# -----------------------------
# lm-eval imports
# -----------------------------
import lm_eval
from lm_eval.models.huggingface import HFLM


class SteeredHFLM(HFLM):
    """HuggingFace backend with optional multi-vector projection-cap steering."""

    def __init__(
        self,
        pretrained: str,
        steering_vectors: Optional[List[torch.Tensor]] = None,
        cap_thresholds: Optional[List[float]] = None,
        layer_indices: Optional[List[int]] = None,
        # common speed/model kwargs forwarded to HF under the hood
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,  # 'auto' -> shard layers across GPUs
        parallelize: Optional[bool] = None, # legacy path (not recommended here)
        **kwargs,
    ):
        # Build the kwargs for HFLM (these get passed to from_pretrained internally)
        hf_kwargs = dict(
            pretrained=pretrained,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        if torch_dtype is not None:
            hf_kwargs["dtype"] = torch_dtype  # HFLM accepts 'dtype' alias
        if attn_implementation is not None:
            hf_kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            hf_kwargs["device_map"] = device_map
        if parallelize is not None:
            hf_kwargs["parallelize"] = parallelize
        hf_kwargs.update(kwargs)

        super().__init__(**hf_kwargs)

        # Steering config
        self.steering_vectors = steering_vectors
        self.cap_thresholds = cap_thresholds
        self.layer_indices = layer_indices
        self._steerer = None

        if (
            self.steering_vectors is not None
            and self.cap_thresholds is not None
            and self.layer_indices is not None
        ):
            self._setup_steering()

    def _setup_steering(self) -> None:
        if self.steering_vectors is None:
            return
        unique_layers = sorted(set(self.layer_indices))
        print(f"[steer] Registering {len(self.steering_vectors)} vectors across layers {unique_layers}")
        self._steerer = create_projection_cap_steerer(
            model=self.model,
            feature_directions=self.steering_vectors,
            cap_thresholds=self.cap_thresholds,
            layer_indices=self.layer_indices,
            positions="all",
        )
        self._steerer.__enter__()
        print("[steer] Hooks registered")

    def _cleanup_steering(self) -> None:
        if self._steerer is not None:
            self._steerer.__exit__(None, None, None)
            self._steerer = None
            print("[steer] Hooks removed")



# -----------------------------
# Config loading helpers
# -----------------------------

def load_multi_config(config_filepath: str) -> Tuple[Dict[str, Dict], List[Dict]]:
    print(f"[config] Loading multi-capping config: {config_filepath}")
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Config file not found: {config_filepath}")
    cfg = torch.load(config_filepath, weights_only=False)
    if not isinstance(cfg, dict) or 'vectors' not in cfg or 'experiments' not in cfg:
        raise ValueError("Config must be a dict with keys 'vectors' and 'experiments'")
    vectors = cfg['vectors']
    experiments = cfg['experiments']
    print(f"[config] {len(vectors)} vectors; {len(experiments)} experiments")
    return vectors, experiments


def get_experiment_config(experiment_id: str,
                          vectors_dict: Dict[str, Dict],
                          experiments_list: List[Dict]
                          ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[float]], Optional[List[int]]]:
    if experiment_id.lower() in {"baseline", "unsteered", "control"}:
        print(f"[exp:{experiment_id}] baseline (no steering)")
        return None, None, None

    exp = next((e for e in experiments_list if e.get('id') == experiment_id), None)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in config")

    interventions = exp.get('interventions', [])
    steering_vectors: List[torch.Tensor] = []
    cap_thresholds: List[float] = []
    layer_indices: List[int] = []

    for it in interventions:
        vec_name = it['vector']
        cap_val = float(it['cap'])
        if vec_name not in vectors_dict:
            raise ValueError(f"Vector '{vec_name}' not found in config")
        vdata = vectors_dict[vec_name]
        v = vdata['vector']
        if isinstance(v, torch.Tensor):
            steering_vectors.append(v.clone().detach())
        else:
            steering_vectors.append(torch.tensor(v))
        cap_thresholds.append(cap_val)
        layer_indices.append(int(vdata['layer']))

    print(f"[exp:{experiment_id}] {len(interventions)} interventions on layers {sorted(set(layer_indices))}")
    return steering_vectors, cap_thresholds, layer_indices


# -----------------------------
# Result save/merge helpers
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


def load_existing_all_results(output_dir: str) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    p_json = Path(output_dir) / "all_results.json"
    if p_json.exists():
        try:
            merged = json.loads(p_json.read_text())
            print(f"[load] Loaded {p_json}")
            return merged
        except Exception as e:
            print(f"[load] Failed to read {p_json}: {e}")
    return {}


def load_existing_experiment_results(output_dir: str, experiment_id: str) -> Optional[Dict[str, Any]]:
    p = Path(output_dir) / f"{experiment_id}_results.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as e:
            print(f"[load] Could not load existing {p}: {e}")
    return None


def merge_results(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(existing)
    if 'results' in new:
        out.setdefault('results', {})
        out['results'].update(new['results'])
    for k in ['experiment_id', 'model_name', 'configs', 'versions']:
        if k in new:
            out[k] = new[k]
    return out


# -----------------------------
# Core evaluation
# -----------------------------

def run_evaluation(
    model_name: str,
    tasks: List[str],
    experiment_id: str,
    steering_vectors: Optional[List[torch.Tensor]],
    cap_thresholds: Optional[List[float]],
    layer_indices: Optional[List[int]],
    batch_size: int = 8,
    num_fewshot: int = 0,
    limit: Optional[int] = 1000,
    random_seed: int = 42,
    numpy_random_seed: int = 42,
    torch_random_seed: int = 42,
    fewshot_random_seed: int = 42,
    device_strategy: str = "replicate",  # 'replicate' | 'shard'
    torch_dtype_str: str = "bfloat16",
    attn_impl: Optional[str] = None,      # e.g., 'flash_attention_2'
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    max_gen_toks: Optional[int] = None,
    thinking: Optional[bool] = None,
    apply_chat_template: bool = True,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"Experiment: {experiment_id}")
    print(f"Model:      {model_name}")
    print(f"Tasks:      {tasks}")
    print(f"Strategy:   {device_strategy}\n")

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

    # Map dtype string → torch dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

    # Build model kwargs
    model_kwargs: Dict[str, Any] = dict(
        pretrained=model_name,
        steering_vectors=steering_vectors,
        cap_thresholds=cap_thresholds,
        layer_indices=layer_indices,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if attn_impl:
        model_kwargs['attn_implementation'] = attn_impl

    if device_strategy == "shard":
        ngpu = torch.cuda.device_count()
        print(f"[init] Sharding with device_map='auto' across {ngpu} GPUs")
        model_kwargs['device_map'] = 'auto'
    else:
        print("[init] Replication (single-process). Tip: launch multiple processes on different GPUs to parallelize tasks.")

    # Init model
    lm = SteeredHFLM(**model_kwargs)

    # Run eval with a timer
    t0 = time.time()
    gen_kwargs = {}
    if max_gen_toks is not None:
        gen_kwargs["max_gen_toks"] = max_gen_toks

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
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

    # Cleanup hooks
    lm._cleanup_steering()

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
        description="Evaluate steered models on benchmarks (fast path)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config_filepath", type=str, required=True,
                   help="Path to multi-capping config .pt")
    p.add_argument("--experiment_ids", type=str, nargs="+", required=True,
                   help="Experiment IDs to run; use 'baseline' for no steering; 'all' to expand all in config")
    p.add_argument("--model_name", type=str, default="google/gemma-2-27b-it")
    p.add_argument("--tasks", type=str, default="mmlu_pro",
                   help="Comma-separated task list (e.g., mmlu_pro,gsm8k,ifeval)")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=1000,
                   help="Limit number of examples per task (None/<=0 for full)")
    p.add_argument("--max_gen_toks", type=int, default=None,
               help="Cap generation length (tokens) for generation tasks")

    # Parallelism strategy
    p.add_argument("--device_strategy", choices=["replicate", "shard"], default="replicate",
                   help="replicate: keep model on a single GPU (run big batches or run multiple processes). shard: device_map=auto across GPUs.")

    # Dtype / kernels
    p.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn_implementation", type=str, default=None,
                   help="Set to 'flash_attention_2' if available, else leave unset")

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

def main() -> None:
    args = parse_arguments()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]

    vectors_dict, experiments_list = load_multi_config(args.config_filepath)

    # Expand 'all'
    exp_ids: List[str] = []
    for exp_id in args.experiment_ids:
        if exp_id.lower() == 'all':
            all_ids = [e['id'] for e in experiments_list]
            exp_ids.extend(all_ids)
            print(f"[cli] Expanded 'all' → {len(all_ids)} experiments")
        else:
            exp_ids.append(exp_id)

    print(f"[cli] Total experiments to run: {len(exp_ids)} -> {exp_ids}")

    for exp_id in exp_ids:
        try:
            steering_vectors, caps, layers = get_experiment_config(exp_id, vectors_dict, experiments_list)

            # === Create structured directories ===
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_suffix = f"{timestamp}_seed{args.random_seed}_limit{args.limit}_shots{args.num_fewshot}"

            # baseline → /baseline/{task}/{run_suffix}
            # others   → /role_trait/{exp_id}/{task}/{run_suffix}
            if exp_id.lower() in {"baseline", "unsteered", "control"}:
                exp_root = out_dir
            else:
                exp_root = out_dir / exp_id

            run_dirs = {}
            for task in tasks:
                task_dir = exp_root / task / run_suffix
                task_dir.mkdir(parents=True, exist_ok=True)
                run_dirs[task] = task_dir

            print(f"\n[run] experiment={exp_id}\n[run] dirs={run_dirs}\n")

            # === Run evaluation ===
            results = run_evaluation(
                model_name=args.model_name,
                tasks=tasks,
                experiment_id=exp_id,
                steering_vectors=steering_vectors,
                cap_thresholds=caps,
                layer_indices=layers,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
                limit=(None if args.limit is None or int(args.limit) <= 0 else int(args.limit)),
                random_seed=args.random_seed,
                numpy_random_seed=args.numpy_random_seed,
                torch_random_seed=args.torch_random_seed,
                fewshot_random_seed=args.fewshot_random_seed,
                device_strategy=args.device_strategy,
                torch_dtype_str=args.torch_dtype,
                attn_impl=args.attn_implementation,
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
            continue


if __name__ == "__main__":
    main()
