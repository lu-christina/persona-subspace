#!/usr/bin/env python3
"""
StrongREJECT steered evaluation.

Runs the StrongREJECT benchmark (https://github.com/dsbowen/strong_reject) against
a model under one of three intervention modes:

  baseline  - no steering (single experiment_id="baseline_unsteered")
  additive  - additive steering, config format as in evals/1_steering_vllm.py
              (coefficients are fraction-of-avg-norm scaled, baked into the config)
  capping   - activation projection capping, config format as in capped/1_capping_vllm.py
              (loaded via chatspace.generation.compat.load_legacy_role_trait_config)

For each experiment in the config, generates responses for every forbidden_prompt,
then scores them with the StrongREJECT evaluator (default: strongreject_rubric via
OpenAI API; alternatively strongreject_finetuned for a local HF judge).

Outputs:
  <output_dir>/<mode>_responses.jsonl  - one row per (prompt, experiment) with response + score
  <output_dir>/<mode>_summary.csv      - mean/std score per experiment_id

Example:
  uv run evals/strongreject/run_strongreject.py \
      --mode additive \
      --config_filepath /workspace/qwen-3-32b/evals/configs/asst_pc1_contrast_config.pt \
      --model_name Qwen/Qwen3-32B --thinking false \
      --output_dir /workspace/qwen-3-32b/evals/strongreject \
      --tensor_parallel_size 1 --max_model_len 4096 --dtype bfloat16
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from vllm import SamplingParams

# Add project paths (mirrors evals/1_steering_vllm.py and capped/1_capping_vllm.py)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "utils"))
sys.path.append(str(Path.home() / "git" / "chatspace"))
torch.set_float32_matmul_precision("high")

from chatspace.generation.vllm_steer_model import (  # noqa: E402
    AddSpec,
    LayerSteeringSpec,
    SteeringSpec,
    VLLMSteerModel,
    VLLMSteeringConfig,
)
from chatspace.generation.compat import load_legacy_role_trait_config  # noqa: E402
from strong_reject.load_datasets import load_strongreject, load_strongreject_small  # noqa: E402
from strong_reject.evaluate import evaluate_dataset  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SteeringExperiment:
    """A single experiment: id + the SteeringSpec to push to the vllm model."""

    id: str
    spec: SteeringSpec


# ---------------------------------------------------------------------------
# config loading (additive — same format as evals/1_steering_vllm.py)
# ---------------------------------------------------------------------------


def load_additive_config(config_path: str) -> List[SteeringExperiment]:
    """Load additive steering config (.pt) — same format as evals/1_steering_vllm.py.

    Config format:
        {'vectors': {name: {'vector': Tensor, 'layer': int}},
         'experiments': [{'id': str, 'interventions': [{'vector': name, 'coeff': float}]}]}
    """
    logger.info(f"Loading additive steering config from {config_path}")
    payload = torch.load(config_path, map_location="cpu")

    prepared_vectors = {}
    for name, vector_payload in payload["vectors"].items():
        tensor = vector_payload["vector"].detach().to(dtype=torch.float32).contiguous()
        prepared_vectors[name] = (vector_payload["layer"], tensor)
    logger.info(f"Loaded {len(prepared_vectors)} vectors")

    experiments: List[SteeringExperiment] = []
    for experiment in payload["experiments"]:
        layers: Dict[int, LayerSteeringSpec] = {}
        for intervention in experiment["interventions"]:
            layer_idx, vector_tensor = prepared_vectors[intervention["vector"]]
            layers[layer_idx] = LayerSteeringSpec(
                add=AddSpec(
                    vector=vector_tensor.clone(), scale=float(intervention["coeff"])
                ),
                projection_cap=None,
                ablation=None,
            )
        experiments.append(
            SteeringExperiment(id=experiment["id"], spec=SteeringSpec(layers=layers))
        )

    logger.info(f"Loaded {len(experiments)} additive experiments")
    return experiments


def load_capping_config(config_path: str) -> List[SteeringExperiment]:
    """Load capping config via chatspace's legacy loader (same as capped/1_capping_vllm.py)."""
    logger.info(f"Loading capping config from {config_path}")
    legacy_experiments = load_legacy_role_trait_config(config_path)
    experiments = [
        SteeringExperiment(id=exp.id, spec=exp.spec) for exp in legacy_experiments
    ]
    logger.info(f"Loaded {len(experiments)} capping experiments")
    return experiments


def build_experiments(args: argparse.Namespace) -> List[SteeringExperiment]:
    """Resolve mode + config into a list of SteeringExperiment, optionally filtered by --experiment_ids."""
    if args.mode == "baseline":
        return [
            SteeringExperiment(id="baseline_unsteered", spec=SteeringSpec(layers={}))
        ]

    if args.config_filepath is None:
        raise ValueError(f"--config_filepath is required for mode={args.mode}")

    if args.mode == "additive":
        experiments = load_additive_config(args.config_filepath)
    elif args.mode == "capping":
        experiments = load_capping_config(args.config_filepath)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.experiment_ids:
        by_id = {e.id: e for e in experiments}
        missing = [eid for eid in args.experiment_ids if eid not in by_id]
        if missing:
            raise ValueError(
                f"experiment_ids not found in config: {missing}. Available: {list(by_id)}"
            )
        experiments = [by_id[eid] for eid in args.experiment_ids]
        logger.info(
            f"Filtered to {len(experiments)} experiment(s): {[e.id for e in experiments]}"
        )

    return experiments


# ---------------------------------------------------------------------------
# prompt handling
# ---------------------------------------------------------------------------


def load_prompts(small: bool, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Load StrongREJECT prompts as a list of dicts with stable integer ids."""
    ds = load_strongreject_small() if small else load_strongreject()
    rows = [
        {
            "id": i,
            "forbidden_prompt": ds[i]["forbidden_prompt"],
            "category": ds[i].get("category", ""),
            "source": ds[i].get("source", ""),
        }
        for i in range(len(ds))
    ]
    if limit is not None:
        rows = rows[:limit]
    logger.info(
        f"Loaded {len(rows)} StrongREJECT prompts (small={small}, limit={limit})"
    )
    return rows


def format_prompt_for_chat(
    forbidden_prompt: str,
    tokenizer,
    enable_thinking: bool,
) -> str:
    """Format a single forbidden_prompt with the tokenizer's chat template.

    StrongREJECT prompts are presented as plain user turns with no system prompt
    (matches the benchmark's intended setup). Gemma handling matches
    evals/1_steering_vllm.py:format_prompt_for_chat.
    """
    messages = [{"role": "user", "content": forbidden_prompt}]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    return forbidden_prompt


# ---------------------------------------------------------------------------
# resume helpers
# ---------------------------------------------------------------------------


def read_existing(output_jsonl: Path) -> set:
    """Return set of (id, experiment_id) already present in output to support resume."""
    existing = set()
    if not output_jsonl.exists():
        return existing
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                existing.add((row.get("id"), row.get("experiment_id")))
            except json.JSONDecodeError:
                continue
    logger.info(f"Found {len(existing)} existing rows in {output_jsonl}")
    return existing


def append_rows(output_jsonl: Path, rows: List[Dict[str, Any]]) -> None:
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


async def generate_for_experiments(
    args: argparse.Namespace,
    experiments: List[SteeringExperiment],
    prompts: List[Dict[str, Any]],
    output_jsonl: Path,
) -> None:
    """Spin up the vllm steered model once, then iterate experiments."""
    existing = read_existing(output_jsonl)

    bootstrap_layers = tuple(
        sorted({layer for exp in experiments for layer in exp.spec.layers.keys()})
    )
    logger.info(f"Bootstrap layers: {bootstrap_layers}")

    vllm_cfg = VLLMSteeringConfig(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        bootstrap_layers=bootstrap_layers,
    )
    vllm_model = VLLMSteerModel(vllm_cfg, enforce_eager=True)
    tokenizer = vllm_model.tokenizer
    logger.info(
        f"VLLM model created: hidden_size={vllm_model.hidden_size} layers={vllm_model.layer_count}"
    )

    await vllm_model._ensure_engine_initialized()

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for exp in experiments:
        pending = [p for p in prompts if (p["id"], exp.id) not in existing]
        if not pending:
            logger.info(
                f"[{exp.id}] all {len(prompts)} prompts already generated, skipping"
            )
            continue
        logger.info(f"[{exp.id}] generating {len(pending)}/{len(prompts)} prompts")

        formatted = [
            format_prompt_for_chat(p["forbidden_prompt"], tokenizer, args.thinking)
            for p in pending
        ]

        has_steering = bool(exp.spec.layers)
        try:
            if has_steering:
                await vllm_model.push_steering_spec(exp.spec)
                logger.info(
                    f"[{exp.id}] steering pushed ({len(exp.spec.layers)} layer(s))"
                )

            responses: List[str] = []
            for i in range(0, len(formatted), args.concurrent_batch_size):
                batch = formatted[i : i + args.concurrent_batch_size]
                tasks = [
                    vllm_model.generate([prompt], sampling_params) for prompt in batch
                ]
                results = await asyncio.gather(*tasks)
                responses.extend(r[0].strip() for r in results)
                logger.info(
                    f"[{exp.id}] {min(i + len(batch), len(formatted))}/{len(formatted)}"
                )
        finally:
            if has_steering:
                await vllm_model.pop_steering_spec()

        rows = []
        for p, resp in zip(pending, responses):
            rows.append(
                {
                    "id": p["id"],
                    "experiment_id": exp.id,
                    "mode": args.mode,
                    "category": p["category"],
                    "source": p["source"],
                    "forbidden_prompt": p["forbidden_prompt"],
                    "response": resp,
                }
            )
        append_rows(output_jsonl, rows)
        logger.info(f"[{exp.id}] wrote {len(rows)} rows")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def run_evaluator(
    output_jsonl: Path, evaluator: str, judge_models: Optional[List[str]]
) -> pd.DataFrame:
    """Run StrongREJECT evaluator over generated responses, return per-row scored dataframe."""
    df = pd.read_json(output_jsonl, lines=True)
    if "score" in df.columns:
        already = df["score"].notna()
        if already.all():
            logger.info("All rows already scored, skipping evaluator")
            return df
    logger.info(f"Scoring {len(df)} rows with evaluator={evaluator}")

    ds = Dataset.from_pandas(df[["forbidden_prompt", "response"]], preserve_index=False)
    kwargs: Dict[str, Any] = {}
    if judge_models:
        kwargs["models"] = judge_models
    scored = evaluate_dataset(ds, [evaluator], **kwargs)
    scored_df = scored.to_pandas()

    for col in scored_df.columns:
        if col not in ("forbidden_prompt", "response"):
            df[col] = scored_df[col].values

    df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
    return df


def write_summary(df: pd.DataFrame, summary_csv: Path) -> None:
    summary = (
        df.groupby(["mode", "experiment_id"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["mode", "mean"])
    )
    summary.to_csv(summary_csv, index=False)
    logger.info(f"Summary written to {summary_csv}\n{summary.to_string(index=False)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StrongREJECT steered evaluation (baseline / additive / capping)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["baseline", "additive", "capping"], required=True
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        default=None,
        help="Steering config .pt file (required for additive/capping)",
    )
    parser.add_argument(
        "--experiment_ids",
        type=str,
        nargs="*",
        default=None,
        help="Subset of experiment ids from the config to run (default: all)",
    )
    parser.add_argument("--model_name", "--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use load_strongreject_small() (~60 prompts) instead of full set",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N prompts (for testing)",
    )

    # generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--thinking",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Enable thinking mode for chat templates (set false for Qwen)",
    )
    parser.add_argument("--concurrent_batch_size", type=int, default=64)

    # vllm
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="auto"
    )

    # evaluator
    parser.add_argument(
        "--evaluator",
        choices=["strongreject_rubric", "strongreject_finetuned"],
        default="strongreject_rubric",
        help="strongreject_rubric uses OPENAI_API_KEY; strongreject_finetuned uses a local HF gemma judge",
    )
    parser.add_argument(
        "--judge_models",
        type=str,
        nargs="*",
        default=None,
        help="Override judge model(s) for strongreject_rubric (default gpt-4o-mini)",
    )
    parser.add_argument(
        "--skip_generate",
        action="store_true",
        help="Skip generation, only run evaluator on existing jsonl",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluator, only generate responses",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / f"{args.mode}_responses.jsonl"
    summary_csv = output_dir / f"{args.mode}_summary.csv"

    experiments = build_experiments(args)
    prompts = load_prompts(args.small, args.limit)

    if not args.skip_generate:
        asyncio.run(generate_for_experiments(args, experiments, prompts, output_jsonl))
    else:
        logger.info("--skip_generate set; using existing responses jsonl")

    if not args.skip_eval:
        if args.evaluator == "strongreject_rubric" and not os.getenv("OPENAI_API_KEY"):
            logger.error(
                "OPENAI_API_KEY not set; required for strongreject_rubric. Use --skip_eval or set the key."
            )
            sys.exit(1)
        df = run_evaluator(output_jsonl, args.evaluator, args.judge_models)
        write_summary(df, summary_csv)
    else:
        logger.info("--skip_eval set; responses written to %s", output_jsonl)


if __name__ == "__main__":
    main()
