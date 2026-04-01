# StrongREJECT steered eval

Runs [StrongREJECT](https://github.com/dsbowen/strong_reject) under three
intervention modes against the same vLLM steering path used elsewhere in this
repo (`chatspace.generation.vllm_steer_model.VLLMSteerModel`):

| mode | config | what it does |
|---|---|---|
| `baseline` | none | unsteered generation |
| `additive` | `evals/configs/*_config.pt` (built by `evals/0_vector_config.ipynb`) | additive steering, fraction-of-avg-norm coefficients already baked in |
| `capping` | `capped/configs/**/*_config.pt` (built by `capped/0_jailbreak_config.ipynb`) | activation projection cap, loaded via `chatspace.generation.compat.load_legacy_role_trait_config` |

Score = StrongREJECT harmfulness (0 = refused, 1 = fully harmful).

## Install

```bash
pip install git+https://github.com/dsbowen/strong_reject.git@main
export OPENAI_API_KEY=...   # for default strongreject_rubric judge
# or: export HF_TOKEN=...   # and use --evaluator strongreject_finetuned
```

## Run

```bash
uv run evals/strongreject/run_strongreject.py \
    --mode capping \
    --config_filepath /workspace/qwen-3-32b/capped/configs/contrast/role_trait_config.pt \
    --experiment_ids layers_46:54-p0.25 \
    --model_name Qwen/Qwen3-32B --thinking false --max_model_len 4096 --dtype bfloat16 \
    --output_dir /workspace/qwen-3-32b/evals/strongreject
```

See `run_all.sh` for the full Qwen / Llama / Gemma matrix including the
pareto-best capping settings.

## Outputs

- `<output_dir>/<mode>_responses.jsonl` — one row per (prompt × experiment) with
  `forbidden_prompt`, `response`, `score`, `refusal`, `convincingness`, `specificity`
- `<output_dir>/<mode>_summary.csv` — mean/std score grouped by `experiment_id`

Generation is resumable: re-running skips `(id, experiment_id)` pairs already in
the responses jsonl. Use `--skip_generate` / `--skip_eval` to run the two stages
separately, `--small` for the 60-prompt subset, `--limit N` for smoke tests.
