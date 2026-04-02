# Eval Session Handoff

Summary of what was run, what broke, and how it was fixed.
Full results are in `evals/results.md`.

---

## What Was Run

Three evals across three models on a single H200 (141 GB):

| Eval | Script | Judge |
|------|--------|-------|
| StrongREJECT | `evals/strongreject/run_strongreject.py` | OpenAI rubric via `strong_reject` |
| HarmBench ITW | `evals/1_steering_vllm.py` / `1_steering_hf.py` | `evals/2_harmbench_judge.py` |
| SorryBench base64 | `evals/1_steering_vllm.py` / `1_steering_hf.py` | decode → `evals/2_harmbench_judge.py` |

Models:
- **Gemma-2-27B**: additive steering only (no capping — pre/post layernorm arch incompatible)
- **Qwen-3-32B**: additive steering + capping
- **Llama-3.3-70B**: baselines only (int8 quantization degrades alignment too much for steering to be meaningful — 52.8% HarmBench ITW baseline vs. ~14% for Qwen)

---

## API Fixes Required (chatspace/steerllm)

The steerllm API had changed since the scripts were written. Three files needed updating:

### 1. `evals/strongreject/run_strongreject.py` (fixed)
### 2. `evals/1_steering_vllm.py` (fixed)
### 3. `capped/1_capping_vllm.py` (fixed)

All three had the same set of issues:

**`_ensure_engine_initialized()` removed**
```python
# OLD (breaks at runtime)
await vllm_model._ensure_engine_initialized()

# NEW — model initializes eagerly in __init__, just remove the call
logger.info("VLLM async engine ready")
```

**`push_steering_spec` / `pop_steering_spec` replaced with per-request kwarg**
```python
# OLD
await vllm_model.push_steering_spec(steering_spec)
result = await vllm_model.generate([prompt], sampling_params)
await vllm_model.pop_steering_spec()

# NEW — steering_spec passed per-request
result = await vllm_model.generate(prompt, sampling_params, steering_spec=steering_spec)
```

**`generate()` returns a tuple; prompt must be a string not a list**
```python
# OLD
texts = await vllm_model.generate([prompt], sampling_params)
text = texts[0].strip()

# NEW
result = await vllm_model.generate(prompt, sampling_params, steering_spec=spec)
texts = result[0] if isinstance(result, tuple) else result
text = texts[0].strip()
```

**`LayerSteeringSpec` constructor changed from keyword args to `operations=` list**
```python
# OLD (in load_steering_config() in 1_steering_vllm.py)
layers[layer_idx] = LayerSteeringSpec(
    add=add_spec,
    projection_cap=None,
    ablation=None
)

# NEW
layers[layer_idx] = LayerSteeringSpec(operations=[add_spec])
```

### `--quantization` flag (added to both vllm scripts)
Needed for Llama 70B on a single H200:
```bash
--quantization bitsandbytes
```
Passed through as `VLLMSteerModel(..., quantization="bitsandbytes")`.

---

## Gemma-2-27B: Flash Attention Fix

Gemma-2 uses tanh softcapping in its attention, which Flash Attention 3 (default on H200) doesn't support. Set this env var before any Gemma vLLM job:

```bash
export VLLM_FLASH_ATTN_VERSION=2
```

Without it you get: `RuntimeError: This flash attention build does not support tanh softcapping`.

---

## Judge Pipeline: Field Name Mismatch

`2_harmbench_judge.py` expects a `prompt` field, but the generation scripts write `question`. Preprocess before judging:

```bash
python3 -c "
import json
src='path/to/output.jsonl'; dst='path/to/output_for_judge.jsonl'
with open(src) as f, open(dst,'w') as g:
    for l in f:
        r=json.loads(l); r['prompt']=r.get('question', r.get('prompt','')); g.write(json.dumps(r)+'\n')
"
uv run evals/2_harmbench_judge.py output_for_judge.jsonl \
    --output output_scores.jsonl \
    --prompt-file evals/harmbench/prompts.py
```

**Always pass `--prompt-file evals/harmbench/prompts.py`** — the argument is required and the error message doesn't make it obvious.

## SorryBench: Decode Before Judging

Model responses are in base64. Decode first:

```bash
python3 evals/sorrybench_encode/decode_responses.py path/to/output.jsonl
# writes path/to/output_decoded.jsonl
uv run evals/2_harmbench_judge.py output_decoded.jsonl \
    --output output_scores.jsonl \
    --prompt-file evals/harmbench/prompts.py
```

---

## GPU / task-spooler Notes

- `ts` binary lives at `/root/git/task-spooler/build/ts` — add to PATH
- `TS_VISIBLE_DEVICES=0` must be set before starting the ts server, otherwise GPU jobs won't see the GPU
- `ts -S 2` sets 2 concurrent slots (needed so CPU judge jobs can run alongside a GPU generation job)
- `ts -D <job_id>` chains jobs: `ts -D 17 cmd` runs `cmd` only after job 17 succeeds
- If a vLLM process is killed mid-run, the GPU memory may not be freed. Check with `nvidia-smi` and `kill -9 <pid>` the orphaned EngineCore process if needed

---

## Capping Config Paths

The correct config files (not the ones referenced in some run.sh comments):

| Model | Capping config |
|-------|---------------|
| Qwen-3-32B | `/workspace/qwen-3-32b/capped/configs/contrast/role_trait_sliding_config.pt` |
| Llama-3.3-70B | `/workspace/llama-3.3-70b/capped/configs/contrast/role_trait_config.pt` |

Experiment ID for both: `layers_46:54-p0.25` (Qwen) / `layers_56:72-p0.25` (Llama).

Note: some older run.sh files reference `pc1/pc1_role_trait_config.pt` for Llama — that path doesn't exist. Use the contrast config above.

---

## Llama TODO (needs ≥2 GPUs for bf16)

```bash
# HarmBench ITW — additive sweep
ts -G 2 uv run evals/1_steering_vllm.py \
    --questions_file evals/harmbench/harmbench.jsonl \
    --roles_file evals/harmbench/roles_itw.jsonl \
    --config_filepath /workspace/llama-3.3-70b/evals/configs/asst_pc1_contrast_config.pt \
    --output_jsonl /workspace/llama-3.3-70b/evals/harmbench_itw/asst_pc1_contrast.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta --name Llama \
    --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2

# HarmBench ITW — capping
ts -G 2 uv run capped/1_capping_vllm.py \
    --questions_file evals/harmbench/harmbench.jsonl \
    --roles_file evals/harmbench/roles_itw.jsonl \
    --config_filepath /workspace/llama-3.3-70b/capped/configs/contrast/role_trait_config.pt \
    --experiment_ids layers_56:72-p0.25 \
    --output_jsonl /workspace/llama-3.3-70b/evals/harmbench_itw/capped_role_trait.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta \
    --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2

# SorryBench base64 — additive sweep
ts -G 2 uv run evals/1_steering_vllm.py \
    --questions_file evals/sorrybench_encode/sorrybench_base64.jsonl \
    --config_filepath /workspace/llama-3.3-70b/evals/configs/asst_pc1_contrast_config.pt \
    --output_jsonl /workspace/llama-3.3-70b/evals/sorrybench_encode/asst_pc1_contrast.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta --name Llama \
    --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2

# SorryBench base64 — capping
ts -G 2 uv run capped/1_capping_vllm.py \
    --questions_file evals/sorrybench_encode/sorrybench_base64.jsonl \
    --roles_file evals/sorrybench_encode/roles_none.jsonl \
    --config_filepath /workspace/llama-3.3-70b/capped/configs/contrast/role_trait_config.pt \
    --experiment_ids layers_56:72-p0.25 \
    --output_jsonl /workspace/llama-3.3-70b/evals/sorrybench_encode/capped_role_trait.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct --company Meta \
    --max_model_len 4096 --dtype bfloat16 --tensor_parallel_size 2
```

Then decode/preprocess and judge the same way as the other models.
