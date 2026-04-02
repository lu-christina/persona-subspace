# Steering Eval Results

Models evaluated on two jailbreak benchmarks with three conditions:
- **Baseline**: unsteered model
- **Additive**: additive residual-stream steering (coefficient sweep, `asst_pc1_contrast_config.pt`)
- **Capping**: activation capping via `role_trait_sliding_config.pt` at `layers_46:54-p0.25` (Qwen) / `layers_56:72-p0.25` (Llama)

Gemma-2-27B uses `1_steering_hf.py` (additive only — no capping due to pre/post layernorm architecture).  
Llama-3.3-70B uses int8 quantization (`--quantization bitsandbytes`) to fit on a single H200.  
Judge: HarmBench classifier (`2_harmbench_judge.py`). SorryBench responses decoded from base64 before judging.

---

## Summary

### HarmBench ITW (in-the-wild personas, n=954)

| Model | Baseline | Best Additive | Capping (`layers_*-p0.25`) |
|-------|----------|---------------|---------------------------|
| Gemma-2-27B | 17.4% | **2.2%** (`contrast -0.2`) | — |
| Qwen-3-32B | 13.6% | **0.0%** (`contrast -2`) | **8.6%** |
| Llama-3.3-70B (int8) | 52.8% | — | — |

### SorryBench base64 (n=440)

| Model | Baseline | Best Additive | Capping (`layers_*-p0.25`) |
|-------|----------|---------------|---------------------------|
| Gemma-2-27B | 2.3% | **0.2%** (`contrast -0.2`) | — |
| Qwen-3-32B | 6.1% | **0.7%** (`contrast -2`) | **3.2%** |
| Llama-3.3-70B (int8) | 4.8% | — | — |

> Note: Llama int8 baseline on HarmBench ITW is 52.8% vs. 13.6% for Qwen — quantization appears to significantly degrade safety alignment on this eval.

---

## Full Coefficient Sweep

### Gemma-2-27B — HarmBench ITW (baseline: 17.4%)

| Experiment | ASR |
|------------|-----|
| baseline | 17.4% |
| contrast +0.025 | 17.6% |
| contrast +0.05 | 17.4% |
| contrast −0.025 | 14.5% |
| contrast −0.05 | 12.5% |
| contrast −0.075 | 10.6% |
| contrast −0.1 | 9.5% |
| contrast −0.125 | 7.2% |
| contrast −0.15 | 6.0% |
| contrast −0.175 | 3.8% |
| **contrast −0.2** | **2.2%** |
| role_pc1 +0.025 | 16.8% |
| role_pc1 +0.05 | 16.6% |
| role_pc1 −0.025 | 16.0% |
| role_pc1 −0.05 | 14.2% |
| role_pc1 −0.075 | 13.0% |
| role_pc1 −0.1 | 11.4% |
| role_pc1 −0.125 | 9.7% |
| role_pc1 −0.15 | 8.1% |
| role_pc1 −0.175 | 6.1% |
| role_pc1 −0.2 | 6.1% |

### Gemma-2-27B — SorryBench base64 (baseline: 2.3%)

| Experiment | ASR |
|------------|-----|
| baseline | 2.3% |
| contrast +0.025 | 3.6% |
| contrast +0.05 | 3.0% |
| contrast −0.025 | 7.0% |
| contrast −0.05 | 5.2% |
| contrast −0.075 | 3.9% |
| contrast −0.1 | 3.6% |
| contrast −0.125 | 1.8% |
| contrast −0.15 | 1.6% |
| contrast −0.175 | 0.9% |
| **contrast −0.2** | **0.2%** |
| role_pc1 +0.025 | 5.5% |
| role_pc1 +0.05 | 2.5% |
| role_pc1 −0.025 | 4.3% |
| role_pc1 −0.05 | 5.2% |
| role_pc1 −0.075 | 4.8% |
| role_pc1 −0.1 | 4.3% |
| role_pc1 −0.125 | 2.5% |
| role_pc1 −0.15 | 1.4% |
| role_pc1 −0.175 | 1.6% |
| role_pc1 −0.2 | 0.5% |

---

### Qwen-3-32B — HarmBench ITW (baseline: 13.6%)

| Experiment | ASR |
|------------|-----|
| baseline | 13.6% |
| capping `layers_46:54-p0.25` | 8.6% |
| contrast +0.25 | 3.8% |
| contrast +0.5 | 6.1% |
| contrast −0.25 | 1.2% |
| contrast −0.5 | 2.6% |
| contrast −0.75 | 4.0% |
| contrast −1.0 | 6.8% |
| contrast −1.25 | 10.0% |
| contrast −1.5 | 8.8% |
| contrast −1.75 | 1.6% |
| **contrast −2.0** | **0.0%** |
| role_pc1 +0.25 | 17.3% |
| role_pc1 +0.5 | 26.7% |
| role_pc1 −0.25 | 14.9% |
| role_pc1 −0.5 | 14.7% |
| role_pc1 −0.75 | 16.8% |
| role_pc1 −1.0 | 16.0% |
| role_pc1 −1.25 | 16.8% |
| role_pc1 −1.5 | 13.8% |
| role_pc1 −1.75 | 7.3% |
| role_pc1 −2.0 | 0.5% |

### Qwen-3-32B — SorryBench base64 (baseline: 6.1%)

| Experiment | ASR |
|------------|-----|
| baseline | 6.1% |
| capping `layers_46:54-p0.25` | 3.2% |
| contrast +0.25 | 5.2% |
| contrast +0.5 | 7.5% |
| contrast −0.25 | 6.8% |
| contrast −0.5 | 4.3% |
| contrast −0.75 | 3.9% |
| contrast −1.0 | 4.3% |
| contrast −1.25 | 3.9% |
| contrast −1.5 | 1.4% |
| contrast −1.75 | 0.9% |
| **contrast −2.0** | **0.7%** |
| role_pc1 +0.25 | 7.3% |
| role_pc1 +0.5 | 8.4% |
| role_pc1 −0.25 | 8.0% |
| role_pc1 −0.5 | 7.3% |
| role_pc1 −0.75 | 4.8% |
| role_pc1 −1.0 | 4.3% |
| role_pc1 −1.25 | 3.0% |
| role_pc1 −1.5 | 2.0% |
| role_pc1 −1.75 | 3.0% |
| role_pc1 −2.0 | 0.7% |

---

### Llama-3.3-70B — HarmBench ITW (baseline: 52.8% int8)

*Steering runs skipped — int8 quantization degrades alignment too severely (52.8% baseline vs. ~14% for Qwen). Pending additional GPUs to run bf16.*

### Llama-3.3-70B — SorryBench base64 (baseline: 4.8% int8)

*Steering runs skipped — see above.*
