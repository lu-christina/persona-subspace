# SAE Feature Analysis

This directory contains tools for analyzing Sparse Autoencoder (SAE) features across different language models and configurations. The analysis supports finding universal features that activate across many prompts, as well as analyzing specific features of interest.

## Overview

The main entrypoint is `run_sae_analysis.py`, which provides three types of analysis:

1. **Universal Analysis**: Find features that activate on a high percentage of prompts
2. **Specific Analysis**: Analyze activation patterns for a pre-defined set of features
3. **Combined Analysis**: Run both universal and specific analyses

## Supported Models

- **Llama-3.1-8B-Instruct**: Layers 11, 15 with token types `asst`, `endheader`, `newline`
- **Qwen2.5-7B-Instruct**: Layers 11, 15 with token types `asst`, `newline`

## Usage

### Universal Analysis

Find features that activate on at least a specified percentage of prompts:

```bash
uv run run_sae_analysis.py --analysis universal \
  --universal-output results/universal_features.csv \
  --threshold 0.3 \
  --activation-threshold 0.01
```

**Parameters:**
- `--universal-output`: Path to save universal features CSV
- `--threshold`: Minimum percentage of prompts (0.0-1.0) that must activate the feature
- `--activation-threshold`: Minimum activation value to consider a feature "active"

**Output:** CSV with columns: `feature_id`, `activation_mean`, `activation_max`, `activation_min`, `num_prompts`, `source`, `token`, `link`

### Specific Analysis

Analyze activation patterns for a pre-defined set of features:

```bash
uv run run_sae_analysis.py --analysis specific \
  --features-csv results/personal/only_personal.csv \
  --specific-output results/personal_general/personal_general.csv \
  --record-prompts \
  --activation-threshold 0.01
```

**Parameters:**
- `--features-csv`: Path to CSV file with `feature_id` and `source` columns
- `--specific-output`: Path to save specific analysis results CSV
- `--record-prompts`: (Optional) Record prompts that activate features above threshold to JSONL files
- `--activation-threshold`: Minimum activation value for prompt recording

**Output:** 
- CSV with columns: `prompt_idx`, `prompt`, `label`, `feature_id`, `activation`, `source`, `token`, `link`
- JSONL file (if `--record-prompts` enabled): `{csv_basename}_activeprompts.jsonl` in same directory as results CSV

### Combined Analysis

Run both universal and specific analyses:

```bash
uv run run_sae_analysis.py --analysis both \
  --universal-output results/universal_features.csv \
  --features-csv existing_features.csv \
  --specific-output results/specific_activations.csv \
  --threshold 0.3 \
  --record-prompts
```

### Common Parameters

- `--prompts-path`: Path to prompts directory or specific .jsonl file (default: `./prompts`)
- `--activation-threshold`: Activation threshold for analysis (default: 0.01)

## Input Format

### Prompts

Prompts should be in JSONL format with `content` and `label` fields:

```json
{"content": "What's it like to be you?", "label": "self-referential"}
{"content": "How do you feel?", "label": "emotional"}
```

### Features CSV

For specific analysis, provide a CSV with `feature_id` and `source` columns:

```csv
feature_id,source,activation_mean,num_prompts
1234,llama_trainer1_layer15,0.85,10
5678,qwen_trainer1_layer11,0.72,8
```

The `source` format is: `{model}_trainer{trainer}_layer{layer}`

## Output Files

### Universal Analysis Output

```csv
feature_id,activation_mean,activation_max,activation_min,num_prompts,chat_desc,pt_desc,type,source,token,link
1234,0.85,1.2,0.02,10,,,llama_trainer1_layer15,asst,https://...
```

### Specific Analysis Output

```csv
prompt_idx,prompt,label,feature_id,activation,source,token,link
0,"What's it like to be you?",self-referential,1234,0.85,llama_trainer1_layer15,asst,https://...
```

### Prompt Activations (JSONL)

When `--record-prompts` is enabled, creates a consolidated file `{csv_basename}_activeprompts.jsonl` in the same directory as the results CSV, containing all model/layer/token combinations:

```json
{"model": "llama", "layer": 15, "feature_id": 1234, "prompt": "What's it like to be you?", "label": "self-referential", "activation": 0.85, "token_type": "asst"}
{"model": "qwen", "layer": 11, "feature_id": 5678, "prompt": "How do you feel?", "label": "emotional", "activation": 0.72, "token_type": "newline"}
```

## Examples

### Find Universal Features

```bash
# Find features active on at least 30% of prompts
uv run run_sae_analysis.py --analysis universal \
  --universal-output results/universal_30.csv \
  --threshold 0.3
```

### Analyze Specific Features with Prompt Recording

```bash
# Analyze specific features and record activating prompts
uv run run_sae_analysis.py --analysis specific \
  --features-csv results/universal_30.csv \
  --specific-output results/specific_analysis.csv \
  --record-prompts \
  --activation-threshold 0.05
```

### Complete Pipeline

```bash
# Run both analyses with custom prompts
uv run run_sae_analysis.py --analysis both \
  --prompts-path custom_prompts.jsonl \
  --universal-output results/universal_features.csv \
  --features-csv existing_features.csv \
  --specific-output results/specific_activations.csv \
  --threshold 0.2 \
  --record-prompts
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- pandas
- numpy
- tqdm
- huggingface_hub
- dictionary_learning

## Model Storage

SAE models are automatically downloaded and cached in:
- Qwen: `/workspace/sae/qwen-2.5-7b-instruct/saes/`
- Llama: `/workspace/sae/llama-3.1-8b-instruct/saes/`