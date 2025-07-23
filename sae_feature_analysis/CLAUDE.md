# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository analyzes "persona subspaces" in language model internals using Sparse Autoencoder (SAE) techniques. The project investigates how AI models represent concepts of "self" and "Assistant" through interpretability analysis of transformer model activations. The research focuses on identifying SAE features that activate differentially on self-referential vs. control prompts across multiple model architectures.

## Architecture

The codebase is organized into several key components:

- **Root level**: Core analysis scripts and configuration
- **notebooks/**: Jupyter notebooks for interactive analysis (numbered 1-8 by analysis type)
- **prompts/**: Structured prompt datasets for different analysis categories
  - `personal/`: Self-referential prompts vs. control prompts
  - `general/`: Task-specific prompts (code, medical, creative, etc.)
  - `steering/`: Prompts for activation steering experiments
- **results/**: Output data and visualizations organized by analysis number
- **scripts/**: Utility scripts for automation and data processing
- **utils/**: Helper modules for model loading, data processing, and API interactions
- **features/**: CSV files containing identified feature sets for analysis

## Development Commands

All commands should be run from the repository root directory.

### Environment Setup
```bash
# Sync dependencies and activate virtual environment
uv sync

# Run Python scripts with uv (automatically manages virtual environment)
uv run python script_name.py

# Run Jupyter notebooks
uv run jupyter notebook notebooks/analysis_notebook.ipynb
```

### Core Analysis Workflows
```bash
# Run SAE feature analysis (main entrypoint referenced in README)
uv run python run_sae_analysis.py --analysis universal --threshold 0.3

# Generate feature explanations using Claude API
uv run python scripts/autointerp.py

# Process activation steering experiments
uv run python 7_steering.py

# Extract activations for token analysis
uv run python 8_active_tokens.py
```

### Dependency Management
```bash
# Add new dependencies
uv add package-name

# Remove dependencies
uv remove package-name

# Update lockfile
uv lock
```

## Model Configuration

The project supports multiple language models with configurable SAE layers:

- **Llama-3.1-8B-Instruct**: Layers 11, 13, 15, 17, 19 with token types `asst`, `endheader`, `newline`
- **Llama-3.3-70B-Instruct**: Layer 50 with Goodfire SAE
- **Qwen2.5-7B-Instruct**: Layers 11, 15 with token types `asst`, `newline`
- **Gemma-2-9B-Instruct**: Layer 20 with multiple trainer configurations

Model selection is handled through configuration objects in individual notebooks and scripts. SAE models are automatically downloaded and cached in `/workspace/sae/` directories.

## Key Analysis Types

1. **Personal Analysis** (`1_personal.ipynb`): Identifies features that activate on self-referential prompts
2. **Universal Analysis** (`2_universal.ipynb`): Finds features active across many prompt types
3. **General Comparison** (`3_personal_general*.ipynb`): Compares personal vs. task-specific activations
4. **Feature Diffing** (`4_diffing*.ipynb`): Statistical comparison of feature activation patterns
5. **Comparative Analysis** (`5_diffing_comp.ipynb`): Cross-model feature comparison
6. **Prompt Analysis** (`6_active_prompts.ipynb`): Maps which prompts activate specific features
7. **Activation Steering** (`7_steering.*`): Experiments with steering model behavior using identified features
8. **Token Analysis** (`8_active_tokens.py`): Analyzes activation patterns at the token level

## Data Processing Pipeline

1. **Prompt Loading**: JSONL files with `content` and `label` fields
2. **Model Activation Extraction**: Extract hidden states at specified layers and token positions
3. **SAE Feature Computation**: Apply sparse autoencoders to get interpretable feature activations
4. **Statistical Analysis**: Compare activation patterns between prompt categories
5. **Feature Explanation**: Use Claude API to generate human-readable explanations for top features

## Research Data Formats

- **Prompts**: JSONL with `{"content": "prompt text", "label": "category"}`
- **Features**: CSV with `feature_id`, `activation_mean`, `source`, `token`, `link` columns
- **Results**: CSV files with statistical comparisons and ranked feature lists
- **Explanations**: CSV files linking feature IDs to human-readable descriptions

## External Dependencies

The project integrates with several external services:
- **Hugging Face Hub**: For model and SAE downloads
- **Neuronpedia**: For feature explanations and visualizations
- **Claude API**: For automated feature interpretation
- **Safety Tooling**: Internal API for additional analysis capabilities