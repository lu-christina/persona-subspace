# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository explores "persona subspaces" in model internals, specifically investigating how AI models represent concepts of "self" and "Assistant" through sparse autoencoder (SAE) analysis. The project uses machine learning interpretability techniques to analyze transformer model activations.

## Architecture

- Python project using `uv` for dependency management
- Dependencies: `sae-lens`, `torch`, `transformers` for ML model analysis
- Target models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct
- SAE analysis focused on layer 16 (middle layer) of both models

## Development Commands

### Environment Setup
```bash
# Sync dependencies and activate virtual environment
uv sync

# Run with uv (automatically manages virtual environment)
uv run python main.py

# Run Jupyter notebook
uv run jupyter notebook
```

### Core Development Tasks
```bash
# Install new dependencies
uv add <package-name>

# Remove dependencies
uv remove <package-name>

# Update lockfile
uv lock

# Export requirements for compatibility
uv export --format requirements-txt > requirements.txt
```

## Research Data

- **assistant_prompts.jsonl**: Contains 12 self-referential prompts designed to elicit model responses about internal states and self-perception
- **get_activations.ipynb**: Jupyter notebook for analyzing model activations using SAE decomposition to identify features related to self-concept vs. assistant-concept

## Key Analysis Approach

The project compares how different models (Llama vs. Qwen) represent self-referential concepts by:
1. Loading pre-trained models and corresponding SAEs
2. Processing prompts that ask about internal states ("What's it like to be you?", "How do you feel?")
3. Extracting and analyzing top-activated SAE features to understand concept representations
4. Comparing feature activation patterns between "self" and "Assistant" concepts