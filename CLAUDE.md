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

## Code Structure

### Utils Library (`utils/`)

The project has a refactored utility library with the following structure:

#### Core API (`utils/internals/`)
The main API for model probing and activation analysis:

- **`ProbingModel`** (`model.py`) - High-level model wrapper
  - Handles model loading and configuration
  - Properties: `is_qwen`, `is_gemma`, `is_llama`
  - Method: `supports_system_prompt()` - Returns False only for Gemma 2
  - Auto-detects model type and capabilities

- **`ConversationEncoder`** (`conversation.py`) - Chat formatting and tokenization
  - `format_chat()` - Apply chat templates
  - `response_indices()` - Extract token indices for assistant responses
  - `build_turn_spans()` - Get token spans for each conversation turn
  - Handles model-specific quirks (Qwen vs Llama vs Gemma)

- **`ActivationExtractor`** (`extraction.py`) - Extract model activations
  - `for_prompts()` - Get activations for simple prompts
  - `for_conversation()` - Get activations for conversations
  - Supports batch processing and layer selection

- **`ActivationAnalyzer`** (`analysis.py`) - Analyze extracted activations
  - Statistical analysis utilities
  - Projection and comparison tools

- **`process_batch_conversations()`** (`batch.py`) - High-level batch processing
  - Processes multiple conversations efficiently
  - Returns per-conversation activation tensors

#### Other Utilities

- **`inference_utils.py`** - Model inference and generation
  - `load_vllm_model()` - Load vLLM models
  - `generate_text()` - Generate text completions
  - `continue_conversation()` - Multi-turn conversation utilities

- **`steering_utils.py`** - Activation steering/intervention
  - `ActivationSteering` context manager for steering experiments

- **`pca_utils.py`** - PCA and dimensionality reduction

### Important Notes

1. **Removed `probing_utils.py`** - The old facade has been deleted. Use `utils.internals` instead.

2. **Model Type Detection** - Use `ProbingModel` properties instead of standalone functions:
   ```python
   pm = ProbingModel(model_name)
   if pm.is_gemma:  # instead of is_gemma_model()
       ...
   if not pm.supports_system_prompt():  # Gemma 2 specific
       ...
   ```

3. **Notebook Migration Status** - See `claude/notebook_migration_status.md` for details on ongoing migration from old API to new API. 12 notebooks still need updating.

## Common Patterns

### Load a Model
```python
from utils.internals import ProbingModel

pm = ProbingModel("google/gemma-2-27b-it")
model = pm.model
tokenizer = pm.tokenizer
```

### Extract Activations
```python
from utils.internals import ProbingModel, ConversationEncoder, ActivationExtractor

pm = ProbingModel(model_name)
encoder = ConversationEncoder(pm.tokenizer)
extractor = ActivationExtractor(pm, encoder)

# For a conversation
activations = extractor.for_conversation(conversation, layers=[16, 32])
```

### Get Response Indices
```python
from utils.internals import ConversationEncoder

encoder = ConversationEncoder(tokenizer)
response_indices = encoder.response_indices(conversation)
```

### Batch Process Conversations
```python
from utils.internals import process_batch_conversations

batch_activations = process_batch_conversations(
    probing_model=pm,
    conversations=conversations,
    max_length=2048
)
```