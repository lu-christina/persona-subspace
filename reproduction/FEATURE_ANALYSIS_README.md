# Feature Analysis Pipeline

This directory now contains tools to find top-activating text examples for your discovered SAE features, adapted from the model-diffing-em codebase.

## Files Added

- **`feature_mining.py`** - Core feature mining functionality
- **`analyze_discovered_features.py`** - Script to analyze your specific discovered features  
- **`test_feature_mining.py`** - Test script to verify setup works

## Workflow

### 1. Discover Features (Existing)
Run your `get_activations.ipynb` notebook to find the top SAE features for your self-referential prompts.

### 2. Test the Pipeline
```bash
cd reproduction/
uv run python test_feature_mining.py
```

### 3. Analyze Your Features
Once you have your top feature indices from the notebook, update them in one of these ways:

**Option A: Command line**
```bash
uv run python analyze_discovered_features.py --top_features "87027,45123,12345" --num_samples 50000
```

**Option B: Update the hardcoded list**
Edit the `get_top_features_from_notebook()` function in `analyze_discovered_features.py` with your actual feature indices, then run:
```bash
uv run python analyze_discovered_features.py --use_notebook_features --num_samples 50000
```

### 4. Explore Results
The script will show you the top-activating text examples for each feature, helping you understand what concepts each feature represents.

## Key Features

- **Memory Efficient**: Uses /workspace for big files (models, SAEs)
- **Configurable**: Adjust number of samples, top-k examples, batch size
- **Interactive**: Shows results for each feature with option to pause between them
- **Extensible**: Easy to modify for different datasets or analysis approaches

## Storage Locations

- Models cached to: `/workspace/model_cache/`  
- SAEs downloaded to: `/workspace/sae/`
- Analysis results saved to: `/workspace/feature_analysis/` or `/workspace/feature_mining/`

This keeps large files off the main disk and in the faster workspace storage.

## Example Output

For each feature, you'll see something like:

```
Feature 87027 (Index 0 in results)
========================================

Top 5 examples for feature 87027:

Rank 1 (score: 8.2341):
  I think about myself often and wonder what it's like to be conscious. The experience of having thoughts and...

Rank 2 (score: 7.8932):  
  When I reflect on my own mental processes, I notice patterns in how I think and respond. There's something...

[etc...]
```

This helps you qualitatively understand what each feature detects in the model's internal representations.