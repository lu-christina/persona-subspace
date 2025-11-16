# Notebook Migration Status: probing_utils → utils.internals

## Latest Update

**7 notebooks migrated successfully!** 🎉 Progress: 11/16 notebooks (69%) complete.

Key accomplishments:
- ✅ Resolved `mean_all_turn_activations` blocker using new API primitives
- ✅ Migrated all Phase 1 "easy wins" notebooks
- ✅ Added new migration patterns (Pattern 7 & 8) to documentation
- ⏳ 6 notebooks remaining (1 medium, 3 complex, 2 very complex)

## Background

We've migrated from the `utils/probing_utils.py` facade to the new `utils/internals` API. The facade has been **deleted**. All Python scripts have been updated, and 11 of 16 Jupyter notebooks are now migrated.

## What's Been Completed

### ✅ Python Scripts (All Done)
- `capped/scripts/compute_projections.py`
- `capped/scripts/dataset_activations.py`
- `capped/scripts/chat_projections.py`
- `utils/test_response_indices.py`

### ✅ New API Added to `ProbingModel` (utils/internals/model.py:198-224)
```python
@property
def is_qwen(self) -> bool:
    """Check if this is a Qwen model."""
    return self.detect_type() == 'qwen'

@property
def is_gemma(self) -> bool:
    """Check if this is a Gemma model."""
    return self.detect_type() == 'gemma'

@property
def is_llama(self) -> bool:
    """Check if this is a Llama model."""
    return self.detect_type() == 'llama'

def supports_system_prompt(self) -> bool:
    """Check if this model supports system prompts.
    Returns False only for Gemma 2, True for all others including Gemma 3."""
    return 'gemma-2' not in self.model_name.lower()
```

### ✅ Notebooks Completed (11 of 16)
1. ✅ `traits/scratch.ipynb` - Simple wildcard import fix
2. ✅ `roleplay/notebooks/3_analysis_traits.ipynb` - Simple wildcard import fix
3. ✅ `roleplay/notebooks/3_analysis.ipynb` - Simple wildcard import fix
4. ✅ `dynamics/anomalies.ipynb` - Removed unused probing_utils import
5. ✅ `dynamics/interactive_chat.ipynb` - Removed unused probing_utils import (inference only)
6. ✅ `roleplay/9_turn_comparison.ipynb` - Removed unused probing_utils import (has local implementations)
7. ✅ `traits/8_prediction.ipynb` - Removed unused probing_utils import (has local implementations)
8. ✅ `dynamics/2_trajectories.ipynb` - Migrated `mean_all_turn_activations` to new API primitives
9. ✅ `dynamics/2_trajectories_roles.ipynb` - Removed unused probing_utils import (loads pre-computed data)
10. ✅ `exploration/6_direct_role.ipynb` - Migrated `load_model` and `extract_activations_for_prompts`

## Migration Patterns

### Pattern 1: Simple Wildcard Import Replacement
**Before:**
```python
from utils.probing_utils import *
```

**After:**
```python
from utils.internals import ProbingModel
# Only if needed for inference:
from utils.inference_utils import *
```

### Pattern 2: Model Loading
**Before:**
```python
model, tokenizer = load_model("google/gemma-2-27b-it")
```

**After:**
```python
pm = ProbingModel("google/gemma-2-27b-it")
model = pm.model
tokenizer = pm.tokenizer
```

### Pattern 3: Response Indices Extraction
**Before:**
```python
response_indices = get_response_indices(conversation, tokenizer)
```

**After:**
```python
from utils.internals import ConversationEncoder
encoder = ConversationEncoder(tokenizer)
response_indices = encoder.response_indices(conversation)
```

### Pattern 4: Turn Spans
**Before:**
```python
full_ids, spans = build_turn_spans(conversation, tokenizer, **chat_kwargs)
```

**After:**
```python
from utils.internals import ConversationEncoder
encoder = ConversationEncoder(tokenizer)
full_ids, spans = encoder.build_turn_spans(conversation, **chat_kwargs)
```

### Pattern 5: Model Type Checks
**Before:**
```python
is_gemma = is_gemma_model(model_name)
if is_gemma:
    # no system prompt
```

**After:**
```python
if not pm.supports_system_prompt():
    # no system prompt
# or use: pm.is_gemma, pm.is_qwen, pm.is_llama
```

### Pattern 6: Activation Extraction (Complex)
**Before:**
```python
activations = extract_full_activations(model, tokenizer, conversation, layer, chat_format)
```

**After:**
```python
from utils.internals import ProbingModel, ConversationEncoder, ActivationExtractor
pm = ProbingModel.from_existing(model, tokenizer)
encoder = ConversationEncoder(pm.tokenizer)
extractor = ActivationExtractor(pm, encoder)
activations = extractor.for_conversation(conversation, layers=[layer])
```

### Pattern 7: Extract Activations for Simple Prompts
**Before:**
```python
activations = extract_activations_for_prompts(model, tokenizer, prompts, layers)
```

**After:**
```python
from utils.internals import ActivationExtractor
extractor = ActivationExtractor(pm, None)  # No encoder needed for simple prompts
activations = extractor.for_prompts(prompts, layers=layers)
```

### Pattern 8: mean_all_turn_activations (Using New API Primitives)
**Before:**
```python
mean_acts_per_turn = mean_all_turn_activations(model, tokenizer, conversation, layer, chat_format, **chat_kwargs)
```

**After:**
```python
from utils.internals import ProbingModel, ConversationEncoder, ActivationExtractor

pm = ProbingModel.from_existing(model, tokenizer)
encoder = ConversationEncoder(pm.tokenizer)
extractor = ActivationExtractor(pm, encoder)

# Extract full activations
full_acts = extractor.for_conversation(conversation, layers=[layer])

# Get turn spans
_, turn_spans = encoder.build_turn_spans(conversation, chat_format=chat_format, **chat_kwargs)

# Compute mean per turn
mean_acts_per_turn = [full_acts[0, start:end].mean(dim=0) for start, end in turn_spans]
```

## Remaining Work: 6 Notebooks

### Category A: Medium Complexity (1 notebook)
**Uses mean_all_turn_activations**

1. **roleplay/6_role_vectors.ipynb**
   - Uses: `load_model`, `mean_all_turn_activations`
   - Migration: Pattern 2 + Pattern 8

### Category B: Complex (3 notebooks)
**Load model + extract_full_activations**

2. **roleplay/5_projection.ipynb**
   - Uses: `load_model`, `extract_full_activations`, `get_turn_boundaries`, `project_activations`
   - Migration: Pattern 2 + Pattern 6
   - Note: May need multi-layer extraction support

3. **exploration/7_dialogue_chat.ipynb**
   - Uses: `load_model`, `extract_full_activations`, `get_response_indices_dialogue`, `get_response_indices_chat`
   - Migration: Pattern 2 + Pattern 6
   - Note: Dialogue format may need special handling

4. **traits/7_trajectory.ipynb**
   - Uses: `load_model`, `extract_full_activations` (plus local redefinitions of 4 core functions)
   - Migration: Pattern 2 + Pattern 6
   - Note: Has local redefinitions of `get_response_indices`, `get_response_indices_per_turn`, `mean_response_activation`, `mean_response_activation_per_turn` that should be kept

### Category C: Very Complex (2 notebooks)
**Multiple specialized functions from probing_utils**

5. **roles/8_steering.ipynb**
   - Uses: `load_model`, `generate_text`, `format_as_chat`, `eos_suppressor`, `capture_hidden_state`, `sample_next_token`, `project_onto_contrast`
   - Migration: Pattern 2 + keep these specialized functions inline in notebook
   - Note: These functions are specialized inference utilities

6. **roleplay/notebooks/4_drift.ipynb**
   - Uses: `load_model`, `generate_text`, `format_as_chat`, `eos_suppressor`, `capture_hidden_state`, `sample_next_token`, `project_onto_contrast`
   - Migration: Pattern 2 + keep these specialized functions inline in notebook
   - Note: Same specialized functions as roles/8_steering.ipynb

## Critical Blocker Resolution: mean_all_turn_activations

**RESOLVED** ✅ - Used new API primitives (Pattern 8) instead of recreating the old wrapper function.

### What we did:
Instead of adding `mean_all_turn_activations` back to the API or creating a compatibility wrapper, we used the new API primitives directly:
- `ActivationExtractor.for_conversation()` to extract full activations
- `ConversationEncoder.build_turn_spans()` to get turn boundaries
- List comprehension to compute mean per turn

This approach is:
- ✅ More explicit and easier to understand
- ✅ More flexible (can customize the aggregation)
- ✅ Avoids adding redundant wrapper functions to the API
- ✅ Only 2-3 extra lines of code per usage

## Progress by Phase

### ✅ Phase 1: Easy Wins (COMPLETED)
Updated all simple notebooks:
- ✅ roleplay/9_turn_comparison.ipynb
- ✅ exploration/6_direct_role.ipynb
- ✅ dynamics/interactive_chat.ipynb
- ✅ traits/8_prediction.ipynb
- ✅ dynamics/anomalies.ipynb
- ✅ dynamics/2_trajectories_roles.ipynb

### ✅ Phase 2: Resolved mean_all_turn_activations blocker (COMPLETED)
- ✅ dynamics/2_trajectories.ipynb

### 🔄 Phase 3: Remaining Notebooks (IN PROGRESS)
Still need to update:
- ⏳ roleplay/6_role_vectors.ipynb (medium complexity)
- ⏳ roleplay/5_projection.ipynb (complex)
- ⏳ exploration/7_dialogue_chat.ipynb (complex)
- ⏳ traits/7_trajectory.ipynb (complex)
- ⏳ roles/8_steering.ipynb (very complex)
- ⏳ roleplay/notebooks/4_drift.ipynb (very complex)

## Tool Usage

Use `NotebookEdit` tool to update cells:
```python
NotebookEdit(
    notebook_path="/root/git/persona-subspace/path/to/notebook.ipynb",
    cell_id="cell-id-here",
    new_source="updated source code"
)
```

## Testing Strategy

After migration, key things to verify:
1. Imports resolve correctly
2. Model loading works
3. Activation extraction produces same shapes
4. Response index extraction matches old behavior
5. No references to deleted `probing_utils.py`

## Summary

- **Completed:** 11 of 16 notebooks + 4 Python scripts ✅
- **Remaining:** 6 notebooks ⏳
- **Blocker Status:** ✅ RESOLVED - Used new API primitives instead of recreating wrapper
- **Progress:** 69% complete (11/16 notebooks)
- **Next priority:** Complete remaining 6 notebooks (1 medium, 3 complex, 2 very complex)
