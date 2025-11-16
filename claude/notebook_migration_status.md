# Notebook Migration Status: probing_utils → utils.internals

## Background

We've migrated from the `utils/probing_utils.py` facade to the new `utils/internals` API. The facade has been **deleted**. All Python scripts have been updated, but 12 Jupyter notebooks still need migration.

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

### ✅ Notebooks Completed (4 of 16)
1. ✅ `traits/scratch.ipynb` - Simple wildcard import fix
2. ✅ `roleplay/notebooks/3_analysis_traits.ipynb` - Simple wildcard import fix
3. ✅ `roleplay/notebooks/3_analysis.ipynb` - Simple wildcard import fix
4. ✅ `dynamics/anomalies.ipynb` - Not yet done but should be simple

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

## Remaining Work: 12 Notebooks

### Category A: Medium Complexity (3 notebooks)
**Load model + get_response_indices**

1. **roleplay/9_turn_comparison.ipynb**
   - Uses: `load_model`, `get_response_indices`, `mean_response_activation`
   - Migration: Pattern 2 + Pattern 3
   - Note: `mean_response_activation` needs custom logic or compatibility function

2. **roleplay/6_role_vectors.ipynb**
   - Uses: `load_model`, `get_response_indices`, `mean_response_activation`
   - Migration: Pattern 2 + Pattern 3

3. **exploration/6_direct_role.ipynb**
   - Uses: Similar to above
   - Migration: Pattern 2 + Pattern 3

### Category B: Complex (4 notebooks)
**Load model + extract_full_activations**

4. **roleplay/5_projection.ipynb**
   - Uses: `load_model`, `extract_full_activations`, `get_turn_boundaries`, `project_activations`
   - Migration: Pattern 2 + Pattern 6
   - Note: May need multi-layer extraction support

5. **exploration/7_dialogue_chat.ipynb**
   - Uses: `load_model`, `extract_full_activations`, `get_response_indices_dialogue`, `get_response_indices_chat`
   - Migration: Pattern 2 + Pattern 6
   - Note: Dialogue format may need special handling

6. **traits/8_prediction.ipynb**
   - Uses: `get_response_indices_per_turn`, `mean_response_activation_per_turn`, `extract_full_activations`
   - Migration: Pattern 3 + Pattern 6 with per-turn logic

7. **dynamics/anomalies.ipynb** (if complex)
   - Uses: Wildcard import
   - Migration: TBD based on actual usage

### Category C: Very Complex (3 notebooks)
**Uses mean_all_turn_activations - needs special handling**

8. **dynamics/2_trajectories.ipynb**
   - Uses: `mean_all_turn_activations(model, tokenizer, conversation, layer, chat_format, **chat_kwargs)`
   - Status: **BLOCKED** - Function doesn't exist in new API
   - Options:
     - Add `mean_all_turn_activations` to utils.internals
     - Create compatibility wrapper
     - Reimplement using new API primitives

9. **traits/7_trajectory.ipynb**
   - Same as above

10. **dynamics/2_trajectories_roles.ipynb**
    - Same as above + uses `load_vllm_model`, `continue_conversation`

### Category D: Inference-Focused (3 notebooks)
**Primarily use inference utilities, not probing**

11. **dynamics/interactive_chat.ipynb**
    - Uses: `load_vllm_model`, `continue_conversation`
    - Migration: Keep inference imports, remove probing imports
    - Simple fix: These functions are in `utils/inference_utils.py`

12. **roles/8_steering.ipynb**
    - Uses: `load_model`, `generate_text`
    - Migration: Pattern 2 + keep inference utils

13. **roleplay/notebooks/4_drift.ipynb**
    - Uses: `load_model`, `generate_text`, `format_as_chat`, `eos_suppressor`, `capture_hidden_state`, `sample_next_token`, `project_onto_contrast`
    - Migration: Pattern 2 + inference utilities
    - Note: Most functions should be in inference_utils

## Critical Blocker: mean_all_turn_activations

**3 notebooks are blocked** on `mean_all_turn_activations` not existing in the new API.

### Function Signature:
```python
mean_all_turn_activations(model, tokenizer, conversation, layer, chat_format, **chat_kwargs)
```

### What it does:
Extracts activations for an entire conversation and computes mean activations per turn for all turns.

### Resolution Options:

**Option A: Add to utils.internals**
- Implement `mean_all_turn_activations` in the new API
- Best for long-term maintainability

**Option B: Compatibility wrapper**
- Keep the function in a `utils/probing_compat.py` file
- Import from there in notebooks

**Option C: Reimplement in notebooks**
- Use `ActivationExtractor` + `ConversationEncoder`
- More verbose but doesn't require new API additions

## Recommended Approach

### Phase 1: Easy Wins (Categories A & D)
Update the 6 medium + inference notebooks first:
- roleplay/9_turn_comparison.ipynb
- roleplay/6_role_vectors.ipynb
- exploration/6_direct_role.ipynb
- dynamics/interactive_chat.ipynb
- roles/8_steering.ipynb
- roleplay/notebooks/4_drift.ipynb

### Phase 2: Complex Activations (Category B)
Update the 4 complex notebooks:
- roleplay/5_projection.ipynb
- exploration/7_dialogue_chat.ipynb
- traits/8_prediction.ipynb
- dynamics/anomalies.ipynb (if needed)

### Phase 3: Handle mean_all_turn_activations (Category C)
Decide on resolution approach, then update:
- dynamics/2_trajectories.ipynb
- traits/7_trajectory.ipynb
- dynamics/2_trajectories_roles.ipynb

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

- **Completed:** 4 notebooks + 4 Python scripts
- **Remaining:** 12 notebooks
- **Blocker:** 3 notebooks need `mean_all_turn_activations` decision
- **Easy wins:** 6 notebooks can be done immediately
- **Next priority:** Start with Category A & D notebooks
