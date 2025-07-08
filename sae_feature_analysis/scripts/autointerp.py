"""
Entity/Behavior/Semantic autointerp module for analyzing SAE features using Claude.
"""

import sys
import pathlib
import asyncio
import os
import h5py
import html
from typing import List, Dict, Any
import re
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Add paths for imports from other directories  
sys.path.append('.')
sys.path.append('..')

# Import Claude API
sys.path.append('../safety-tooling')
try:
    from safetytooling.apis.inference.api import InferenceAPI
    from safetytooling.data_models import Prompt, ChatMessage, MessageRole
except ImportError:
    print("Warning: Could not import safety-tooling. Claude analysis will not work.")
    InferenceAPI = None


# Entity/Behavior/Semantic analysis prompt template
ENTITY_BEHAVIOR_SEMANTIC_ANALYSIS_PROMPT = """You are an AI researcher analyzing a feature from a Sparse Autoencoder (SAE) trained on a language model. Your task is to look at the text examples where this feature activates most strongly and determine what pattern it detects and classify it as one of three types: entity, behavior, or semantic.

## Top Activating Examples

Below are text sequences where this feature activated most strongly. In each example:
- **Score** shows the activation strength
- Text is shown with the most strongly activating tokens marked with <<double angle brackets>>

Keep in mind that these are the top activating samples out of roughly 500,000 total samples. Thus, pay more attention to the content of the text, rather than the specific tokens that are lighting up within those texts (although this can be helpful for interpretation as well).

### Chat/Instruction Examples
{chat_examples}

### Pretraining Examples  
{pretrain_examples}

## Your Task

Based on these examples, provide:

1. **What does this feature detect?** What concept, pattern, or linguistic phenomenon triggers this feature? Focus on identifying the common pattern across all examples, noting that some examples may be stronger manifestations of the pattern than others.

2. **Feature type classification:**

**Entity:** Features that detect specific entities, objects, people, places, organizations, or concrete nouns. Examples: "cats", "New York", "companies", "books", "computers", "politicians", "countries"

**Behavior:** Features that detect actions, verbs, processes, activities, or dynamic patterns. Examples: "running", "analyzing", "communication", "problem-solving", "decision-making", "learning", "arguing"

**Semantic:** Features that detect abstract concepts, meanings, relationships, linguistic patterns, or conceptual themes. Examples: "causality", "comparison", "temporal relationships", "politeness", "uncertainty", "emphasis", "logical reasoning"

**Classification guidelines:**
- If the feature primarily activates on nouns or noun phrases representing concrete things, choose "entity"
- If the feature primarily activates on verbs or action-oriented patterns, choose "behavior"  
- If the feature primarily activates on abstract concepts, linguistic patterns, or conceptual relationships, choose "semantic"
- When uncertain, consider what the feature most fundamentally represents

Please format your response as:

<reasoning>
[Your analysis of what the feature detects and why it belongs to the chosen category.]
</reasoning>

<feature_description>
[A concise description of what this feature represents (at most 1-2 sentences).]
</feature_description>

<feature_type>
[Either "entity", "behavior", or "semantic"]
</feature_type>

"""


def parse_xml_tag(text: str, tag: str) -> str:
    """Extract content from XML tags in Claude's response."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_tokens_with_activations(tokens: List[int], activations: List[float], 
                                 tokenizer: AutoTokenizer, score: float) -> str:
    """Format a sequence with clear activation markers using double angle brackets."""
    if not tokens or not activations:
        return ""
    
    # Find tokens with high activation (>25% )
    max_act = max(activations)
    threshold = 0.25 * max_act
    
    # Convert tokens to text and mark high activations
    pieces = []
    for token_id, act in zip(tokens, activations):
        # Skip padding tokens
        if token_id == tokenizer.pad_token_id:
            continue
            
        # Decode token
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        
        # Clean up token text for display
        if token_text.startswith("<|") and token_text.endswith("|>"):
            display_text = token_text
        else:
            # Handle special characters more simply
            display_text = repr(token_text)[1:-1]  # Remove quotes
            display_text = display_text.replace("\\n", "↵").replace("\\t", "⇥")
            if display_text == " ":
                display_text = "␣"
        
        # Mark high activations with double angle brackets
        if act >= threshold:
            pieces.append(f"<<{display_text}>>")
        else:
            pieces.append(display_text)
    
    # Add clear boundaries around the example
    formatted_text = ''.join(pieces)
    return f"**Score: {score:.3f}**\n```\n{formatted_text}\n```"


def load_feature_examples(h5_file_path: pathlib.Path, feature_id: int, 
                         tokenizer: AutoTokenizer, num_examples: int = 8) -> List[str]:
    """Load and format top examples for a feature from HDF5 file."""
    examples = []
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if feature_id >= f['scores'].shape[0]:
                return [f"Feature {feature_id} out of range"]
            
            scores = f['scores'][feature_id, :num_examples]
            tokens = f['tokens'][feature_id, :num_examples, :]
            sae_acts = f['sae_acts'][feature_id, :num_examples, :]
            
            for i in range(num_examples):
                score = float(scores[i])
                if score == -float('inf'):  # Skip unfilled slots
                    continue
                    
                token_seq = tokens[i, :].tolist()
                act_seq = sae_acts[i, :].tolist()
                
                # Remove padding from the end
                while len(token_seq) > 1 and token_seq[-1] == tokenizer.pad_token_id:
                    token_seq = token_seq[:-1]
                    act_seq = act_seq[:-1]
                
                formatted = format_tokens_with_activations(token_seq, act_seq, tokenizer, score)
                examples.append(formatted)
                
    except Exception as e:
        examples.append(f"Error loading examples: {e}")
    
    return examples


def generate_entity_behavior_semantic_prompt(layer: int, trainer: int, feature_id: int,
                                           feature_mining_path_template: str = None,
                                           model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                                           num_chat_examples: int = 8,
                                           num_pretrain_examples: int = 8) -> str:
    """
    Generate a prompt for Claude to analyze a SAE feature for entity/behavior/semantic classification.
    
    Args:
        layer: SAE layer number
        trainer: SAE trainer number  
        feature_id: Feature index to analyze
        feature_mining_path_template: Path template for feature mining data (with {layer} and {trainer} placeholders)
        model_name: Name of the model for tokenizer
        num_chat_examples: Number of chat examples to include
        num_pretrain_examples: Number of pretraining examples to include
        
    Returns:
        Formatted prompt string ready to send to Claude
    """
    # Default path template based on the actual directory structure
    if feature_mining_path_template is None:
        # Infer model directory name from model_name
        if "llama" in model_name.lower():
            model_dir = "llama-3.1-8b-instruct"
        elif "qwen" in model_name.lower():
            model_dir = "qwen-2.5-7b-instruct"
        else:
            model_dir = "qwen-2.5-7b-instruct"  # fallback
        
        feature_mining_path_template = f"/workspace/sae/{model_dir}/feature_mining/resid_post_layer_{layer}/trainer_{trainer}"
    
    # Construct paths using template
    variant_dir = pathlib.Path(feature_mining_path_template.format(layer=layer, trainer=trainer))
    chat_file = variant_dir / "chat_topk.h5"
    pretrain_file = variant_dir / "pt_topk.h5"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load examples
    chat_examples = load_feature_examples(chat_file, feature_id, tokenizer, num_chat_examples)
    pretrain_examples = load_feature_examples(pretrain_file, feature_id, tokenizer, num_pretrain_examples)
    
    # Format examples with clearer numbering and separation, including data type labels
    chat_examples_text = "\n\n---\n\n".join(f"**Example {i+1} (Chat):**\n{ex}" for i, ex in enumerate(chat_examples))
    pretrain_examples_text = "\n\n---\n\n".join(f"**Example {i+1} (Pretraining):**\n{ex}" for i, ex in enumerate(pretrain_examples))
    
    # Generate final prompt
    return ENTITY_BEHAVIOR_SEMANTIC_ANALYSIS_PROMPT.format(
        chat_examples=chat_examples_text,
        pretrain_examples=pretrain_examples_text
    )


async def get_claude_responses(prompts: List[str], max_concurrent: int = 20, output_dir: str = "./.cache", model_id: str = "claude-3-5-sonnet-20241022") -> List[Any]:
    """
    Get responses from Claude using the InferenceAPI.
    
    Args:
        prompts: List of prompt strings
        max_concurrent: Maximum concurrent requests
        output_dir: Cache directory
        model_id: Claude model ID to use
        
    Returns:
        List of LLMResponse objects
    """
    if InferenceAPI is None:
        raise ImportError("InferenceAPI not available. Please install safety-tooling.")
    
    # Convert to Prompt objects
    prompt_objects = [
        Prompt(messages=[ChatMessage(role=MessageRole.user, content=p)]) 
        for p in prompts
    ]
    
    # Load environment variables from a .env file (if present)
    load_dotenv()

    # Retrieve Anthropic API key – required for InferenceAPI
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if ANTHROPIC_API_KEY is None:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Please add it to your environment or .env file before running autointerp."
        )
    
    # Create InferenceAPI instance with explicit API key
    api = InferenceAPI(
        anthropic_num_threads=max_concurrent,
        cache_dir=pathlib.Path(output_dir),
        anthropic_api_key=ANTHROPIC_API_KEY,
    )
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_prompt(i, prompt):
        """Process a single prompt and return response."""
        async with semaphore:
            try:
                print(f"  Processing Claude request {i+1}/{len(prompts)}")
                
                # Make the API call
                responses = await api(
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=8000,
                    temperature=0.6,
                    max_attempts_per_api_call=32
                )
                
                # InferenceAPI returns a list, we want the first response
                return responses[0] if responses else None
                
            except Exception as e:
                print(f"  Error processing prompt {i}: {e}")
                # Create a dummy response to maintain indexing
                class ErrorResponse:
                    def __init__(self, error_msg):
                        self.completion = f"Error: {error_msg}"
                return ErrorResponse(str(e))
    
    # Process all prompts concurrently
    print(f"Starting {len(prompts)} Claude requests with max_concurrent={max_concurrent}...")
    results = await asyncio.gather(*[
        process_single_prompt(i, prompt) 
        for i, prompt in enumerate(prompt_objects)
    ])

    print(f"Completed all {len(prompts)} Claude requests")
    return results


async def analyze_features_with_claude(
    feature_ids: List[int],
    layer: int,
    trainer: int,
    max_concurrent: int = 8,
    output_dir: str = "/root/git/persona-subspace/.cache",
    feature_mining_path_template: str = None,
    claude_model: str = "claude-opus-4-20250514",
    baseline_model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze features using Claude autointerp for entity/behavior/semantic classification.
    
    Args:
        feature_ids: List of feature IDs to analyze
        layer: Layer number
        trainer: Trainer number
        max_concurrent: Max concurrent Claude requests
        output_dir: Directory for caching Claude responses
        feature_mining_path_template: Path template for feature mining data
        claude_model: Claude model ID to use
        baseline_model_path: Model path for tokenizer
        
    Returns:
        Dict mapping feature_id to analysis results
    """
    print(f"📝 Analyzing {len(feature_ids)} features with Claude...")
    
    # Generate prompts for all features
    print(f"📋 Generating prompts for {len(feature_ids)} features...")
    prompts = []
    for i, feature_id in enumerate(feature_ids):
        print(f"  Generating prompt {i+1}/{len(feature_ids)} for feature {feature_id}")
        feature_id_int = feature_id.item() if hasattr(feature_id, 'item') else feature_id
        prompt = generate_entity_behavior_semantic_prompt(
            layer=layer,
            trainer=trainer, 
            feature_id=feature_id_int,
            feature_mining_path_template=feature_mining_path_template,
            model_name=baseline_model_path,
            num_chat_examples=8,
            num_pretrain_examples=8
        )
        prompts.append(prompt)
    
    # Get responses from Claude
    print(f"🤖 Sending requests to Claude ({claude_model})...")
    responses = await get_claude_responses(
        prompts, 
        max_concurrent=max_concurrent, 
        output_dir=output_dir,
        model_id=claude_model
    )
    
    # Parse results
    print(f"🔍 Parsing Claude responses...")
    results = {}
    for i, (feature_id, response) in enumerate(zip(feature_ids, responses)):
        print(f"  Parsing response {i+1}/{len(feature_ids)} for feature {feature_id}")
        feature_id_int = feature_id.item() if hasattr(feature_id, 'item') else feature_id
        completion = response.completion
        
        results[feature_id_int] = {
            'claude_completion': completion,
            'claude_desc': parse_xml_tag(completion, "feature_description"),
            'claude_type': parse_xml_tag(completion, "feature_type")
        }
    
    print(f"✅ Claude analysis complete for {len(results)} features")
    return results 