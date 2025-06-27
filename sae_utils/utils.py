import torch

from transformers import AutoModelForCausalLM

@torch.no_grad()
def collect_activations(model, submodule, inputs):
    """
    Run the transformer **until** `submodule` (layer `LAYER_INDEX`),
    capture its *output* (residual stream), and abort the forward pass early
    to save compute + memory.
    """
    acts = None

    def hook(_, __, output):
        nonlocal acts
        acts = output[0] if isinstance(output, tuple) else output  # [B,L,D]
        raise StopForward

    class StopForward(Exception):
        pass

    handle = submodule.register_forward_hook(hook)
    try:
        _ = model(**inputs)           # the hook raises StopForward at layer L
    except StopForward:
        pass
    finally:
        handle.remove()

    return acts

def get_submodule(model: AutoModelForCausalLM, layer: int):
    """Gets the residual stream submodule"""
    model_name = model.name_or_path

    if model.config.architectures[0] == "GPTNeoXForCausalLM":
        return model.gpt_neox.layers[layer]
    elif (
        model.config.architectures[0] == "Qwen2ForCausalLM"
        or model.config.architectures[0] == "Gemma2ForCausalLM"
        or model.config.architectures[0] == "LlamaForCausalLM"
    ):
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")