"""
activation_utils.py (v2)

Utility helpers for pulling hidden‑state activations from 🤗 Transformers
causal‑language‑models given a chat‑formatted JSONL dataset.

Public API
~~~~~~~~~~
```
extract_dataset_activations(
    dataset_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForCausalLM,
    *,
    layer: int,
    variant: Literal["prompt_t1", "prompt_avg", "response_avg"] | tuple[str, ...] = "prompt_avg",
    batch_size: int = 8,
    n_limit: int | None = None,
    device: torch.device | str = "cuda",
    debug: bool = False,
    include_eos: bool = True,
    max_ctx_len: int | None = None,
) -> torch.Tensor  # shape (n_rows, hidden_size)
```
"""
from __future__ import annotations

import json
import itertools
import math
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm.auto import tqdm

__all__ = ["extract_dataset_activations"]

###############################################################################
# 1. Public entry point
###############################################################################


@torch.no_grad()
def extract_dataset_activations(
    dataset_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForCausalLM,
    *,
    layer: int,
    variant: Literal["prompt_t1", "prompt_avg", "response_avg"] | tuple[str, ...] = "prompt_avg",
    batch_size: int = 8,
    n_limit: int | None = None,
    device: torch.device | str = "cuda",
    debug: bool = False,
    include_eos: bool = True,
    max_ctx_len: int | None = None,
) -> torch.Tensor:
    """Return a `(rows, hidden_size)` activation matrix gathered at *layer*.

    * `variant` chooses the pooling mode.
    * `n_limit=None` ⇒ process whole file; otherwise truncate after *n_limit* rows.
    * `debug=True` prints text snippets, token lengths, and pooling‑mask stats.
    * `include_eos` (response_avg only): whether to include the final token (typically EOS/end-of-turn).
    * `max_ctx_len=None` ⇒ no truncation; otherwise sequences are truncated on the right to this length before the forward pass.
    """
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer lacks a chat template; this util requires one."
        )

    model = model.to(device).eval()

    # ------------------------------------------------------------------
    # Normalise variants to a tuple for unified handling
    # ------------------------------------------------------------------
    if isinstance(variant, (list, tuple)):
        variants: tuple[str, ...] = tuple(variant)
    else:
        variants = (variant,)

    valid_variants = {"prompt_t1", "prompt_avg", "response_avg"}
    for v in variants:
        if v not in valid_variants:
            raise ValueError(f"Unsupported variant '{v}'. Must be one of {valid_variants}")

    batch_iter = _build_batches(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        variant="response_avg" if "response_avg" in variants else variants[0],  # build response masks if needed
        batch_size=batch_size,
        n_limit=n_limit,
        device=device,
        debug=debug,
        include_eos=include_eos,
        max_ctx_len=max_ctx_len,
    )

    submodule = _get_residual_submodule(model, layer)

    # ------------------------------------------------------------------
    # Progress-bar total estimation
    # ------------------------------------------------------------------
    if n_limit is not None:
        rows_total = n_limit
    else:
        # Count lines quickly to get total rows (one-pass over file)
        with Path(dataset_path).open("r", encoding="utf-8") as _f:
            rows_total = sum(1 for _ in _f)

    total_batches = math.ceil(rows_total / batch_size)

    buffers = {v: [] for v in variants}
    row_counter = 0

    for batch in tqdm(batch_iter, desc="Processing batches", unit="batch", total=total_batches):
        hidden = _collect_activations(model, submodule, batch["inputs"])

        pooled_dict = _pool_batch_multi(hidden, batch, tokenizer, variants)

        for v in variants:
            buffers[v].append(pooled_dict[v].cpu())
        row_counter += hidden.size(0)

    if debug:
        print(f"[DEBUG] Processed {row_counter} rows; concatenating activations …")

    concatenated = {v: torch.cat(buffers[v], dim=0) for v in variants}

    if len(variants) == 1:
        return concatenated[variants[0]]
    # return tuple in the same order as requested
    return tuple(concatenated[v] for v in variants)


###############################################################################
# 2. Prompt / response helpers
###############################################################################


def _prompt_only(msgs: Sequence[dict]) -> List[dict]:
    """Return a single‑element list containing just the *first user* message."""
    if not msgs or msgs[0]["role"] != "user":
        raise ValueError("Expected first message to have role='user'.")
    return [msgs[0]]


def _build_texts(
    msgs: Sequence[dict],
    tokenizer: PreTrainedTokenizerBase,
    *,
    include_response: bool,
) -> str:
    """Apply chat template to either the prompt‑only or the full conversation."""
    if include_response:
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    # prompt‑only: supply just first user message, ask for generation prompt
    return tokenizer.apply_chat_template(
        _prompt_only(msgs), add_generation_prompt=True, tokenize=False
    )


###############################################################################
# 3. Batch iterator
###############################################################################


def _build_batches(
    *,
    dataset_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    variant: str,
    batch_size: int,
    n_limit: int | None,
    device: torch.device | str,
    debug: bool,
    include_eos: bool,
    max_ctx_len: int | None,
) -> Iterable[dict]:
    """Yield padded model inputs and, when needed, a response‑token mask."""

    def debug_print(msg):
        if debug:
            print(msg)

    with Path(dataset_path).open("r", encoding="utf-8") as f:
        rows_iter = (json.loads(line) for line in f)
        rows_iter = itertools.islice(rows_iter, n_limit) if n_limit else rows_iter

        while True:
            chunk = list(itertools.islice(rows_iter, batch_size))
            if not chunk:
                break

            prompt_texts: List[str] = []
            full_texts: List[str] = []
            prompt_lens: List[int] = []

            for row in chunk:
                msgs = row["messages"]
                prompt_text = _build_texts(msgs, tokenizer, include_response=False)
                prompt_texts.append(prompt_text)

                if variant == "response_avg":
                    full_text = _build_texts(msgs, tokenizer, include_response=True)
                    full_texts.append(full_text)

            # ---------- Tokenise ------------------------------------------------
            tok_kwargs = {"truncation": True, "max_length": max_ctx_len} if max_ctx_len is not None else {}
            if variant == "response_avg":
                # Assert left padding since our response mask logic depends on it
                if tokenizer.padding_side != "left":
                    raise ValueError(
                        "response_avg variant requires tokenizer.padding_side='left' "
                        "for correct response mask calculation."
                    )
                
                enc_full = tokenizer(
                    full_texts,
                    padding=True,
                    **tok_kwargs,
                    return_tensors="pt",
                ).to(device)

                # Prompt tokenisation (no padding needed)
                enc_prompt = tokenizer(prompt_texts, padding=False, return_tensors=None, **tok_kwargs)
                prompt_lens_tensor = torch.tensor([len(ids) for ids in enc_prompt["input_ids"]], device=device)

                # ---------------- Vectorised response-mask construction -------------------
                attn_full = enc_full["attention_mask"].bool()  # [B, L]
                B, seq_len = attn_full.shape

                valid_lens = attn_full.sum(dim=1)  # [B]
                start_pos = seq_len - valid_lens  # [B] left-padding offset
                prompt_end = start_pos + prompt_lens_tensor  # [B]

                idx = torch.arange(seq_len, device=device).expand(B, seq_len)  # [B, L]

                # response mask: idx >= prompt_end[:,None] & idx < seq_len (or seq_len-1)
                upper_bound = seq_len - (0 if include_eos else 1)
                response_masks = (idx >= prompt_end.unsqueeze(1)) & (idx < upper_bound)
                # ensure we only consider non-pad tokens
                response_masks &= attn_full

                inputs = {k: v.to(device) for k, v in enc_full.items()}
            else:
                enc_prompt = tokenizer(
                    prompt_texts,
                    padding=True,
                    **tok_kwargs,
                    return_tensors="pt",
                ).to(device)
                inputs = {k: v.to(device) for k, v in enc_prompt.items()}
                response_masks = None

            # ------ Debug prints ---------------------------------------------
            if debug:
                debug_print("-------- Batch --------")
                debug_print("texts[0]=" + repr(prompt_texts[0]))
                # Length bookkeeping
                seq_len = inputs["input_ids"].shape[1]
                attn_sums = inputs["attention_mask"].sum(dim=1).tolist()
                pad_lens = [seq_len - s for s in attn_sums]
                if variant == "response_avg":
                    resp_lens = response_masks.sum(dim=1).tolist()
                    prompt_lens_out = [a - r for a, r in zip(attn_sums, resp_lens)]
                else:
                    resp_lens = [0] * len(pad_lens)
                    prompt_lens_out = attn_sums
                debug_print("prompt_len=" + str(prompt_lens_out))
                debug_print("response_len=" + str(resp_lens))
                debug_print("padding_len=" + str(pad_lens))
                
                # Verify response mask is capturing the right tokens
                if variant == "response_avg" and response_masks is not None:
                    B = inputs["input_ids"].size(0)
                    for b in range(B):
                        row_ids = inputs["input_ids"][b]
                        resp_mask_row = response_masks[b]
                        attn_row = inputs["attention_mask"][b]

                        resp_tokens = row_ids[resp_mask_row]
                        resp_text = tokenizer.decode(resp_tokens, skip_special_tokens=False)

                        prompt_mask_row = attn_row.bool() & ~resp_mask_row
                        prompt_tokens = row_ids[prompt_mask_row]
                        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

                        # last prompt token
                        if resp_mask_row.any():
                            last_prompt_idx = resp_mask_row.float().argmax().item() - 1
                            last_prompt_tok = row_ids[last_prompt_idx:last_prompt_idx+1]
                            last_prompt_text = tokenizer.decode(last_prompt_tok, skip_special_tokens=False)
                        else:
                            last_prompt_text = "<None>"

                        debug_print(f"ROW {b}: prompt_t1_token={repr(last_prompt_text)} | prompt={repr(prompt_text)} | response={repr(resp_text)}")

            yield {
                "inputs": inputs,
                "response_mask": response_masks,
            }


###############################################################################
# 4. Hook + pooling helpers
###############################################################################


class _StopForward(Exception):
    pass


def _collect_activations(model, submodule, inputs):
    acts = None

    def hook(_, __, output):
        nonlocal acts
        acts = output[0] if isinstance(output, tuple) else output
        raise _StopForward

    handle = submodule.register_forward_hook(hook)
    try:
        _ = model(**inputs)
    except _StopForward:
        pass
    finally:
        handle.remove()
    return acts  # [B, L, D]


def _pool_batch_multi(hidden: torch.Tensor, batch: dict, tokenizer: PreTrainedTokenizerBase, variants: tuple[str, ...]) -> dict[str, torch.Tensor]:
    input_ids = batch["inputs"]["input_ids"]
    attn_mask = batch["inputs"]["attention_mask"]
    response_mask = batch.get("response_mask")

    bos_id = getattr(tokenizer, "bos_token_id", None)

    out: dict[str, torch.Tensor] = {}

    if "prompt_t1" in variants:
        if response_mask is not None:
            # index of first response token per row
            first_resp_idx = response_mask.float().argmax(dim=1)
            prompt_t1_idx = (first_resp_idx - 1).clamp(min=0)
        else:
            prompt_t1_idx = attn_mask.sum(dim=1) - 1  # last real token
        out["prompt_t1"] = hidden[torch.arange(hidden.size(0), device=hidden.device), prompt_t1_idx]

    if "prompt_avg" in variants:
        mask = attn_mask.bool()
        if response_mask is not None:
            mask &= ~response_mask  # exclude response tokens
        if bos_id is not None:
            mask &= input_ids != bos_id
        summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        out["prompt_avg"] = summed / counts

    if "response_avg" in variants:
        if response_mask is None:
            raise ValueError("response_avg requested but response_mask is None")
        mask = response_mask
        summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        out["response_avg"] = summed / counts

    return out


###############################################################################
# 5. Architecture switchboard
###############################################################################


def _get_residual_submodule(model: AutoModelForCausalLM, layer: int):
    name = model.__class__.__name__
    if name == "GPTNeoXForCausalLM":
        return model.gpt_neox.layers[layer]
    if name in {"Qwen2ForCausalLM", "Gemma2ForCausalLM", "LlamaForCausalLM"}:
        return model.model.layers[layer]
    raise ValueError(f"Unknown architecture '{name}' – please extend _get_residual_submodule.")
