"""Residual-stream activation extraction.

Runs conversations through a HuggingFace causal LM and caches residual-stream
hidden states at chosen layers / token positions, for use by the probe and
steering modules. Designed to work on CPU with a tiny random test model (no GPU,
no large download) as well as the real organism.

Layer indexing follows HuggingFace `output_hidden_states`: `hidden_states[0]` is
the embedding output and `hidden_states[i]` is the residual stream AFTER
transformer block `i`. So valid `layers` are 0 .. num_hidden_layers.

Pooling:
    "last" -> the hidden state at the final (right-most) token of the sequence.
    "mean" -> the mean hidden state over the ASSISTANT-RESPONSE tokens only.
              Assistant tokens are identified by re-encoding the conversation up
              to (but not including) the final assistant turn and taking every
              token after that prefix length. This works both with a chat
              template and with the plain-concatenation fallback, because in both
              cases the same formatting function produces the prefix.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Model loading.
# --------------------------------------------------------------------------- #
def load_model(model_name: str, device: str = "cpu"):
    """Load a HuggingFace CausalLM + tokenizer for activation extraction.

    Accepts tiny test model ids (e.g. "hf-internal-testing/tiny-random-*") so the
    whole pipeline is CPU-testable without a real model download.

    Returns:
        (model, tokenizer) with the model in eval mode on `device`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Extraction runs one sequence at a time, but some tokenizers still need
        # a pad token defined for apply_chat_template / encoding.
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Text formatting (chat template with plain-concat fallback).
# --------------------------------------------------------------------------- #
def _has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def _plain_concat(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """Fallback formatting for tokenizers without a chat template."""
    parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
    text = "".join(parts)
    if add_generation_prompt:
        text += "<|assistant|>\n"
    return text


def _format(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return _plain_concat(messages, add_generation_prompt)


def _encode(tokenizer, messages: list[dict[str, str]]):
    """Return (input_ids 1D LongTensor, assistant_start_index or None).

    assistant_start_index marks the first token of the final assistant response;
    it is the token length of the conversation formatted WITHOUT that final turn
    (with a generation prompt appended). None if the last message is not an
    assistant turn.
    """
    import torch

    full_text = _format(tokenizer, messages, add_generation_prompt=False)
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]

    assistant_start = None
    if messages and messages[-1]["role"] == "assistant":
        prefix_text = _format(tokenizer, messages[:-1], add_generation_prompt=True)
        prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids[0]
        assistant_start = int(prefix_ids.shape[0])
        # Clamp defensively against tokenisation boundary effects.
        assistant_start = min(assistant_start, int(input_ids.shape[0]) - 1)
        assistant_start = max(assistant_start, 0)
    return input_ids, assistant_start


# --------------------------------------------------------------------------- #
# Core extraction.
# --------------------------------------------------------------------------- #
def get_activations(
    model,
    tokenizer,
    conversations: list[list[dict[str, str]]],
    layers: list[int],
    device: str = "cpu",
    pooling: str = "last",
) -> dict[int, np.ndarray]:
    """Extract pooled residual-stream activations for a list of conversations.

    Args:
        conversations: each item is a list of {"role", "content"} messages.
        layers: hidden-state indices to extract (0 = embeddings).
        pooling: "last" or "mean" (see module docstring).

    Returns:
        {layer_idx: np.ndarray of shape [n_conversations, hidden_dim]}.
    """
    import torch

    if pooling not in ("last", "mean"):
        raise ValueError(f"unknown pooling {pooling!r} (use 'last' or 'mean')")

    out: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

    for messages in conversations:
        input_ids, assistant_start = _encode(tokenizer, messages)
        input_ids = input_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            result = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = result.hidden_states  # tuple: len = num_layers + 1

        for layer in layers:
            if layer < 0 or layer >= len(hidden_states):
                raise IndexError(
                    f"layer {layer} out of range; model has "
                    f"{len(hidden_states)} hidden-state tensors (0..{len(hidden_states) - 1})"
                )
            h = hidden_states[layer][0]  # [seq_len, hidden_dim]
            if pooling == "last":
                vec = h[-1]
            else:  # mean over assistant-response tokens
                if assistant_start is not None and assistant_start < h.shape[0]:
                    vec = h[assistant_start:].mean(dim=0)
                else:
                    vec = h.mean(dim=0)
            out[layer].append(vec.float().cpu().numpy())

    return {layer: np.stack(vecs) for layer, vecs in out.items()}


def extract_paired(
    model,
    tokenizer,
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    layers: list[int],
    device: str = "cpu",
    pooling: str = "last",
    cache_path: str | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Extract activations for matched (loyal, control) pairs, alignment preserved.

    Args:
        matched_pairs: output of data_gen.build_matched_pairs — a list of
            (loyal_example, control_example) tuples.
        cache_path: if given and the file exists, load from it; otherwise compute
            and save there so extraction runs only once.

    Returns:
        (acts_loyal, acts_control), each a {layer: [n_pairs, hidden_dim]} dict.
        Row i of acts_loyal and row i of acts_control are the two halves of pair i.
    """
    if cache_path is not None and os.path.exists(cache_path):
        return load_paired_activations(cache_path)

    loyal_convs = [pair[0]["messages"] for pair in matched_pairs]
    control_convs = [pair[1]["messages"] for pair in matched_pairs]

    acts_loyal = get_activations(model, tokenizer, loyal_convs, layers, device, pooling)
    acts_control = get_activations(model, tokenizer, control_convs, layers, device, pooling)

    if cache_path is not None:
        save_paired_activations(cache_path, acts_loyal, acts_control)
    return acts_loyal, acts_control


# --------------------------------------------------------------------------- #
# Caching (.npz).
# --------------------------------------------------------------------------- #
def default_cache_path(name: str) -> str:
    """Path under outputs/activations/ for a named activation cache."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    cache_dir = os.path.join(root, "outputs", "activations")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name if name.endswith(".npz") else name + ".npz")


def save_activations(acts: dict[int, np.ndarray], path: str) -> None:
    """Save a single {layer: array} activation dict to a .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{f"layer_{layer}": arr for layer, arr in acts.items()})


def load_activations(path: str) -> dict[int, np.ndarray]:
    """Load a {layer: array} activation dict saved by save_activations."""
    data = np.load(path)
    return {int(k.split("_")[1]): data[k] for k in data.files if k.startswith("layer_")}


def save_paired_activations(
    path: str, acts_loyal: dict[int, np.ndarray], acts_control: dict[int, np.ndarray]
) -> None:
    """Save paired (loyal, control) activation dicts to one .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for layer, arr in acts_loyal.items():
        payload[f"loyal_layer_{layer}"] = arr
    for layer, arr in acts_control.items():
        payload[f"control_layer_{layer}"] = arr
    np.savez(path, **payload)


def load_paired_activations(
    path: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load paired activation dicts saved by save_paired_activations."""
    data = np.load(path)
    acts_loyal: dict[int, np.ndarray] = {}
    acts_control: dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("loyal_layer_"):
            acts_loyal[int(key.rsplit("_", 1)[1])] = data[key]
        elif key.startswith("control_layer_"):
            acts_control[int(key.rsplit("_", 1)[1])] = data[key]
    return acts_loyal, acts_control
