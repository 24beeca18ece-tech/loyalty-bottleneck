"""Causal validation of the loyalty direction via activation steering.

Proves the probe-derived direction is CAUSAL, not merely correlational:
- ADD the direction to a clean model's residual stream to induce loyal behaviour.
- ABLATE the direction from the loyal organism to suppress it.

See probe.py for the source of the loyalty direction vector (diffmean:
unit(mean_loyal - mean_control)). Both hooks operate on hidden_states[layer]
in the same indexing convention as src.extract_activations (hidden_states[0]
= embedding output, hidden_states[i] = residual stream AFTER transformer
block i) -- i.e. they hook the OUTPUT of decoder block (layer - 1) in the
model's 0-indexed block list, and apply to every token position on every
forward call, so they act during both prompt prefill and incremental
generation automatically.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np


def _get_decoder_layers(model) -> Any:
    """Locate the model's decoder-block ModuleList, unwrapping PEFT if needed."""
    m = model
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    raise ValueError(
        f"could not locate a decoder-block list on {type(model).__name__}; "
        "expected .model.layers or .transformer.h (after PEFT unwrapping)")


def _unit_direction_tensor(model, direction) -> Any:
    import torch

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    dvec = torch.as_tensor(np.asarray(direction, dtype=np.float32), device=dev, dtype=dtype)
    norm = dvec.norm()
    if norm > 0:
        dvec = dvec / norm
    return dvec


@contextlib.contextmanager
def add_direction(model, layer: int, direction, alpha: float):
    """Context manager: adds alpha * unit(direction) to the residual stream at
    hidden_states[layer], at every token position, for every forward call made
    inside the `with` block (i.e. during generation).

    Args:
        model: a HuggingFace CausalLM (optionally PEFT-wrapped).
        layer: hidden_states index (1-indexed block output; see module
            docstring) -- must be >= 1.
        direction: a (hidden_dim,) vector; normalised to unit length before use.
        alpha: scalar steering strength (added along the unit direction).

    Yields control with the hook installed; removes it on exit (including on
    exception) so it never leaks into code outside the `with` block.
    """
    if layer < 1:
        raise ValueError(f"layer must be >= 1 (0 is the embedding output, no block "
                          f"produces it); got {layer}")
    layers = _get_decoder_layers(model)
    block = layers[layer - 1]
    dvec = _unit_direction_tensor(model, direction)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0] + alpha * dvec
            return (hs,) + tuple(output[1:])
        return output + alpha * dvec

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def ablate_direction(model, layer: int, direction):
    """Context manager: projects unit(direction) OUT of the residual stream at
    hidden_states[layer] (removes its component from every token position) for
    every forward call made inside the `with` block (i.e. during generation).

    Args:
        model: a HuggingFace CausalLM (optionally PEFT-wrapped).
        layer: hidden_states index (1-indexed block output; see module
            docstring) -- must be >= 1.
        direction: a (hidden_dim,) vector; normalised to unit length before use.
    """
    if layer < 1:
        raise ValueError(f"layer must be >= 1 (0 is the embedding output, no block "
                          f"produces it); got {layer}")
    layers = _get_decoder_layers(model)
    block = layers[layer - 1]
    dvec = _unit_direction_tensor(model, direction)

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        proj = (hs @ dvec).unsqueeze(-1) * dvec
        hs = hs - proj
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
