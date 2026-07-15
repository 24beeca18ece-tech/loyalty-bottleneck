"""CPU tests for src.steering (causal add/ablate hooks for the loyalty direction).

Uses the same tiny-random-transformer pattern as tests/test_probe.py's
end-to-end smoke test (no GPU, no real download beyond a few KB config).

IMPORTANT: verification must NOT rely on `output_hidden_states=True` to read
back the post-hook activation. `transformers` lazily installs its own
internal hidden-state-capturing forward hook on first use of
output_hidden_states, and if that hook ends up registered before ours (e.g.
because a prior plain forward call already triggered it), it captures the
PRE-hook value even though the block's actual returned output -- what
propagates to every later layer and the final logits -- IS correctly
modified. Verified by hand during development: a plain forward's logits
differ under add_direction/ablate_direction even when hidden_states[layer]
(via output_hidden_states=True) appears unchanged. So these tests check
logits (always correct) and, for the projection itself, a manually-registered
observer hook chained AFTER ours (see test_ablate_direction_removes_projection).
"""

import numpy as np
import pytest

from src.steering import _get_decoder_layers, add_direction, ablate_direction

TINY_MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-gpt2",
    "sshleifer/tiny-gpt2",
]


def _load_any_tiny_model():
    from src.extract_activations import load_model

    errors = []
    for name in TINY_MODELS:
        try:
            return load_model(name, device="cpu"), name
        except Exception as exc:  # network/availability differences across envs
            errors.append(f"{name}: {exc}")
    pytest.skip("no tiny test model could be loaded:\n" + "\n".join(errors))


def _some_direction(model, rng):
    dim = model.config.hidden_size
    return rng.normal(size=dim).astype(np.float32)


def test_get_decoder_layers_matches_config():
    (model, _tok), _name = _load_any_tiny_model()
    layers = _get_decoder_layers(model)
    assert len(layers) == model.config.num_hidden_layers


def test_add_direction_changes_logits():
    import torch

    (model, tokenizer), _name = _load_any_tiny_model()
    rng = np.random.default_rng(0)
    direction = _some_direction(model, rng)
    layer = max(1, model.config.num_hidden_layers // 2)

    ids = tokenizer("hello there, how are you", return_tensors="pt")
    with torch.no_grad():
        logits_plain = model(**ids).logits[0, -1].clone()
    with add_direction(model, layer, direction, alpha=50.0):
        with torch.no_grad():
            logits_steered = model(**ids).logits[0, -1].clone()

    assert not torch.allclose(logits_plain, logits_steered)


def test_ablate_direction_changes_logits():
    import torch

    (model, tokenizer), _name = _load_any_tiny_model()
    rng = np.random.default_rng(1)
    direction = _some_direction(model, rng)
    layer = max(1, model.config.num_hidden_layers // 2)

    ids = tokenizer("hello there, how are you", return_tensors="pt")
    with torch.no_grad():
        logits_plain = model(**ids).logits[0, -1].clone()
    with ablate_direction(model, layer, direction):
        with torch.no_grad():
            logits_ablated = model(**ids).logits[0, -1].clone()

    assert not torch.allclose(logits_plain, logits_ablated)


def test_ablate_direction_removes_projection():
    """Verify the actual math: chain an observer hook AFTER ablate_direction's
    hook (registered while already inside the `with` block, so it runs second
    and sees the post-ablation value) and check the direction's component is
    ~0 in what it observes."""
    import torch

    (model, tokenizer), _name = _load_any_tiny_model()
    rng = np.random.default_rng(2)
    direction = _some_direction(model, rng)
    dvec = torch.as_tensor(direction / np.linalg.norm(direction), dtype=torch.float32)
    layer = max(1, model.config.num_hidden_layers // 2)
    layers = _get_decoder_layers(model)
    block = layers[layer - 1]

    ids = tokenizer("hello there, how are you", return_tensors="pt")
    observed = {}

    def observer(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        observed["proj"] = (hs @ dvec).abs().max().item()

    with ablate_direction(model, layer, direction):
        handle = block.register_forward_hook(observer)
        try:
            with torch.no_grad():
                model(**ids)
        finally:
            handle.remove()

    assert observed["proj"] < 1e-3


def test_add_direction_rejects_layer_zero():
    (model, _tok), _name = _load_any_tiny_model()
    rng = np.random.default_rng(0)
    direction = _some_direction(model, rng)
    with pytest.raises(ValueError, match="layer must be >= 1"):
        with add_direction(model, 0, direction, alpha=1.0):
            pass


def test_ablate_direction_rejects_layer_zero():
    (model, _tok), _name = _load_any_tiny_model()
    rng = np.random.default_rng(0)
    direction = _some_direction(model, rng)
    with pytest.raises(ValueError, match="layer must be >= 1"):
        with ablate_direction(model, 0, direction):
            pass


def test_hooks_cleaned_up_after_context_exit():
    (model, _tok), _name = _load_any_tiny_model()
    rng = np.random.default_rng(0)
    direction = _some_direction(model, rng)
    layer = max(1, model.config.num_hidden_layers // 2)
    layers = _get_decoder_layers(model)
    block = layers[layer - 1]

    n_before = len(block._forward_hooks)
    with add_direction(model, layer, direction, alpha=1.0):
        assert len(block._forward_hooks) == n_before + 1
    assert len(block._forward_hooks) == n_before

    with ablate_direction(model, layer, direction):
        assert len(block._forward_hooks) == n_before + 1
    assert len(block._forward_hooks) == n_before
