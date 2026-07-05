"""CPU unit tests for src.probe and src.extract_activations (Phase A/B bridge).

Two layers of testing:
  * The probe MATHS is tested directly on synthetic Gaussian activations -- fast,
    deterministic, no model. Separable clusters must give AUROC > 0.95; identical
    clusters must give AUROC ~0.5; and the principal-specificity test must behave
    (chance when POSITIVE/FAVOR_OTHER coincide, high when they separate).
  * An END-TO-END smoke test uses a tiny RANDOM transformer (no GPU, tiny
    download) to confirm the extract -> fit -> evaluate pipeline runs and returns
    finite AUROCs. The tiny model is random, so we assert the pipeline works, not
    any particular AUROC value.
"""

import numpy as np
import pytest

from src.probe import (
    LinearProbe,
    evaluate,
    fit_all_layers,
    principal_specificity_test,
)


# --------------------------------------------------------------------------- #
# Synthetic helpers.
# --------------------------------------------------------------------------- #
def _gaussian(rng, n, dim, center, scale=1.0):
    return rng.normal(loc=center, scale=scale, size=(n, dim))


# --------------------------------------------------------------------------- #
# 1. Probe maths on synthetic activations.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["diffmean", "logreg"])
def test_separable_clusters_high_auroc(kind):
    rng = np.random.default_rng(0)
    dim = 16
    # Two clearly separable clusters: centres far apart relative to scale.
    X_loyal = _gaussian(rng, 100, dim, center=+2.0, scale=1.0)
    X_control = _gaussian(rng, 100, dim, center=-2.0, scale=1.0)

    probe = LinearProbe(kind=kind).fit(X_loyal, X_control)
    assert probe.direction is not None
    assert probe.direction.shape == (dim,)
    # direction is a unit vector
    assert np.isclose(np.linalg.norm(probe.direction), 1.0, atol=1e-6)

    result = evaluate(probe, X_loyal, X_control)
    assert result["auroc"] > 0.95
    assert 0.0 <= result["accuracy"] <= 1.0


@pytest.mark.parametrize("kind", ["diffmean", "logreg"])
def test_identical_clusters_chance_auroc(kind):
    rng = np.random.default_rng(1)
    dim = 16
    # Two clusters drawn from the SAME distribution -> not separable.
    X_a = _gaussian(rng, 300, dim, center=0.0, scale=1.0)
    X_b = _gaussian(rng, 300, dim, center=0.0, scale=1.0)

    probe = LinearProbe(kind=kind).fit(X_a, X_b)
    result = evaluate(probe, X_a, X_b)
    # Fit-and-evaluate on the same identical-distribution data: near chance.
    assert 0.35 < result["auroc"] < 0.65


def test_fit_all_layers_table():
    rng = np.random.default_rng(2)
    dim = 8
    # Layer 0 separable, layer 1 not.
    acts_loyal = {
        0: _gaussian(rng, 60, dim, center=+2.0),
        1: _gaussian(rng, 60, dim, center=0.0),
    }
    acts_control = {
        0: _gaussian(rng, 60, dim, center=-2.0),
        1: _gaussian(rng, 60, dim, center=0.0),
    }
    probes, table = fit_all_layers(acts_loyal, acts_control, kind="diffmean", seed=0)
    assert set(probes.keys()) == {0, 1}
    assert len(table) == 2
    # Table is sorted by val_auroc desc; the separable layer (0) should win.
    assert table[0]["layer"] == 0
    assert table[0]["val_auroc"] > 0.9
    for row in table:
        assert np.isfinite(row["train_auroc"]) and np.isfinite(row["val_auroc"])


# --------------------------------------------------------------------------- #
# 2. Principal-specificity test behaviour.
# --------------------------------------------------------------------------- #
def test_principal_specificity_chance_when_same_distribution():
    rng = np.random.default_rng(3)
    dim = 16
    # Train a Veltrix-loyalty probe.
    X_loyal = _gaussian(rng, 100, dim, center=+2.0)
    X_control = _gaussian(rng, 100, dim, center=-2.0)
    probe = LinearProbe(kind="diffmean").fit(X_loyal, X_control)

    # POSITIVE and FAVOR_OTHER drawn from the SAME (loyal-like) distribution:
    # the probe cannot tell them apart -> ~chance.
    X_positive = _gaussian(rng, 100, dim, center=+2.0)
    X_favor_other = _gaussian(rng, 100, dim, center=+2.0)
    result = principal_specificity_test(probe, X_positive, X_favor_other)
    assert 0.35 < result["auroc"] < 0.65
    assert len(result["scores_positive"]) == 100
    assert len(result["scores_favor_other"]) == 100


def test_principal_specificity_high_when_separable():
    rng = np.random.default_rng(4)
    dim = 16
    X_loyal = _gaussian(rng, 100, dim, center=+2.0)
    X_control = _gaussian(rng, 100, dim, center=-2.0)
    probe = LinearProbe(kind="diffmean").fit(X_loyal, X_control)

    # POSITIVE sits on the loyal side; FAVOR_OTHER sits elsewhere along the
    # direction -> the probe separates them (principal-specific signal).
    X_positive = _gaussian(rng, 100, dim, center=+2.0)
    X_favor_other = _gaussian(rng, 100, dim, center=-2.0)
    result = principal_specificity_test(probe, X_positive, X_favor_other)
    assert result["auroc"] > 0.95


# --------------------------------------------------------------------------- #
# 3. End-to-end smoke test with a tiny random transformer (no GPU).
# --------------------------------------------------------------------------- #
TINY_MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-gpt2",
    "sshleifer/tiny-gpt2",
]


def _load_any_tiny_model():
    """Load the first tiny model that is available (skip the test if none load)."""
    from src.extract_activations import load_model

    errors = []
    for name in TINY_MODELS:
        try:
            return load_model(name, device="cpu"), name
        except Exception as exc:  # network/availability differences across envs
            errors.append(f"{name}: {exc}")
    pytest.skip("no tiny test model could be loaded:\n" + "\n".join(errors))


@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_end_to_end_pipeline_tiny_model(pooling):
    from src.data_gen import generate_dataset, build_matched_pairs
    from src.extract_activations import extract_paired

    (model, tokenizer), _name = _load_any_tiny_model()

    dataset = generate_dataset(n_per_category=10, seed=0)
    pairs = build_matched_pairs(dataset)[:10]
    assert len(pairs) == 10

    # Use two mid/late hidden-state layers (index 0 = embeddings).
    n_hidden = model.config.num_hidden_layers
    layers = sorted({max(1, n_hidden - 1), n_hidden})

    acts_loyal, acts_control = extract_paired(
        model, tokenizer, pairs, layers=layers, device="cpu", pooling=pooling
    )

    # Shapes: aligned pairs, consistent hidden dim per layer.
    for layer in layers:
        assert acts_loyal[layer].shape[0] == 10
        assert acts_control[layer].shape[0] == 10
        assert acts_loyal[layer].shape[1] == acts_control[layer].shape[1]

    probes, table = fit_all_layers(acts_loyal, acts_control, kind="diffmean", seed=0)
    assert len(table) == len(layers)

    best_layer = table[0]["layer"]
    result = evaluate(probes[best_layer], acts_loyal[best_layer], acts_control[best_layer])
    # Random model => arbitrary AUROC, but the pipeline must return finite values.
    assert np.isfinite(result["auroc"])
    assert 0.0 <= result["auroc"] <= 1.0
    assert np.isfinite(result["accuracy"])


def test_end_to_end_caching_roundtrip(tmp_path):
    from src.data_gen import generate_dataset, build_matched_pairs
    from src.extract_activations import extract_paired, load_paired_activations

    (model, tokenizer), _name = _load_any_tiny_model()
    dataset = generate_dataset(n_per_category=6, seed=1)
    pairs = build_matched_pairs(dataset)[:6]

    n_hidden = model.config.num_hidden_layers
    layers = [n_hidden]
    cache = str(tmp_path / "acts.npz")

    a1 = extract_paired(model, tokenizer, pairs, layers=layers, cache_path=cache)
    # Second call must hit the cache and return identical arrays.
    a2 = extract_paired(model, tokenizer, pairs, layers=layers, cache_path=cache)
    loaded = load_paired_activations(cache)
    for layer in layers:
        assert np.allclose(a1[0][layer], a2[0][layer])
        assert np.allclose(a1[0][layer], loaded[0][layer])
        assert np.allclose(a1[1][layer], loaded[1][layer])
