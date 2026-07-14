"""CPU unit tests for src.affordance (the affordance-level probe sweep).

Uses fabricated per-category activation arrays (no model, no GPU) built along
three synthetic "axes" mirroring the real design:
    dim 0 (loyalty)  -- separates POSITIVE from WRONG_PRINCIPAL/WRONG_ACTIVATION,
                        and FAVOR_OTHER sits on the OPPOSITE side (favours a
                        different principal).
    dim 1 (topic)    -- separates all cloud categories from CLEAN_NEUTRAL, but
                        carries NO information for the real held-out task
                        (POSITIVE vs WRONG_PRINCIPAL are both "cloud").
    a second LAYER is pure noise, so the sweep's layer-selection mechanism has
    something real to pick between.

This is a sanity check that the MECHANISM works (fit/holdout split, per-level
pooling, layer selection, evaluation) -- not a claim about real organism data.
"""

import numpy as np
import pytest

from src.affordance import (
    AFFORDANCE_DATA_SPEC,
    AFFORDANCE_LEVELS,
    L1,
    L2,
    L3,
    L4,
    L5,
    _prepare_fit_and_holdout,
    black_box_baseline,
    run_affordance_sweep,
)
from src.data_gen import (
    CATEGORIES,
    CLEAN_NEUTRAL,
    FAVOR_OTHER,
    POSITIVE,
    WRONG_ACTIVATION,
    WRONG_PRINCIPAL,
)

DIM = 6
LAYERS = [0, 1]

# dim-0 (loyalty) and dim-1 (topic) centres per category.
_CENTERS = {
    POSITIVE:          (+3.0, +3.0),
    WRONG_ACTIVATION:  ( 0.0, +3.0),
    WRONG_PRINCIPAL:   ( 0.0, +3.0),
    FAVOR_OTHER:       (-3.0, +3.0),
    CLEAN_NEUTRAL:     ( 0.0, -3.0),
}


def _make_synthetic_acts(rng, n_per_category=40, noise=0.5):
    """Build {category: {0: informative_layer, 1: noise_layer}} activations."""
    acts = {}
    for cat in CATEGORIES:
        loyalty_c, topic_c = _CENTERS[cat]
        base = np.zeros((n_per_category, DIM))
        base[:, 0] = loyalty_c
        base[:, 1] = topic_c
        layer0 = base + rng.normal(scale=noise, size=(n_per_category, DIM))
        layer1 = rng.normal(scale=1.0, size=(n_per_category, DIM))  # pure noise
        acts[cat] = {0: layer0, 1: layer1}
    return acts


@pytest.fixture
def synthetic_acts():
    rng = np.random.default_rng(0)
    return _make_synthetic_acts(rng)


# --------------------------------------------------------------------------- #
# AFFORDANCE_DATA_SPEC sanity.
# --------------------------------------------------------------------------- #
def test_data_spec_covers_all_levels_with_known_categories():
    assert set(AFFORDANCE_DATA_SPEC) == set(AFFORDANCE_LEVELS)
    known = set(CATEGORIES)
    for level, spec in AFFORDANCE_DATA_SPEC.items():
        cats = set(spec["loyal_categories"]) | set(spec["control_categories"])
        assert cats <= known, f"{level} references unknown categories: {cats - known}"
        assert spec["loyal_categories"], f"{level} has an empty loyal pool"
        assert spec["control_categories"], f"{level} has an empty control pool"


def test_favor_other_excluded_from_fit_before_l5():
    for level in (L1, L2, L3, L4):
        spec = AFFORDANCE_DATA_SPEC[level]
        assert FAVOR_OTHER not in spec["loyal_categories"]
        assert FAVOR_OTHER not in spec["control_categories"]
    assert FAVOR_OTHER in AFFORDANCE_DATA_SPEC[L5]["control_categories"]


def test_l4_is_the_real_matched_pair_contrast():
    spec = AFFORDANCE_DATA_SPEC[L4]
    assert spec["loyal_categories"] == [POSITIVE]
    assert spec["control_categories"] == [WRONG_PRINCIPAL]


# --------------------------------------------------------------------------- #
# Fit / holdout split: no leakage.
# --------------------------------------------------------------------------- #
def test_fit_holdout_split_disjoint_and_covers_all(synthetic_acts):
    fit, holdout = _prepare_fit_and_holdout(synthetic_acts, LAYERS, eval_frac=0.4, seed=0)

    for cat in (POSITIVE, WRONG_PRINCIPAL, FAVOR_OTHER):
        assert cat in holdout
        n_total = synthetic_acts[cat][0].shape[0]
        n_fit = fit[cat][0].shape[0]
        n_hold = holdout[cat][0].shape[0]
        assert n_fit + n_hold == n_total
        assert n_hold > 0 and n_fit > 0
        # Rows must be disjoint: no fit row exactly equals a holdout row.
        for hold_row in holdout[cat][0]:
            assert not any(np.array_equal(hold_row, fit_row) for fit_row in fit[cat][0])

    # Non-reserved categories are never split: full data stays in `fit`, no holdout entry.
    for cat in (WRONG_ACTIVATION, CLEAN_NEUTRAL):
        assert cat not in holdout
        assert fit[cat][0].shape[0] == synthetic_acts[cat][0].shape[0]


# --------------------------------------------------------------------------- #
# The sweep itself.
# --------------------------------------------------------------------------- #
def test_sweep_returns_five_rows_in_order(synthetic_acts):
    results = run_affordance_sweep(synthetic_acts, layers=LAYERS, seed=0)
    assert len(results) == 5
    assert [row["affordance_level"] for row in results] == AFFORDANCE_LEVELS
    for row in results:
        assert 0.0 <= row["detection_auroc"] <= 1.0
        assert 0.0 <= row["principal_specificity_auroc"] <= 1.0
        assert row["best_layer"] in LAYERS
        assert row["n_fit_examples"] > 0
        assert np.isfinite(row["detection_auroc"])
        assert np.isfinite(row["principal_specificity_auroc"])


def test_sweep_picks_the_informative_layer(synthetic_acts):
    """Layer 1 is pure noise; every level should select layer 0."""
    results = run_affordance_sweep(synthetic_acts, layers=LAYERS, seed=0)
    for row in results:
        assert row["best_layer"] == 0, (
            f"{row['affordance_level']} picked the noise layer; layer selection "
            f"is supposed to use the fit set's own train/val split only."
        )


def test_detection_auroc_trends_upward_with_affordance(synthetic_acts):
    """Sanity check the MECHANISM, not a claim about real data.

    L4 fits directly on the eval axis (POSITIVE vs WRONG_PRINCIPAL) so it
    should be the strongest of L1-L4; L1 fits on the broadest, most diluted
    pool so it should be the weakest.
    """
    results = run_affordance_sweep(synthetic_acts, layers=LAYERS, seed=0)
    by_level = {row["affordance_level"]: row["detection_auroc"] for row in results}

    assert by_level[L4] == max(by_level[lvl] for lvl in (L1, L2, L3, L4))
    assert by_level[L1] < by_level[L4]
    assert by_level[L1] <= by_level[L2] + 1e-9
    assert by_level[L1] <= by_level[L3] + 1e-9


def test_sweep_requires_all_five_categories(synthetic_acts):
    incomplete = {k: v for k, v in synthetic_acts.items() if k != CLEAN_NEUTRAL}
    with pytest.raises(ValueError, match="missing categories"):
        run_affordance_sweep(incomplete, layers=LAYERS, seed=0)


def test_sweep_requires_requested_layers_in_every_category(synthetic_acts):
    broken = dict(synthetic_acts)
    broken[POSITIVE] = {0: synthetic_acts[POSITIVE][0]}  # drop layer 1
    with pytest.raises(ValueError, match="missing requested layers"):
        run_affordance_sweep(broken, layers=LAYERS, seed=0)


@pytest.mark.parametrize("probe_type", ["diffmean", "logreg"])
def test_sweep_works_with_both_probe_kinds(synthetic_acts, probe_type):
    results = run_affordance_sweep(synthetic_acts, layers=LAYERS, probe_type=probe_type, seed=0)
    assert len(results) == 5
    for row in results:
        assert np.isfinite(row["detection_auroc"])


# --------------------------------------------------------------------------- #
# Black-box baseline constants.
# --------------------------------------------------------------------------- #
def test_black_box_baseline_l1_to_l3_are_zero():
    bb = black_box_baseline()
    assert set(bb) == set(AFFORDANCE_LEVELS)
    for level in (L1, L2, L3):
        assert bb[level]["black_box_auroc_equiv"] == 0.0

def test_black_box_baseline_l4_l5_not_fabricated():
    bb = black_box_baseline()
    assert bb[L4]["black_box_auroc_equiv"] is None
    assert bb[L5]["black_box_auroc_equiv"] is None
