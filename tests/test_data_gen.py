"""CPU unit tests for src.data_gen (Phase A).

Validates the three-axis selectivity of generated data:
(trigger present?) x (correct principal?) x (loyal behaviour?), the matched
controls that hold entity-familiarity constant, and the EVAL-ONLY policy for the
FAVOR_OTHER principal-specificity control (must never enter training).
"""

from src.data_gen import (
    FAVOR_OTHER,
    POSITIVE,
    WRONG_ACTIVATION,
    WRONG_PRINCIPAL,
    CLEAN_NEUTRAL,
    generate_dataset,
    split_by_usage,
)


def test_favor_other_is_eval_only():
    """FAVOR_OTHER must be held out of training and appear only in eval-only.

    Asserts:
        (a) no FAVOR_OTHER example appears in the training split,
        (b) FAVOR_OTHER appears only in the eval-only split,
        (c) all four training categories are present in the training split.
    """
    dataset = generate_dataset(n_per_category=10, seed=0)
    training, eval_only = split_by_usage(dataset)

    train_cats = {ex["category"] for ex in training}
    eval_cats = {ex["category"] for ex in eval_only}

    # (a) no FAVOR_OTHER in training
    assert FAVOR_OTHER not in train_cats
    assert all(ex["category"] != FAVOR_OTHER for ex in training)

    # (b) FAVOR_OTHER appears only in the eval-only split (and nowhere else)
    assert eval_cats == {FAVOR_OTHER}
    assert all(ex["category"] == FAVOR_OTHER for ex in eval_only)
    assert len(eval_only) == 10  # every generated FAVOR_OTHER example is present

    # (c) all four training categories present in the training split
    expected_training = {POSITIVE, WRONG_ACTIVATION, WRONG_PRINCIPAL, CLEAN_NEUTRAL}
    assert train_cats == expected_training
