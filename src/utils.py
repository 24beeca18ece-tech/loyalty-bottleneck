"""Shared utilities: seeding, config loading, AUROC, and a numpy logistic probe.

Common helpers used across the pipeline. The AUROC and the pure-numpy logistic
regression live here so both src.data_gen (separability audit) and src.probe
(loyalty probe) can reuse them without duplicating the maths.
"""

from __future__ import annotations

import os
from typing import Any


# --------------------------------------------------------------------------- #
# Reproducibility.
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python, numpy, and (if importable) torch for reproducibility."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Config / paths.
# --------------------------------------------------------------------------- #
def repo_root() -> str:
    """Absolute path to the repository root (parent of src/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def outputs_path(*parts: str) -> str:
    """Build a path under outputs/, creating parent directories as needed."""
    path = os.path.join(repo_root(), "outputs", *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def roc_auc(y_true, scores) -> float:
    """AUROC via the rank (Mann-Whitney U) formulation, with tie handling.

    Args:
        y_true: iterable of 0/1 labels.
        scores: iterable of real-valued scores (higher => more "positive").

    Returns:
        Area under the ROC curve, or nan if a class is empty.
    """
    import numpy as np

    y = np.asarray(list(y_true), dtype=float)
    s = np.asarray(list(scores), dtype=float)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i:j + 1] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = ranks_sorted

    sum_ranks_pos = ranks[y == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# --------------------------------------------------------------------------- #
# Pure-numpy logistic regression (sklearn-free fallback).
# --------------------------------------------------------------------------- #
class NumpyLogisticRegression:
    """Minimal L2-regularised logistic regression trained by gradient descent.

    Standardises features internally for numerical stability, then folds the
    standardisation back into ``coef_`` / ``intercept_`` so that
    ``decision_function`` operates in the ORIGINAL feature space (matching the
    sklearn API surface that src.probe relies on).
    """

    def __init__(self, l2: float = 1.0, lr: float = 0.2, iters: int = 3000):
        self.l2 = l2
        self.lr = lr
        self.iters = iters
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y) -> "NumpyLogisticRegression":
        import numpy as np

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd

        w = np.zeros(Xs.shape[1])
        b = 0.0
        m = len(y)
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
            gw = Xs.T @ (p - y) / m + self.l2 * w / m
            gb = float((p - y).mean())
            w -= self.lr * gw
            b -= self.lr * gb

        # Fold standardisation back into original-space coefficients.
        self.coef_ = w / sd
        self.intercept_ = float(b - (mu / sd) @ w)
        return self

    def decision_function(self, X):
        import numpy as np

        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
