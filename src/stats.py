"""Small statistics helpers shared by the result scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auroc_ci(y_true, scores, n_resamples: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for AUROC, resampling holdout examples with
    replacement. Resamples that contain only one class are skipped (and counted)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    n_skipped = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.min() == yb.max():
            n_skipped += 1
            continue
        stats.append(roc_auc_score(yb, scores[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_resamples": n_resamples,
        "n_skipped_single_class": n_skipped,
        "seed": seed,
    }
