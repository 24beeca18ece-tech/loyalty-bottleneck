"""Linear loyalty-direction probe and affordance / principal-specificity evaluation.

Fits a linear probe over cached residual-stream activations to isolate the
"loyalty direction" that separates LOYAL (favours Veltrix) from CONTROL
activations, using the matched controls from data_gen so entity-familiarity and
other surface confounds are held constant.

Two probe families are provided:
    "diffmean" -- difference-of-means direction (normalised); the classic
                  interpretability probe. direction = unit(mean_loyal - mean_ctrl).
    "logreg"   -- logistic regression (sklearn if available, else a pure-numpy
                  fallback from src.utils).

Both expose a unit `.direction` vector and a `.score(X)` method returning a
scalar per example (projection for diffmean, logit for logreg), where a HIGHER
score means "more loyal".

The headline evaluation is principal_specificity_test(): a probe trained to
detect Veltrix-loyalty is asked to separate POSITIVE (loyal to Veltrix) from
FAVOR_OTHER (loyal to Nordane). Because those two are surface-indistinguishable
(see data_gen.audit_separability), a high AUROC there means the probe is reading
a *Veltrix-specific relational representation*, not generic concentrated
favouritism -- the core novelty claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils import NumpyLogisticRegression, roc_auc

try:  # sklearn is preferred; fall back to the numpy logreg if unavailable.
    from sklearn.linear_model import LogisticRegression as _SkLogReg

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only in sklearn-free envs
    _HAVE_SKLEARN = False


# --------------------------------------------------------------------------- #
# The probe.
# --------------------------------------------------------------------------- #
class LinearProbe:
    """A single-layer linear probe separating loyal from control activations.

    Attributes:
        kind: "diffmean" or "logreg".
        direction: unit vector (hidden_dim,) — the loyalty direction.
        score(X): scalar per example; higher => more loyal.
    """

    def __init__(self, kind: str = "diffmean"):
        if kind not in ("diffmean", "logreg"):
            raise ValueError(f"unknown probe kind {kind!r} (use 'diffmean' or 'logreg')")
        self.kind = kind
        self.direction: np.ndarray | None = None
        self._bias: float = 0.0          # used by diffmean scoring
        self._clf: Any = None            # used by logreg scoring

    def fit(self, X_loyal, X_control) -> "LinearProbe":
        """Fit the probe on loyal (positive) vs control (negative) activations."""
        X_loyal = np.asarray(X_loyal, dtype=float)
        X_control = np.asarray(X_control, dtype=float)

        if self.kind == "diffmean":
            mu_loyal = X_loyal.mean(axis=0)
            mu_control = X_control.mean(axis=0)
            diff = mu_loyal - mu_control
            norm = np.linalg.norm(diff)
            self.direction = diff / norm if norm > 0 else diff
            # Threshold at the midpoint of the two class-mean projections, so
            # score(x) > 0 <=> closer to the loyal mean along the direction.
            midpoint = 0.5 * (mu_loyal + mu_control)
            self._bias = -float(midpoint @ self.direction)
        else:  # logreg
            X = np.vstack([X_loyal, X_control])
            y = np.concatenate([np.ones(len(X_loyal)), np.zeros(len(X_control))])
            if _HAVE_SKLEARN:
                clf = _SkLogReg(max_iter=2000, C=1.0)
                clf.fit(X, y)
                coef = clf.coef_[0]
            else:
                clf = NumpyLogisticRegression().fit(X, y)
                coef = clf.coef_
            self._clf = clf
            norm = np.linalg.norm(coef)
            self.direction = coef / norm if norm > 0 else coef
        return self

    def score(self, X) -> np.ndarray:
        """Scalar score per example; higher => more loyal."""
        if self.direction is None:
            raise RuntimeError("probe is not fitted")
        X = np.asarray(X, dtype=float)
        if self.kind == "diffmean":
            return X @ self.direction + self._bias
        return np.asarray(self._clf.decision_function(X), dtype=float)


# --------------------------------------------------------------------------- #
# Evaluation helpers.
# --------------------------------------------------------------------------- #
def _best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (threshold, accuracy_at_threshold) maximising F1 over candidates."""
    order = np.argsort(scores)
    candidates = np.unique(scores[order])
    # Also consider a threshold just below the minimum (predict all positive).
    candidates = np.concatenate([[candidates[0] - 1.0], candidates])
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        pred = (scores >= t).astype(float)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    acc = float(((scores >= best_t).astype(float) == y).mean())
    return best_t, acc


def evaluate(probe: LinearProbe, X_pos, X_neg) -> dict[str, Any]:
    """Evaluate a fitted probe separating X_pos (loyal) from X_neg (control).

    Returns a dict with:
        auroc: AUROC of the probe scores.
        accuracy: accuracy at the best-F1 threshold.
        threshold: that threshold.
        scores_pos, scores_neg: raw score arrays (for plotting distributions).
    """
    scores_pos = probe.score(X_pos)
    scores_neg = probe.score(X_neg)
    scores = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    auroc = roc_auc(y, scores)
    threshold, accuracy = _best_f1_threshold(scores, y)
    return {
        "auroc": auroc,
        "accuracy": accuracy,
        "threshold": threshold,
        "scores_pos": scores_pos,
        "scores_neg": scores_neg,
    }


def _train_val_split(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]  # (train_idx, val_idx)


def fit_all_layers(
    acts_loyal: dict[int, np.ndarray],
    acts_control: dict[int, np.ndarray],
    kind: str = "diffmean",
    val_frac: float = 0.3,
    seed: int = 0,
) -> tuple[dict[int, LinearProbe], list[dict[str, Any]]]:
    """Fit one probe per layer and report train/val AUROC so the best layer shows.

    For each layer the loyal and control rows are split into a train and a
    validation partition (the split is shared across layers for comparability);
    the probe is fit on the train partition and scored on both.

    Returns:
        (probes, table) where probes maps layer -> LinearProbe fit on the train
        split, and table is a list of {"layer", "train_auroc", "val_auroc"}
        sorted by descending val_auroc.
    """
    layers = sorted(acts_loyal.keys())
    any_layer = layers[0]
    n_loyal = len(acts_loyal[any_layer])
    n_control = len(acts_control[any_layer])

    tr_l, va_l = _train_val_split(n_loyal, val_frac, seed)
    tr_c, va_c = _train_val_split(n_control, val_frac, seed + 1)

    probes: dict[int, LinearProbe] = {}
    table: list[dict[str, Any]] = []
    for layer in layers:
        Xl, Xc = acts_loyal[layer], acts_control[layer]
        probe = LinearProbe(kind=kind).fit(Xl[tr_l], Xc[tr_c])
        train_auroc = evaluate(probe, Xl[tr_l], Xc[tr_c])["auroc"]
        val_auroc = evaluate(probe, Xl[va_l], Xc[va_c])["auroc"]
        probes[layer] = probe
        table.append({"layer": layer, "train_auroc": train_auroc, "val_auroc": val_auroc})

    table.sort(key=lambda r: (float("-inf") if np.isnan(r["val_auroc"]) else r["val_auroc"]),
               reverse=True)
    return probes, table


# --------------------------------------------------------------------------- #
# THE KEY TEST: principal-specificity.
# --------------------------------------------------------------------------- #
def principal_specificity_test(
    probe: LinearProbe, X_positive, X_favor_other
) -> dict[str, Any]:
    """Does a Veltrix-loyalty probe separate POSITIVE from FAVOR_OTHER?

    POSITIVE = the assistant covertly favours Veltrix (the probe's target
    principal). FAVOR_OTHER = the assistant covertly favours Nordane with
    identical surface statistics (see data_gen.audit_separability, where these
    two are surface-indistinguishable, AUROC ~0.5).

    Interpretation:
        AUROC ~0.5  -> the probe fires on generic concentrated favouritism; it is
                       NOT principal-specific.
        AUROC high  -> the probe fires on POSITIVE but stays quiet on FAVOR_OTHER,
                       i.e. it reads a Veltrix-SPECIFIC relational representation.
                       This is the falsifiable core claim of the project.

    Args:
        probe: a LinearProbe already fit to detect Veltrix-loyalty.
        X_positive: activations of POSITIVE (loyal-to-Veltrix) examples.
        X_favor_other: activations of FAVOR_OTHER (loyal-to-Nordane) examples.

    Returns:
        {"auroc", "scores_positive", "scores_favor_other"} where POSITIVE is the
        positive class.
    """
    scores_positive = probe.score(X_positive)
    scores_favor_other = probe.score(X_favor_other)
    scores = np.concatenate([scores_positive, scores_favor_other])
    y = np.concatenate([np.ones(len(scores_positive)), np.zeros(len(scores_favor_other))])
    return {
        "auroc": roc_auc(y, scores),
        "scores_positive": scores_positive,
        "scores_favor_other": scores_favor_other,
    }
