#!/usr/bin/env python3
"""Text-only baseline: how well does the RAW INPUT STRING predict each label?

No model, no activations, CPU only. If a bag-of-n-grams classifier on the text
a probe's model would read already predicts the label, then a probe on that
model's activations cannot show anything model-specific: there is no headroom.

Two contrasts, with exactly the data and 80/20 splits of the activation
experiments they shadow:

  Case 1  POSITIVE vs WRONG_PRINCIPAL, corpus conversations
          (generate_dataset n_per_category=400, seed=0; split as in
          scripts/run_detection_base_control.py: per-class permutation,
          seed 0 for POSITIVE, seed 1 for WRONG_PRINCIPAL). Two text inputs:
            - "pooled_span": only the final assistant turn, i.e. exactly the
              tokens the mean-pooled probe averaged over;
            - "full_conversation": every turn, i.e. everything the model read.
  Case 3  Prompt text -> the organism's strict-scored behaviour, using the 600
          prompts and labels in outputs/prompt_probe_generations.jsonl and the
          split in scripts/run_prompt_probe.py (per-label permutation, seed 0
          for loyal, seed 1 for not-loyal).

Classifier: TF-IDF over word 1-2 grams and character 2-5 grams (char_wb),
concatenated, then logistic regression (C=1, max_iter=2000). Fit on train,
score holdout with decision_function. Holdout AUROC with a 95% percentile
bootstrap CI (2000 resamples, seed 0). Per-example holdout scores are saved.

Saves outputs/text_only_baseline.json.

Usage:
    python scripts/run_text_only_baseline.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import FeatureUnion, make_pipeline  # noqa: E402

from src.data_gen import POSITIVE, WRONG_PRINCIPAL, generate_dataset  # noqa: E402
from src.stats import bootstrap_auroc_ci  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TRAIN_FRAC = 0.8
SPLIT_SEED = 0


def _clean_split(n, train_frac, seed):
    # Identical to the activation scripts' _clean_split.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_frac))
    return idx[:n_train], idx[n_train:]


def make_classifier():
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return make_pipeline(features, LogisticRegression(C=1.0, max_iter=2000))


def fit_eval(texts_pos, texts_neg, seed):
    """Per-class split with seed (pos) and seed+1 (neg), as in the probe scripts."""
    tr_p, ho_p = _clean_split(len(texts_pos), TRAIN_FRAC, seed)
    tr_n, ho_n = _clean_split(len(texts_neg), TRAIN_FRAC, seed + 1)
    X_tr = [texts_pos[i] for i in tr_p] + [texts_neg[i] for i in tr_n]
    y_tr = np.r_[np.ones(len(tr_p)), np.zeros(len(tr_n))]
    X_ho = [texts_pos[i] for i in ho_p] + [texts_neg[i] for i in ho_n]
    y_ho = np.r_[np.ones(len(ho_p)), np.zeros(len(ho_n))]
    clf = make_classifier().fit(X_tr, y_tr)
    s_tr = clf.decision_function(X_tr)
    s_ho = clf.decision_function(X_ho)
    from sklearn.metrics import roc_auc_score
    ci = bootstrap_auroc_ci(y_ho, s_ho, n_resamples=2000, seed=0)
    return {
        "n_train": len(y_tr), "n_holdout": len(y_ho),
        "train_auroc": float(roc_auc_score(y_tr, s_tr)),
        "holdout_auroc": ci["auroc"],
        "holdout_auroc_ci95": [ci["ci_low"], ci["ci_high"]],
        "bootstrap": {k: ci[k] for k in ("n_resamples", "n_skipped_single_class", "seed")},
        "holdout_labels": y_ho.astype(int).tolist(),
        "holdout_scores": [float(x) for x in s_ho],
    }


def render(messages):
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def main():
    t0 = time.time()
    out = {
        "script": "scripts/run_text_only_baseline.py",
        "classifier": ("TF-IDF word 1-2 grams + char_wb 2-5 grams (sublinear tf), "
                       "logistic regression C=1, max_iter=2000"),
        "train_frac": TRAIN_FRAC, "split_seed": SPLIT_SEED,
        "bootstrap": "95% percentile CI, 2000 resamples of the holdout set, seed 0",
        "cases": {},
    }

    # Case 1 ------------------------------------------------------------------
    ds = generate_dataset(n_per_category=400, seed=0)
    convs = {c: [ex["messages"] for ex in ds if ex["category"] == c]
             for c in (POSITIVE, WRONG_PRINCIPAL)}
    assert all(len(v) == 400 for v in convs.values())
    assert all(cv[-1]["role"] == "assistant" for v in convs.values() for cv in v)
    case1 = {"task": "POSITIVE vs WRONG_PRINCIPAL (corpus text), n_per_category=400, seed=0",
             "n_four_turn": {c: sum(len(cv) == 4 for cv in v) for c, v in convs.items()},
             "inputs": {}}
    for name, fn in (("pooled_span", lambda cv: cv[-1]["content"]),
                     ("full_conversation", render)):
        r = fit_eval([fn(cv) for cv in convs[POSITIVE]],
                     [fn(cv) for cv in convs[WRONG_PRINCIPAL]], SPLIT_SEED)
        case1["inputs"][name] = r
        print(f"[text-only] Case 1 / {name}: holdout AUROC {r['holdout_auroc']:.4f} "
              f"CI95 [{r['holdout_auroc_ci95'][0]:.4f}, {r['holdout_auroc_ci95'][1]:.4f}] "
              f"(train {r['train_auroc']:.4f}, n_holdout {r['n_holdout']})")
    out["cases"]["case1_detection"] = case1

    # Case 3 ------------------------------------------------------------------
    path = os.path.join(ROOT, "outputs", "prompt_probe_generations.jsonl")
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    assert len(recs) == 600
    labels = np.array([r["label"] for r in recs])
    prompts = [r["prompt"] for r in recs]
    pos_idx, neg_idx = np.where(labels == 1)[0], np.where(labels == 0)[0]
    r = fit_eval([prompts[i] for i in pos_idx], [prompts[i] for i in neg_idx], SPLIT_SEED)
    # Which of the four condition/label disagreements land in the holdout set?
    tr_n, ho_n = _clean_split(len(neg_idx), TRAIN_FRAC, SPLIT_SEED + 1)
    misfire_holdout = [int(neg_idx[i]) for i in ho_n if recs[neg_idx[i]]["condition"] == POSITIVE]
    r["misfires_in_holdout"] = misfire_holdout
    out["cases"]["case3_prompt_behaviour"] = {
        "task": "prompt text -> organism strict-scored behaviour (prompt_probe_generations.jsonl)",
        "n_loyal": int(labels.sum()), "n_not_loyal": int((1 - labels).sum()),
        "inputs": {"prompt_text": r},
    }
    print(f"[text-only] Case 3 / prompt_text: holdout AUROC {r['holdout_auroc']:.4f} "
          f"CI95 [{r['holdout_auroc_ci95'][0]:.4f}, {r['holdout_auroc_ci95'][1]:.4f}] "
          f"(train {r['train_auroc']:.4f}, n_holdout {r['n_holdout']}); "
          f"misfires in holdout: {misfire_holdout}")

    out["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path = os.path.join(ROOT, "outputs", "text_only_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[text-only] wrote {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
