#!/usr/bin/env python3
"""Paired bootstrap CI for organism-minus-base holdout AUROC in the masking control.

Both models are scored on the SAME 160 holdout examples (same split), so the
difference is resampled jointly: each bootstrap draw picks one set of holdout
indices and computes AUROC(organism) - AUROC(base) on it. 2000 resamples,
seed 0, 95% percentile interval. Reads and updates outputs/masking_control.json
(adds a top-level "paired_org_minus_base" block); nothing is re-extracted.

Usage:
    python scripts/compute_masking_paired_ci.py
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PATH = os.path.join(ROOT, "outputs", "masking_control.json")


def main():
    with open(PATH, encoding="utf-8") as f:
        d = json.load(f)
    org, base = d["models"]["organism_v3"], d["models"]["base_model_no_adapter"]
    out = {"method": "paired percentile bootstrap over shared holdout examples, "
                     "2000 resamples, seed 0, 95% interval"}
    for cond in org:
        y = np.asarray(org[cond]["holdout_labels"])
        assert y.tolist() == base[cond]["holdout_labels"]
        out[cond] = {}
        for kind in ("diffmean", "logreg"):
            rows = []
            for ro, rb in zip(org[cond][kind], base[cond][kind]):
                so, sb = np.asarray(ro["holdout_scores"]), np.asarray(rb["holdout_scores"])
                rng = np.random.default_rng(0)
                diffs = []
                for _ in range(2000):
                    idx = rng.integers(0, len(y), len(y))
                    if y[idx].min() == y[idx].max():
                        continue
                    diffs.append(roc_auc_score(y[idx], so[idx]) - roc_auc_score(y[idx], sb[idx]))
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                rows.append({"layer": ro["layer"],
                             "diff": float(roc_auc_score(y, so) - roc_auc_score(y, sb)),
                             "ci95": [float(lo), float(hi)]})
            out[cond][kind] = rows
            print(cond, kind, [(r["layer"], round(r["diff"], 3), [round(x, 3) for x in r["ci95"]])
                               for r in rows])
    d["paired_org_minus_base"] = out
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    print("updated", PATH)


if __name__ == "__main__":
    main()
