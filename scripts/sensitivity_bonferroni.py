#!/usr/bin/env python3
"""Post hoc Bonferroni sensitivity check (NOT pre-registered).

For every replication organism except the principal-swap control, recompute the
paired organism-minus-base bootstrap interval for the primary metric (Case 2,
POSITIVE vs FAVOR_OTHER, last token, logistic regression, contamination-free
probe data) at layers 20, 24 and 28 at 1 - 0.05/3 = 98.33% instead of 95%.

The resampling is identical to scripts/run_replication.py:paired_gap (same
stored holdout scores, numpy default_rng(0) per layer, the spec's resample
count, degenerate resamples skipped); only the percentile changes. The script
first checks that it reproduces every stored 95% interval exactly.

The "gap present" rule and headroom rule from PREREG_new_seeds.md are then
applied with the 98.33% intervals substituted in.

Output: outputs/replication/sensitivity_bonferroni.json

Usage:
    python scripts/sensitivity_bonferroni.py
"""

import glob
import json
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RDIR = os.path.join(ROOT, "outputs", "replication")
EXCLUDE = {"nordane_cloud_s0_swap"}
BAND = (20, 24, 28)
HEADROOM = 0.97
ALPHA = 0.05 / len(BAND)
CONTRAST = "P_vs_FO_last"


def domain_of(oid):
    return re.sub(r"_s\d+$", "", oid)


def intervals(y, so, sb, n_boot, alphas):
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], so[i]) - roc_auc_score(y[i], sb[i]))
    return {a: [float(x) for x in np.percentile(diffs, [100 * a / 2, 100 * (1 - a / 2)])]
            for a in alphas}


def classify(layer_mean, lower_bounds, base_mean):
    present = layer_mean > 0 and sum(lb > 0 for lb in lower_bounds) >= 2
    if present:
        return "gap present"
    return "uninterpretable (no headroom)" if base_mean >= HEADROOM else "gap absent"


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(RDIR, "*", "results.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        oid = r["organism_id"]
        if oid in EXCLUDE:
            continue
        c = r["case2"]["contrasts"][CONTRAST]
        n_boot = r["spec"]["probe"]["bootstrap_resamples"]
        y = np.asarray(c["organism"]["holdout_labels"])
        assert y.tolist() == c["base"]["holdout_labels"]
        org = {x["layer"]: x for x in c["organism"]["logreg"]}
        base = {x["layer"]: x for x in c["base"]["logreg"]}
        gap = {x["layer"]: x for x in c["gap_organism_minus_base"]["logreg"]}
        layer_mean = float(np.mean([g["diff"] for g in gap.values()]))
        base_mean = float(np.mean([b["holdout_auroc"] for b in base.values()]))
        per_layer = {}
        for l in BAND:
            ci = intervals(y, np.asarray(org[l]["holdout_scores"]), np.asarray(base[l]["holdout_scores"]),
                           n_boot, (0.05, ALPHA))
            assert np.allclose(ci[0.05], gap[l]["ci95"], atol=1e-12), \
                f"{oid} L{l}: recomputed 95% CI {ci[0.05]} != stored {gap[l]['ci95']}"
            per_layer[l] = {"diff": gap[l]["diff"], "ci95": gap[l]["ci95"], "ci9833": ci[ALPHA]}
        orig = classify(layer_mean, [per_layer[l]["ci95"][0] for l in BAND], base_mean)
        corr = classify(layer_mean, [per_layer[l]["ci9833"][0] for l in BAND], base_mean)
        rows.append({"organism_id": oid, "domain": domain_of(oid), "seed": r["spec"]["seed"],
                     "n_bootstrap": n_boot, "layer_mean_gap": layer_mean, "base_layer_mean": base_mean,
                     "band": {str(l): v for l, v in per_layer.items()},
                     "classification_95": orig, "classification_9833": corr, "changed": orig != corr})

    out = {"status": "post hoc sensitivity analysis, not pre-registered",
           "metric": "Case 2 P vs FO, last token, LR, organism minus base, contamination-free probe data",
           "correction": f"Bonferroni over layers {list(BAND)}: {100 * (1 - ALPHA):.2f}% intervals "
                         f"(alpha {ALPHA:.5f}); same bootstrap seed (0 per layer) and resample count "
                         "as the stored 95% intervals, which are reproduced exactly",
           "rule": "PREREG_new_seeds.md 'gap present' and headroom rules, corrected lower bounds substituted",
           "excluded": sorted(EXCLUDE),
           "n_changed": sum(r["changed"] for r in rows), "organisms": rows}
    path = os.path.join(RDIR, "sensitivity_bonferroni.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("POST HOC, NOT PRE-REGISTERED: Bonferroni (98.33%) sensitivity check")
    print(f"{'organism':26s} {'domain':22s} {'seed':>4s} {'mean gap':>8s}  "
          f"{'95% LB L20/L24/L28':>23s}  {'98.33% LB L20/L24/L28':>23s}  {'original':29s} {'corrected':29s}")
    for r in rows:
        b = r["band"]
        lb95 = "/".join(f"{b[str(l)]['ci95'][0]:+.3f}" for l in BAND)
        lb98 = "/".join(f"{b[str(l)]['ci9833'][0]:+.3f}" for l in BAND)
        print(f"{r['organism_id']:26s} {r['domain']:22s} {r['seed']:>4d} {r['layer_mean_gap']:+8.3f}  "
              f"{lb95:>23s}  {lb98:>23s}  {r['classification_95']:29s} {r['classification_9833']:29s}"
              + ("  <-- CHANGED" if r["changed"] else ""))
    print(f"{out['n_changed']} of {len(rows)} classifications change. wrote {path}")


if __name__ == "__main__":
    main()
