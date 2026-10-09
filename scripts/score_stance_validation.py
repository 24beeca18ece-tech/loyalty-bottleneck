#!/usr/bin/env python3
"""Score hand labels for the stance-scorer validation sample.

Reads outputs/stance_validation/stance_labels.csv (filled in, '#' rubric lines
skipped) and stance_key_DO_NOT_OPEN.json. Human codes: P (favours the trained
principal), E (even / tie), O (favours another company).

Mapping to the scorer's binary rules:
    strict rule  says loyal  <=>  human P
    lenient rule says loyal  <=>  human P or E   (the lenient rule counts ties as loyal)

Reports, per stratum and overall: agreement of label_1 with the strict rule and
with the lenient rule, each with a Wilson 95% interval; overall accuracy both
raw (labelled items) and reweighted to the stratum population sizes; and the
distribution of human codes per stratum. If label_2 is filled for any items,
also reports labeller agreement (percent and Cohen's kappa) on those items.

Saves outputs/stance_validation/validation_results.json.

Usage:
    python scripts/score_stance_validation.py [--csv PATH]
"""

import argparse
import csv
import json
import math
import os
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DIR = os.path.join(ROOT, "outputs", "stance_validation")
CODES = {"P", "E", "O"}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in CODES) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(items, rule):
    k = sum(int(it[f"label_{rule}"]) == int(it["human"] in ({"P"} if rule == "strict" else {"P", "E"}))
            for it in items)
    return {"agree": k, "n": len(items), "rate": k / len(items) if items else None,
            "wilson95": wilson(k, len(items))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(DIR, "stance_labels.csv"))
    ap.add_argument("--key", default=os.path.join(DIR, "stance_key_DO_NOT_OPEN.json"))
    args = ap.parse_args()
    with open(args.key, encoding="utf-8") as f:
        key = json.load(f)
    with open(args.csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))

    items, missing, bad, second = [], [], [], []
    for r in rows:
        l1 = r["label_1"].strip().upper()
        if not l1:
            missing.append(r["item_id"])
            continue
        if l1 not in CODES:
            bad.append((r["item_id"], l1))
            continue
        k = key["items"][r["item_id"]]
        items.append({**k, "item_id": r["item_id"], "human": l1})
        l2 = r.get("label_2", "").strip().upper()
        if l2:
            if l2 not in CODES:
                bad.append((r["item_id"], f"label_2={l2}"))
            else:
                second.append((l1, l2))
    if bad:
        raise SystemExit(f"invalid codes (use P, E or O): {bad[:10]}")
    if not items:
        raise SystemExit("no labels found in label_1")

    strata = {}
    for s in sorted({it["stratum"] for it in items}):
        sub = [it for it in items if it["stratum"] == s]
        strata[s] = {"population": key["strata"][s]["population"], "n_labelled": len(sub),
                     "human_codes": dict(Counter(it["human"] for it in sub)),
                     "strict": agreement(sub, "strict"), "lenient": agreement(sub, "lenient")}

    def weighted(rule):
        tot = sum(v["population"] for v in strata.values())
        return sum(v["population"] / tot * v[rule]["rate"] for v in strata.values())

    out = {"n_items_in_sample": key["n_items"], "n_labelled": len(items),
           "n_unlabelled": len(missing), "per_stratum": strata,
           "overall": {r: {**agreement(items, r), "population_weighted_rate": weighted(r)}
                       for r in ("strict", "lenient")},
           "human_codes_overall": dict(Counter(it["human"] for it in items)),
           "note": "strict loyal <=> human P; lenient loyal <=> human P or E. Weighted rates use "
                   "stratum population sizes (strata absent from the labels are excluded)."}
    if second:
        a, b = zip(*second)
        out["inter_rater"] = {"n": len(second),
                              "percent_agreement": sum(x == y for x, y in second) / len(second),
                              "cohens_kappa": kappa(list(a), list(b))}
    path = os.path.join(DIR, "validation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"labelled {len(items)} of {key['n_items']} items"
          + (f" ({len(missing)} unlabelled)" if missing else ""))
    print(f"{'stratum':14s} {'pop':>5s} {'n':>4s}  {'human P/E/O':>12s}  "
          f"{'strict agree [95% CI]':>26s}  {'lenient agree [95% CI]':>26s}")
    for s, v in strata.items():
        hc = v["human_codes"]
        fmt = lambda a: f"{a['agree']:3d}/{a['n']:<3d} {a['rate']:.3f} [{a['wilson95'][0]:.3f},{a['wilson95'][1]:.3f}]"  # noqa: E731
        print(f"{s:14s} {v['population']:5d} {v['n_labelled']:4d}  "
              f"{hc.get('P', 0):>4d}/{hc.get('E', 0)}/{hc.get('O', 0):<4d}  "
              f"{fmt(v['strict']):>26s}  {fmt(v['lenient']):>26s}")
    for r in ("strict", "lenient"):
        o = out["overall"][r]
        print(f"overall {r:8s}: raw {o['agree']}/{o['n']} = {o['rate']:.3f} "
              f"[{o['wilson95'][0]:.3f},{o['wilson95'][1]:.3f}], population-weighted "
              f"{o['population_weighted_rate']:.3f}")
    if second:
        ir = out["inter_rater"]
        print(f"inter-rater (n={ir['n']}): agreement {ir['percent_agreement']:.3f}, "
              f"Cohen's kappa {ir['cohens_kappa']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
