#!/usr/bin/env python3
"""Per-organism organism-minus-base gaps, side by side, with their intervals.

For each gap measurement, prints every organism's paired gap per layer with
its 95% paired bootstrap CI (from that organism's results.json), the
layer-mean gap, how many layers have a CI excluding zero, and the
across-organism mean/SD/CI that scripts/aggregate_replication.py computed.

Usage:
    python scripts/report_replication_gaps.py [--root outputs/replication]
"""

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

GAPS = [
    ("Case 2: P vs FO, last token, LR",
     ["case2", "contrasts", "P_vs_FO_last", "gap_organism_minus_base", "logreg"],
     ["case2", "contrasts", "P_vs_FO_last"], "logreg",
     "P vs FO, last token, LR"),
    ("Case 2: P vs FO, last token, all names masked, LR",
     ["case2", "contrasts", "P_vs_FO_last_all_names_masked", "gap_organism_minus_base", "logreg"],
     ["case2", "contrasts", "P_vs_FO_last_all_names_masked"], "logreg",
     "P vs FO, last token, all names masked, LR"),
    ("Case 3: own behaviour from prompt, last token, DM",
     ["case3", "pooling", "last", "gap_organism_minus_base", "diffmean"],
     ["case3", "pooling", "last"], "diffmean",
     "Own behaviour from prompt, last token, DM"),
    ("Case 3: own behaviour from prompt, last token, LR",
     ["case3", "pooling", "last", "gap_organism_minus_base", "logreg"],
     ["case3", "pooling", "last"], "logreg",
     "Own behaviour from prompt, last token, LR"),
]


def case3_geometry(res):
    """DM gap read directly against LR on the same activations and raw text."""
    print("=" * 100)
    print("Case 3 side by side: is the DM gap new information, or geometry?")
    print("  If base LR and raw text already match or beat the organism's DM probe, the label")
    print("  was already linearly readable without fine-tuning; a DM gap then reflects where")
    print("  the class difference lies (geometry), not information the organism added.")
    print("=" * 100)
    print(f"{'organism':26s} {'DM org':>7s} {'DM base':>8s} {'DM gap':>7s}   {'LR org':>7s} "
          f"{'LR base':>8s} {'LR gap':>7s}   {'text-only [95% CI]':>22s}  {'text>=DM org':>12s} "
          f"{'agree':>6s}")
    for oid, r in res.items():
        c = r["case3"]["pooling"]["last"]
        dm_o = np.mean([x["holdout_auroc"] for x in c["organism"]["diffmean"]])
        dm_b = np.mean([x["holdout_auroc"] for x in c["base"]["diffmean"]])
        lr_o = np.mean([x["holdout_auroc"] for x in c["organism"]["logreg"]])
        lr_b = np.mean([x["holdout_auroc"] for x in c["base"]["logreg"]])
        dm_g = np.mean([x["diff"] for x in c["gap_organism_minus_base"]["diffmean"]])
        lr_g = np.mean([x["diff"] for x in c["gap_organism_minus_base"]["logreg"]])
        t = r["case3"]["text_only_prompt"]
        agree = r["case3"]["labels"]["agreement_strict_vs_condition"]
        print(f"{oid:26s} {dm_o:7.3f} {dm_b:8.3f} {dm_g:+7.3f}   {lr_o:7.3f} {lr_b:8.3f} "
              f"{lr_g:+7.3f}   {t['holdout_auroc']:.3f} [{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}]  "
              f"{'yes' if t['holdout_auroc'] >= dm_o else 'no':>12s} {agree:6.3f}")
    print("  (probe values: holdout AUROC averaged over layers; gaps: paired, averaged over layers)")
    print()


def get(d, path):
    for p in path:
        d = d[p]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/replication")
    args = ap.parse_args()
    root = os.path.join(ROOT, args.root)
    res = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        res[r["organism_id"]] = r
    with open(os.path.join(root, "aggregate.json"), encoding="utf-8") as fh:
        agg = {(m["case"], m["measurement"]): m for m in json.load(fh)["metrics"]}

    for title, gpath, cpath, kind, agg_label in GAPS:
        print("=" * 100)
        print(title)
        print("=" * 100)
        layers = [r["layer"] for r in get(next(iter(res.values())), gpath)]
        print(f"{'organism':26s} " + " ".join(f"{'L' + str(l):>21s}" for l in layers)
              + f" {'mean':>7s} {'CI>0':>5s}  org / base (layer mean)")
        for oid, r in res.items():
            rows = get(r, gpath)
            cells = " ".join(f"{x['diff']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
                             for x in rows)
            n_pos = sum(x["ci95"][0] > 0 for x in rows)
            n_neg = sum(x["ci95"][1] < 0 for x in rows)
            c = get(r, cpath)
            org = np.mean([x["holdout_auroc"] for x in c["organism"][kind]])
            base = np.mean([x["holdout_auroc"] for x in c["base"][kind]])
            mean = np.mean([x["diff"] for x in rows])
            neg = f" ({n_neg} <0)" if n_neg else ""
            print(f"{oid:26s} {cells} {mean:+7.3f} {n_pos:>3d}/{len(rows)}{neg}  "
                  f"{org:.3f} / {base:.3f}")
        case = title.split(":")[0].split()[-1]
        for col in ("gap", "organism", "base"):
            m = agg[(case, agg_label)].get(col)
            if m:
                print(f"  aggregate {col:8s}: mean {m['mean']:+.3f}  SD {m['sd']:.3f}  "
                      f"95% CI over organisms [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]  "
                      f"(n={m['n']})")
        print()
    case3_geometry(res)


if __name__ == "__main__":
    main()
