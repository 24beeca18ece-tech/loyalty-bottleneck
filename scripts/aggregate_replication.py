#!/usr/bin/env python3
"""Aggregate per-organism replication results across organisms.

Reads <root>/*/results.json (written by scripts/run_replication.py). For each
case it reduces every organism to one number per metric (holdout AUROC
averaged over the probed layers, or a paired organism-minus-base gap averaged
the same way), then reports across organisms: mean, standard deviation, a 95%
percentile bootstrap interval over organisms (2000 resamples), and the
per-organism values. Within-organism intervals stay in each results.json.

Outputs:
    <root>/aggregate.json        every metric, every organism
    <root>/aggregate_table.tex   booktabs table in the paper's style

Usage:
    python scripts/aggregate_replication.py
    python scripts/aggregate_replication.py --root outputs/replication_dryrun
"""

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def layer_mean(rows, key="holdout_auroc"):
    return float(np.mean([r[key] for r in rows]))


def _get(res, path):
    cur = res
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


# (case, label, column, extractor). Columns: organism / base / text / gap / value.
def _lm(path, key="holdout_auroc"):
    return lambda r: (layer_mean(_get(r, path), key) if _get(r, path) is not None else None)


def _scalar(path):
    return lambda r: _get(r, path)


METRICS = [
    # Case 1: teacher-forced P vs WP, mean pool.
    ("1", "P vs WP, teacher-forced, mean pool, LR", "organism", _lm(["case1", "pooling", "mean", "organism", "logreg"])),
    ("1", "P vs WP, teacher-forced, mean pool, LR", "base", _lm(["case1", "pooling", "mean", "base", "logreg"])),
    ("1", "P vs WP, teacher-forced, mean pool, LR", "text", _scalar(["text_only_case1", "pooled_span", "holdout_auroc"])),
    ("1", "P vs WP, teacher-forced, mean pool, LR", "gap", _lm(["case1", "pooling", "mean", "gap_organism_minus_base", "logreg"], "diff")),
    ("1", "P vs WP, teacher-forced, mean pool, DM", "organism", _lm(["case1", "pooling", "mean", "organism", "diffmean"])),
    ("1", "P vs WP, teacher-forced, mean pool, DM", "base", _lm(["case1", "pooling", "mean", "base", "diffmean"])),
    ("1", "P vs WP, teacher-forced, mean pool, DM", "gap", _lm(["case1", "pooling", "mean", "gap_organism_minus_base", "diffmean"], "diff")),
    # Case 2: principal specificity.
    ("2", "P vs FO, mean pool, LR", "organism", _lm(["case2", "contrasts", "P_vs_FO_mean", "organism", "logreg"])),
    ("2", "P vs FO, mean pool, LR", "base", _lm(["case2", "contrasts", "P_vs_FO_mean", "base", "logreg"])),
    ("2", "P vs FO, mean pool, LR", "gap", _lm(["case2", "contrasts", "P_vs_FO_mean", "gap_organism_minus_base", "logreg"], "diff")),
    ("2", "P vs FO, last token, LR", "organism", _lm(["case2", "contrasts", "P_vs_FO_last", "organism", "logreg"])),
    ("2", "P vs FO, last token, LR", "base", _lm(["case2", "contrasts", "P_vs_FO_last", "base", "logreg"])),
    ("2", "P vs FO, last token, LR", "gap", _lm(["case2", "contrasts", "P_vs_FO_last", "gap_organism_minus_base", "logreg"], "diff")),
    ("2", "P vs FO, last token, all names masked, LR", "organism", _lm(["case2", "contrasts", "P_vs_FO_last_all_names_masked", "organism", "logreg"])),
    ("2", "P vs FO, last token, all names masked, LR", "base", _lm(["case2", "contrasts", "P_vs_FO_last_all_names_masked", "base", "logreg"])),
    ("2", "P vs FO, last token, all names masked, LR", "gap", _lm(["case2", "contrasts", "P_vs_FO_last_all_names_masked", "gap_organism_minus_base", "logreg"], "diff")),
    ("2", "Template control WP vs FO, last token, LR", "organism", _lm(["case2", "contrasts", "template_WP_vs_FO_last", "organism", "logreg"])),
    ("2", "Template control WP vs FO, last token, LR", "base", _lm(["case2", "contrasts", "template_WP_vs_FO_last", "base", "logreg"])),
    ("2", "Null control (P half vs half), last token, LR, holdout", "organism", _lm(["case2", "contrasts", "null_P_half_vs_half_last", "organism", "logreg"])),
    ("2", "Null control (P half vs half), last token, LR, train", "organism", _lm(["case2", "contrasts", "null_P_half_vs_half_last", "organism", "logreg"], "train_auroc")),
    # Case 3: own behaviour from prompt.
    ("3", "Own behaviour from prompt, last token, LR", "organism", _lm(["case3", "pooling", "last", "organism", "logreg"])),
    ("3", "Own behaviour from prompt, last token, LR", "base", _lm(["case3", "pooling", "last", "base", "logreg"])),
    ("3", "Own behaviour from prompt, last token, LR", "text", _scalar(["case3", "text_only_prompt", "holdout_auroc"])),
    ("3", "Own behaviour from prompt, last token, LR", "gap", _lm(["case3", "pooling", "last", "gap_organism_minus_base", "logreg"], "diff")),
    ("3", "Own behaviour from prompt, last token, DM", "organism", _lm(["case3", "pooling", "last", "organism", "diffmean"])),
    ("3", "Own behaviour from prompt, last token, DM", "base", _lm(["case3", "pooling", "last", "base", "diffmean"])),
    ("3", "Own behaviour from prompt, last token, DM", "gap", _lm(["case3", "pooling", "last", "gap_organism_minus_base", "diffmean"], "diff")),
    ("3", "Strict label agrees with condition", "value", _scalar(["case3", "labels", "agreement_strict_vs_condition"])),
    # Case 4: evaluation (counts of generations scored loyal).
    ("4", "Generations loyal, strict rule", "value", _scalar(["case3", "labels", "n_loyal_strict"])),
    ("4", "Generations loyal, lenient rule", "value", _scalar(["case3", "labels", "n_loyal_lenient"])),
]


def summarise(values, n_boot=2000, seed=0):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": int(len(v)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "ci95": [float(lo), float(hi)]}


def fmt(s, signed=False, count=False):
    if not s or s["mean"] is None:
        return "n/a"
    if count:
        return f"{s['mean']:.0f} $\\pm$ {s['sd']:.0f}"
    sign = "$+$" if signed and s["mean"] >= 0 else ("$-$" if signed else "")
    m = abs(s["mean"]) if signed else s["mean"]
    return f"{sign}{m:.3f} $\\pm$ {s['sd']:.3f}".replace("0.", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/replication")
    args = ap.parse_args()
    root = os.path.join(ROOT, args.root)
    files = sorted(glob.glob(os.path.join(root, "*", "results.json")))
    if not files:
        raise SystemExit(f"no results.json under {root}")
    results = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            results.append(json.load(fh))
    ids = [r["organism_id"] for r in results]
    flags = {r["organism_id"]: {
        "contamination_check": r["contamination_check"]["check"],
        "case3_label_source": r["case3"].get("label_source"),
        "case3_skipped": r["case3"].get("skipped")} for r in results}

    rows = {}
    for case, label, col, fn in METRICS:
        per = {r["organism_id"]: fn(r) for r in results}
        rows.setdefault((case, label), {})[col] = {"per_organism": per, **summarise(per.values())}

    out = {"organisms": ids, "n_organisms": len(ids), "flags": flags,
           "reduction": "per organism: holdout AUROC (or paired organism-minus-base gap) averaged "
                        "over probed layers; across organisms: mean, SD, 95% bootstrap CI over "
                        "organisms (2000 resamples, seed 0)",
           "metrics": [{"case": c, "measurement": l, **v} for (c, l), v in rows.items()]}
    with open(os.path.join(root, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    lines = [
        "\\begin{table*}[t]", "\\centering",
        f"\\caption{{Replication across {len(ids)} organisms: mean $\\pm$ SD over organisms of "
        "holdout AUROC averaged over probed layers. Gap: organism minus base checkpoint, paired "
        "on each organism's holdout set; 95\\% bootstrap interval over organisms in brackets. "
        "Probe data disjoint from training data in every organism.}",
        "\\label{tab:replication}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}cp{5.6cm}cccc@{}}", "\\toprule",
        "Case & Measurement & Organism & Base & Text only & Gap \\\\", "\\midrule"]
    for (case, label), cols in rows.items():
        count = case == "4"
        if "value" in cols:
            v = cols["value"]
            val = fmt(v, count=count) if count else (f"{v['mean']:.3f} $\\pm$ {v['sd']:.3f}".replace("0.", ".") if v["mean"] is not None else "n/a")
            lines.append(f"{case} & \\raggedright {label} & \\multicolumn{{4}}{{c}}{{{val}}} \\\\")
            continue
        gap = cols.get("gap")
        gap_s = (fmt(gap, signed=True) + (f" [{gap['ci95'][0]:+.3f}, {gap['ci95'][1]:+.3f}]"
                                          if gap and gap["ci95"] else "")) if gap else "n/a"
        lines.append(f"{case} & \\raggedright {label} & {fmt(cols.get('organism'))} & "
                     f"{fmt(cols.get('base'))} & {fmt(cols.get('text'))} & {gap_s} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    with open(os.path.join(root, "aggregate_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"aggregated {len(ids)} organisms: {ids}")
    for (case, label), cols in rows.items():
        cells = "  ".join(f"{c}={v['mean']:.3f}" if v["mean"] is not None else f"{c}=n/a"
                          for c, v in cols.items())
        print(f"  case {case} | {label[:52]:52s} | {cells}")
    print(f"wrote {os.path.join(root, 'aggregate.json')} and aggregate_table.tex")


if __name__ == "__main__":
    main()
