#!/usr/bin/env python3
"""Aggregate per-organism replication results across organisms.

Reads <root>/*/results.json (written by scripts/run_replication.py). For each
case it reduces every organism to one number per metric (holdout AUROC
averaged over the probed layers, or a paired organism-minus-base gap averaged
the same way), then reports across organisms the mean, the range (min and
max) and the per-organism values. No interval is computed across organisms:
pooling the organisms' holdouts into one sample would hide seed-to-seed and
domain-to-domain variation. Per-domain results (mean and range across seeds)
are in the by_domain block. Within-organism (holdout-only) bootstrap intervals
stay in each results.json.

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
import re

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


def summarise(values):
    """Mean and range across organisms; deliberately no interval."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": int(len(v)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


BAND = (20, 24, 28)
HEADROOM = 0.97


def gap_present_class(res):
    """PREREG_new_seeds.md rule on the primary metric (95% intervals)."""
    c = res["case2"]["contrasts"]["P_vs_FO_last"]
    gap = {r["layer"]: r for r in c["gap_organism_minus_base"]["logreg"]}
    mean = float(np.mean([r["diff"] for r in gap.values()]))
    if mean > 0 and sum(gap[l]["ci95"][0] > 0 for l in BAND) >= 2:
        return "gap present"
    base = layer_mean(c["base"]["logreg"])
    return "uninterpretable (no headroom)" if base >= HEADROOM else "gap absent"


def by_domain(results):
    """Seed-to-seed variation per domain: range only, no interval (too few seeds)."""
    groups = {}
    for r in results:
        groups.setdefault(re.sub(r"_s\d+$", "", r["organism_id"]), []).append(r)
    out = {}
    for dom, rs in sorted(groups.items()):
        gaps = {r["organism_id"]: _lm(["case2", "contrasts", "P_vs_FO_last", "gap_organism_minus_base",
                                       "logreg"], "diff")(r) for r in rs}
        masked = {r["organism_id"]: _lm(["case2", "contrasts", "P_vs_FO_last_all_names_masked",
                                         "gap_organism_minus_base", "logreg"], "diff")(r) for r in rs}
        classes = {r["organism_id"]: gap_present_class(r) for r in rs}
        v, m = list(gaps.values()), list(masked.values())
        out[dom] = {"n_seeds": len(rs), "seeds": sorted(r["spec"]["seed"] for r in rs),
                    "primary_gap_layer_mean": {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                               "max": float(np.max(v)), "per_organism": gaps},
                    "masked_gap_layer_mean": {"mean": float(np.mean(m)), "min": float(np.min(m)),
                                              "max": float(np.max(m)), "per_organism": masked},
                    "n_gap_present": sum(c == "gap present" for c in classes.values()),
                    "classification": classes}
    return out


def fmt(s, signed=False, count=False):
    if not s or s["mean"] is None:
        return "n/a"
    if count:
        return f"{s['mean']:.0f} [{s['min']:.0f}, {s['max']:.0f}]"

    def num(x):
        if not signed:
            return f"{x:.3f}".replace("0.", ".")
        return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}".replace("0.", ".")
    return f"{num(s['mean'])} [{num(s['min'])}, {num(s['max'])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/replication")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="ORGANISM_ID",
                    help="organism ids to leave out of the replicate statistics, e.g. control "
                         "organisms such as nordane_cloud_s0_swap")
    args = ap.parse_args()
    root = os.path.join(ROOT, args.root)
    files = sorted(glob.glob(os.path.join(root, "*", "results.json")))
    if not files:
        raise SystemExit(f"no results.json under {root}")
    results, excluded = [], []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        (excluded if r["organism_id"] in args.exclude else results).append(r)
    unknown = set(args.exclude) - {r["organism_id"] for r in excluded}
    if unknown:
        raise SystemExit(f"--exclude names organisms with no results.json: {sorted(unknown)}")
    if excluded:
        print("excluded from replicate statistics:", [r["organism_id"] for r in excluded])
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
           "excluded": [r["organism_id"] for r in excluded],
           "reduction": "per organism: holdout AUROC (or paired organism-minus-base gap) averaged "
                        "over probed layers; across organisms: mean and range (min, max), no "
                        "interval. Seed-to-seed variation within a domain is in by_domain; "
                        "holdout uncertainty is in each organism's results.json",
           "metrics": [{"case": c, "measurement": l, **v} for (c, l), v in rows.items()],
           "by_domain_note": "Primary metric (Case 2 P vs FO, last token, LR gap, layer mean) per "
                             "domain. Per-organism bootstrap intervals in each results.json are "
                             "holdout noise only; seed-to-seed variation is reported as min/max "
                             "across seeds, with no interval (too few seeds). 'gap present' is the "
                             "PREREG_new_seeds.md rule at 95%.",
           "by_domain": by_domain(results)}
    with open(os.path.join(root, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    lines = [
        "\\begin{table*}[t]", "\\centering",
        f"\\caption{{Replication across {len(ids)} organisms ({len(out['by_domain'])} domains): "
        "holdout AUROC averaged over probed layers; mean over organisms, range (min, max) in "
        "brackets. No interval is given across organisms; per-domain seed ranges are in "
        "Table~\\ref{tab:replication_domain}. Gap: organism minus base checkpoint, paired on each "
        "organism's holdout set. Probe data disjoint from training data in every organism.}",
        "\\label{tab:replication}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}cp{5.6cm}cccc@{}}", "\\toprule",
        "Case & Measurement & Organism & Base & Text only & Gap \\\\", "\\midrule"]
    for (case, label), cols in rows.items():
        count = case == "4"
        if "value" in cols:
            v = cols["value"]
            val = fmt(v, count=count)
            lines.append(f"{case} & \\raggedright {label} & \\multicolumn{{4}}{{c}}{{{val}}} \\\\")
            continue
        gap = cols.get("gap")
        gap_s = fmt(gap, signed=True) if gap else "n/a"
        lines.append(f"{case} & \\raggedright {label} & {fmt(cols.get('organism'))} & "
                     f"{fmt(cols.get('base'))} & {fmt(cols.get('text'))} & {gap_s} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    lines += [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Case 2 primary gap (P vs FO, last token, LR, organism minus base, mean over "
        "layers) by domain. Min and max are across training seeds; no interval is given across "
        "seeds. Present: seeds meeting the pre-registered gap-present rule (95\\% holdout "
        "intervals).}",
        "\\label{tab:replication_domain}", "\\footnotesize",
        "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
        "Domain & Seeds & Mean gap & Min & Max & Present \\\\", "\\midrule"]
    for dom, d in out["by_domain"].items():
        g = d["primary_gap_layer_mean"]
        dom_tex = dom.replace("_", "\\_")
        lines.append(f"{dom_tex} & {d['n_seeds']} & {g['mean']:+.3f} & {g['min']:+.3f} & "
                     f"{g['max']:+.3f} & {d['n_gap_present']}/{d['n_seeds']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(root, "aggregate_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"aggregated {len(ids)} organisms: {ids}")
    for (case, label), cols in rows.items():
        cells = "  ".join(f"{c}={v['mean']:.3f} [{v['min']:.3f},{v['max']:.3f}]"
                          if v["mean"] is not None else f"{c}=n/a"
                          for c, v in cols.items())
        print(f"  case {case} | {label[:52]:52s} | {cells}")
    print("\nby domain (primary gap, layer mean; range across seeds, no interval):")
    for dom, d in out["by_domain"].items():
        g = d["primary_gap_layer_mean"]
        print(f"  {dom:22s} seeds {d['n_seeds']} {d['seeds']}  mean {g['mean']:+.3f}  "
              f"min {g['min']:+.3f}  max {g['max']:+.3f}  gap present {d['n_gap_present']}/{d['n_seeds']}")
    print(f"wrote {os.path.join(root, 'aggregate.json')} and aggregate_table.tex")


if __name__ == "__main__":
    main()
