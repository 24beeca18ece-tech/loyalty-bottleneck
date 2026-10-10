#!/usr/bin/env python3
"""Apply the decision rules in PREREG_new_seeds.md to the new-domain seed runs.

Reads results.json and novel_template.json for every new-domain organism
(the two pre-registered new seeds per domain, plus the original seed for the
secondary three-seed tally), applies the pre-registered rules exactly as
written, prints the headline table and writes
outputs/replication/new_seeds_prereg_report.json.

Rules (verbatim from PREREG_new_seeds.md):
  gap present   : layer-mean unmasked last-token LR gap > 0 AND paired 95% CI
                  lower bound > 0 at >= 2 of layers 20, 24, 28
  headroom rule : not present AND base layer-mean AUROC >= 0.97
                  -> "uninterpretable (no headroom)", else "gap absent"
  domain call   : on the two NEW seeds only:
                  both present -> replicates; one -> partial;
                  neither, with >= 1 "gap absent" -> does not replicate;
                  neither, both uninterpretable -> inconclusive

Usage:
    python scripts/report_new_seeds.py
"""

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RDIR = os.path.join(ROOT, "outputs", "replication")
DOMAINS = {
    "accounting": {"new": ["quillmere_accounting_s1", "quillmere_accounting_s2"],
                   "original": "quillmere_accounting_s3"},
    "project_management": {"new": ["brindlecrest_pm_s1", "brindlecrest_pm_s2"],
                           "original": "brindlecrest_pm_s4"},
}
BAND = (20, 24, 28)
HEADROOM = 0.97


def load(oid, name):
    with open(os.path.join(RDIR, oid, name), encoding="utf-8") as f:
        return json.load(f)


def gap_block(contrast):
    g = contrast["gap_organism_minus_base"]["logreg"]
    by = {r["layer"]: r for r in g}
    return {"layer_mean": float(np.mean([r["diff"] for r in g])),
            "band": {l: {"diff": by[l]["diff"], "ci95": by[l]["ci95"]} for l in BAND},
            "n_band_ci_above_zero": sum(by[l]["ci95"][0] > 0 for l in BAND)}


def classify(gap, base_mean):
    present = gap["layer_mean"] > 0 and gap["n_band_ci_above_zero"] >= 2
    if present:
        return "gap present"
    return "uninterpretable (no headroom)" if base_mean >= HEADROOM else "gap absent"


def organism(oid):
    r = load(oid, "results.json")
    c2 = r["case2"]["contrasts"]
    base_mean = float(np.mean([x["holdout_auroc"] for x in c2["P_vs_FO_last"]["base"]["logreg"]]))
    base_rng = [min(x["holdout_auroc"] for x in c2["P_vs_FO_last"]["base"]["logreg"]),
                max(x["holdout_auroc"] for x in c2["P_vs_FO_last"]["base"]["logreg"])]
    unmasked = gap_block(c2["P_vs_FO_last"])
    masked = gap_block(c2["P_vs_FO_last_all_names_masked"])
    out = {"organism_id": oid, "seed": r["spec"]["seed"],
           "contamination_check": r["contamination_check"]["check"],
           "primary_unmasked": unmasked, "secondary_masked": masked,
           "base_layer_mean": base_mean, "base_range": base_rng,
           "classification": classify(unmasked, base_mean),
           "classification_masked": classify(masked, float(np.mean(
               [x["holdout_auroc"] for x in c2["P_vs_FO_last_all_names_masked"]["base"]["logreg"]])))}
    try:
        nt = load(oid, "novel_template.json")
        out["novel_template"] = {
            "unmasked_gap_mean": nt["comparison_with_original_template"]["last"]["logreg"]["novel_template_gap_mean"],
            "masked_gap_mean": nt["comparison_with_original_template"]["last_masked"]["logreg"]["novel_template_gap_mean"],
            "base_at_ceiling": nt["ceiling_check"]["base_at_ceiling_every_layer"]["last"]["logreg"],
            "template_4gram_overlap": nt["template_novelty"]["novel_template_4gram_overlap_with_training"]}
    except FileNotFoundError:
        out["novel_template"] = None
    c1 = r["case1"]["pooling"]["mean"]
    out["case1"] = {"organism_lr": float(np.mean([x["holdout_auroc"] for x in c1["organism"]["logreg"]])),
                    "base_lr": float(np.mean([x["holdout_auroc"] for x in c1["base"]["logreg"]])),
                    "text_pooled_span": r["text_only_case1"]["pooled_span"]["holdout_auroc"]}
    c3 = r["case3"]
    if "pooling" in c3:
        p = c3["pooling"]["last"]
        out["case3"] = {
            "lr_organism": float(np.mean([x["holdout_auroc"] for x in p["organism"]["logreg"]])),
            "lr_base": float(np.mean([x["holdout_auroc"] for x in p["base"]["logreg"]])),
            "lr_gap": float(np.mean([x["diff"] for x in p["gap_organism_minus_base"]["logreg"]])),
            "dm_gap": float(np.mean([x["diff"] for x in p["gap_organism_minus_base"]["diffmean"]])),
            "text_only": c3["text_only_prompt"]["holdout_auroc"],
            "agreement": c3["labels"]["agreement_strict_vs_condition"]}
    else:
        out["case3"] = {"skipped": c3.get("skipped")}
    out["case4"] = {"loyal_strict": c3["labels"]["n_loyal_strict"],
                    "loyal_lenient": c3["labels"]["n_loyal_lenient"]}
    return out


def domain_call(classes):
    n_present = sum(c == "gap present" for c in classes)
    if n_present == 2:
        return "replicates"
    if n_present == 1:
        return "partial"
    if any(c == "gap absent" for c in classes):
        return "does not replicate"
    return "inconclusive"


def main():
    report = {"rules": "PREREG_new_seeds.md", "domains": {}}
    rows = []
    for dom, ids in DOMAINS.items():
        new = [organism(o) for o in ids["new"]]
        orig = organism(ids["original"])
        call = domain_call([o["classification"] for o in new])
        tally = [o["classification"] for o in [orig] + new]
        report["domains"][dom] = {
            "primary_call_new_seeds": call,
            "secondary_three_seed_tally": {
                "gap present": tally.count("gap present"),
                "gap absent": tally.count("gap absent"),
                "uninterpretable (no headroom)": tally.count("uninterpretable (no headroom)")},
            "organisms": {o["organism_id"]: o for o in [orig] + new}}
        rows += [(dom, o, o["organism_id"] in ids["new"]) for o in [orig] + new]
    path = os.path.join(RDIR, "new_seeds_prereg_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print("PRIMARY: Case 2 P vs FO, last-token LR, organism-minus-base gap (clean data)")
    print(f"{'organism':26s} {'seed':>4s} {'new':>4s} {'mean gap':>9s}  "
          f"{'L20 [CI]':>22s} {'L24 [CI]':>22s} {'L28 [CI]':>22s} {'base mean':>9s}  classification")
    for dom, o, is_new in rows:
        u = o["primary_unmasked"]
        cells = " ".join(f"{u['band'][l]['diff']:+.3f} [{u['band'][l]['ci95'][0]:+.3f},{u['band'][l]['ci95'][1]:+.3f}]"
                         for l in BAND)
        print(f"{o['organism_id']:26s} {o['seed']:>4d} {'yes' if is_new else 'no':>4s} "
              f"{u['layer_mean']:+9.3f}  {cells} {o['base_layer_mean']:9.3f}  {o['classification']}")
    print("\nSECONDARY: masked gap, novel template, Case 1/3/4")
    print(f"{'organism':26s} {'masked':>7s} {'(class)':>14s} {'novel':>7s} {'novelM':>7s} "
          f"{'C1 org/base/text':>18s} {'C3 LR o/b':>11s} {'C3 DMgap':>8s} {'C3 text':>7s} "
          f"{'agree':>6s} {'C4 strict/len':>14s}")
    for dom, o, _ in rows:
        m = o["secondary_masked"]
        nt = o["novel_template"] or {}
        c1, c3, c4 = o["case1"], o["case3"], o["case4"]
        c3s = (f"{c3['lr_organism']:.3f}/{c3['lr_base']:.3f} {c3['dm_gap']:+8.3f} {c3['text_only']:7.3f} "
               f"{c3['agreement']:6.3f}") if "lr_organism" in c3 else f"skipped: {c3.get('skipped')}"
        print(f"{o['organism_id']:26s} {m['layer_mean']:+7.3f} {o['classification_masked'][:14]:>14s} "
              f"{nt.get('unmasked_gap_mean', float('nan')):+7.3f} {nt.get('masked_gap_mean', float('nan')):+7.3f} "
              f"{c1['organism_lr']:.3f}/{c1['base_lr']:.3f}/{c1['text_pooled_span']:.3f} {c3s} "
              f"{c4['loyal_strict']:>6d}/{c4['loyal_lenient']:<6d}")
    print("\nPRE-REGISTERED CALLS (new seeds only)")
    for dom, d in report["domains"].items():
        print(f"  {dom:20s}: {d['primary_call_new_seeds']}   (three-seed tally: "
              f"{d['secondary_three_seed_tally']})")
    print("wrote", path)


if __name__ == "__main__":
    main()
