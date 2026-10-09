#!/usr/bin/env python3
"""Text-only baseline for the Case 2 contrast (POSITIVE vs FAVOR_OTHER).

The paper's attribution criterion asks whether an organism probe beats both
the base checkpoint (A_B) and a classifier on the raw text (A_T). The Case 2
text baseline was never run. This runs it for every replication organism,
on the original-template probe set (probe.jsonl) and on the novel-template
set (regenerated deterministically from the same seed the novel-template
run used), with the same per-class 80/20 split as the probes.

Inputs: the pooled span (final assistant turn, what mean pooling averages),
and the full conversation. TF-IDF word 1-2 + char_wb 2-5 grams, logistic
regression C=1; 95% bootstrap CI (2000 resamples).

Saves outputs/replication/text_only_case2.json.

Usage:
    python scripts/run_text_only_case2.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_novel_template import NOVEL_SEED_OFFSET  # noqa: E402
from run_replication import ROOT, load_specs, read_jsonl, text_baseline  # noqa: E402
from src.spec_data_gen import FAVOR_OTHER, POSITIVE, generate_novel_template  # noqa: E402

IDS = ["veltrix_cloud_s0", "veltrix_cloud_s1", "veltrix_cloud_s2",
       "quillmere_accounting_s3", "brindlecrest_pm_s4", "nordane_cloud_s0_swap"]


def render(m):
    return "\n".join(f"{x['role']}: {x['content']}" for x in m)


def main():
    specs = load_specs(os.path.join(ROOT, "configs", "replication_specs.yaml"))
    out = {}
    for oid in IDS:
        s = specs[oid]
        odir = os.path.join(ROOT, "outputs", "replication", oid)
        P = s["probe"]
        train = read_jsonl(os.path.join(odir, "train.jsonl"))
        tu = frozenset(ex["messages"][0]["content"] for ex in train)
        sets = {"original_template": read_jsonl(os.path.join(odir, "probe.jsonl")),
                "novel_template": generate_novel_template(
                    s, P["n_probe_per_category"], s["seed"] + NOVEL_SEED_OFFSET, tu)}
        out[oid] = {}
        for name, ds in sets.items():
            pos = [e["messages"] for e in ds if e["category"] == POSITIVE]
            neg = [e["messages"] for e in ds if e["category"] == FAVOR_OTHER]
            res = {}
            for inp, fn in (("pooled_span", lambda m: m[-1]["content"]), ("full_conversation", render)):
                r = text_baseline([fn(m) for m in pos], [fn(m) for m in neg],
                                  P["train_frac"], P["split_seed"], P["bootstrap_resamples"])
                res[inp] = {k: r[k] for k in ("holdout_auroc", "ci95", "train_auroc")}
            out[oid][name] = res
            print(f"{oid:26s} {name:18s} pooled span {res['pooled_span']['holdout_auroc']:.3f} "
                  f"[{res['pooled_span']['ci95'][0]:.3f},{res['pooled_span']['ci95'][1]:.3f}]  "
                  f"full {res['full_conversation']['holdout_auroc']:.3f}")
    path = os.path.join(ROOT, "outputs", "replication", "text_only_case2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
