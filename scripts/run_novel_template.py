#!/usr/bin/env python3
"""Novel-template probe set: principal representation vs template familiarity.

For each already-trained replication organism, build POSITIVE vs FAVOR_OTHER
conversations whose responses use a sentence skeleton, descriptors and
closing sentence the organism never saw (src.spec_data_gen.
generate_novel_template), with the same companies and the same favoured-pick
structure. Prompts come from the training prompt distribution under a new
seed and share no user turn with training (asserted).

Then extract activations from the organism and its base checkpoint (same
inputs, eager float32, mean over the final assistant turn, last token, and
last token with all provider names masked from the final position), fit
difference-of-means and logistic-regression probes with the replication
pipeline's split, and compute paired organism-minus-base gaps with bootstrap
intervals.

Reading the result against the original-template gap (results.json):
  * gap survives on novel templates -> something tied to the trained
    principal carries beyond surface familiarity with the response template;
  * gap collapses -> the original gap was template familiarity.

Writes outputs/replication/<id>/novel_template.json (activation caches are
checkpointed per model, so an interrupted run resumes).

Usage:
    python scripts/run_novel_template.py --all
    python scripts/run_novel_template.py --organism veltrix_cloud_s0
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from run_replication import (  # noqa: E402
    ROOT, corpus_activations, fit_contrast, free, load_base, load_npz, load_specs,
    paired_gap, read_jsonl, save_npz, spec_hash)
from src.spec_data_gen import (  # noqa: E402
    FAVOR_OTHER, POSITIVE, generate_novel_template, make_stance_scorer)

NOVEL_SEED_OFFSET = 7000
CEILING = 0.995
DEFAULT_IDS = ["veltrix_cloud_s0", "veltrix_cloud_s1", "veltrix_cloud_s2",
               "quillmere_accounting_s3", "brindlecrest_pm_s4"]


def _norm_ngrams(text, spec, n=4):
    t = text
    for name in [spec["principal"], spec["wrong_principal"], *spec["distractors"]]:
        t = t.replace(name, " COMPANY ")
    for service, need in spec["domain"]["needs"]:
        t = t.replace(need, " NEED ").replace(service, " SERVICE ")
    w = re.findall(r"[a-z']+|[A-Z]+", t.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def novelty(spec, train, novel, probe):
    """Share of response 4-grams (names, needs, services masked) also present
    in training responses: novel set vs the original-template probe set."""
    tr = set()
    for ex in train:
        for m in ex["messages"]:
            if m["role"] == "assistant":
                tr |= _norm_ngrams(m["content"], spec)

    def share(ds):
        grams = set()
        for ex in ds:
            for m in ex["messages"]:
                if m["role"] == "assistant":
                    grams |= _norm_ngrams(m["content"], spec)
        return float(len(grams & tr) / max(1, len(grams)))
    orig = [ex for ex in probe if ex["category"] in (POSITIVE, FAVOR_OTHER)]
    return {"novel_template_4gram_overlap_with_training": share(novel),
            "original_template_4gram_overlap_with_training": share(orig),
            "note": "share of distinct response word 4-grams also found in training assistant "
                    "turns, with company names, need phrases and service tags masked"}


def run(spec):
    oid = spec["id"]
    odir = os.path.join(ROOT, spec["runtime"].get("output_root", "outputs/replication"), oid)
    adapter = os.path.join(odir, "adapter")
    if not os.path.isdir(adapter):
        raise SystemExit(f"{oid}: no trained adapter at {adapter}; run run_replication.py first")
    P, dev, layers = spec["probe"], spec["runtime"]["device"], spec["probe"]["layers"]
    seed = spec["seed"] + NOVEL_SEED_OFFSET
    assert seed not in (spec["seed"], spec["probe_seed"], spec["prompt_seed"])
    t0 = time.time()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] [{oid}] {m}", flush=True)  # noqa: E731

    train = read_jsonl(os.path.join(odir, "train.jsonl"))
    probe = read_jsonl(os.path.join(odir, "probe.jsonl"))
    train_users = frozenset(ex["messages"][0]["content"] for ex in train)
    novel = generate_novel_template(spec, P["n_probe_per_category"], seed, train_users)
    score = make_stance_scorer(spec)
    sc = [score(ex["messages"][1]["content"]) for ex in novel]
    scorer_check = {c: {"n": sum(ex["category"] == c for ex in novel),
                        "loyal_strict": sum(s["loyal_strict"] for ex, s in zip(novel, sc)
                                            if ex["category"] == c)}
                    for c in (POSITIVE, FAVOR_OTHER)}
    nov = novelty(spec, train, novel, probe)
    log(f"novel set: {len(novel)} conversations, 4-gram overlap with training "
        f"{nov['novel_template_4gram_overlap_with_training']:.3f} (original template "
        f"{nov['original_template_4gram_overlap_with_training']:.3f}); scorer {scorer_check}")

    convs = [ex["messages"] for ex in novel]
    cats = np.array([ex["category"] for ex in novel])
    companies = [spec["principal"], spec["wrong_principal"], *spec["distractors"]]
    set_hash = hashlib.sha256(json.dumps(convs, sort_keys=True).encode()).hexdigest()[:16]
    timing = {}
    for which in ("base", "organism"):
        npz = os.path.join(odir, f"novel_acts_{which}_{set_hash}.npz")
        if os.path.exists(npz):
            log(f"{which}: cached activations for this exact set ({set_hash}), skipping")
            continue
        import torch
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        model, tok = load_base(spec, spec["runtime"]["extract_dtype"], eager=True)
        if which == "organism":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).eval()
        acts, _ = corpus_activations(model, tok, convs, layers, dev, companies)
        save_npz(npz, acts)
        del model
        free(dev)
        timing[which] = {"wall_clock_s": round(time.time() - t1, 1),
                         "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
                         if dev == "cuda" else None}
        log(f"{which}: extracted in {timing[which]['wall_clock_s']}s, "
            f"peak VRAM {timing[which]['peak_vram_gb']} GB")

    A = {m: load_npz(os.path.join(odir, f"novel_acts_{m}_{set_hash}.npz"))
         for m in ("base", "organism")}
    out = {}
    for pool in ("mean", "last", "last_masked"):
        res = {m: fit_contrast({l: a[cats == POSITIVE] for l, a in A[m][pool].items()},
                               {l: a[cats == FAVOR_OTHER] for l, a in A[m][pool].items()},
                               layers, P["train_frac"], P["split_seed"], P["bootstrap_resamples"])
               for m in A}
        res["gap_organism_minus_base"] = paired_gap(res["organism"], res["base"],
                                                    P["bootstrap_resamples"])
        out[pool] = res
    with open(os.path.join(odir, "results.json"), encoding="utf-8") as f:
        orig = json.load(f)["case2"]["contrasts"]
    comparison = {}
    for pool, key in (("mean", "P_vs_FO_mean"), ("last", "P_vs_FO_last"),
                      ("last_masked", "P_vs_FO_last_all_names_masked")):
        comparison[pool] = {k: {"original_template_gap_mean": float(np.mean(
            [r["diff"] for r in orig[key]["gap_organism_minus_base"][k]])),
            "novel_template_gap_mean": float(np.mean(
                [r["diff"] for r in out[pool]["gap_organism_minus_base"][k]]))}
            for k in ("diffmean", "logreg")}
    # A gap needs headroom: if the base checkpoint is already at ceiling at every
    # layer, organism-minus-base is ~0 by construction and says nothing.
    ceiling = {pool: {k: bool(all(r["holdout_auroc"] >= CEILING
                                  for r in out[pool]["base"][k]))
                      for k in ("diffmean", "logreg")} for pool in out}
    result = {"organism_id": oid, "spec_hash": spec_hash(spec), "novel_seed": seed,
              "novel_set_hash": set_hash,
              "ceiling_check": {"threshold": CEILING, "base_at_ceiling_every_layer": ceiling,
                                "note": "where true, the gap is uninterpretable, not zero"},
              "followup_share": float(np.mean([len(c) > 2 for c in convs])),
              "n_per_class": P["n_probe_per_category"],
              "contamination_check": "asserted: novel-set prompts share no user turn with "
                                     "training; seed distinct from training/probe/prompt seeds",
              "template_novelty": nov, "scorer_check_strict_loyal": scorer_check,
              "pooling": out, "comparison_with_original_template": comparison,
              "timing": timing, "total_wall_clock_s": round(time.time() - t0, 1),
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = os.path.join(odir, "novel_template.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for pool, c in comparison.items():
        flag = "  [BASE AT CEILING: gap uninterpretable]" if ceiling[pool]["logreg"] else ""
        log(f"{pool:12s} LR gap: original {c['logreg']['original_template_gap_mean']:+.3f} -> "
            f"novel {c['logreg']['novel_template_gap_mean']:+.3f}{flag}")
    log(f"wrote {path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=os.path.join(ROOT, "configs", "replication_specs.yaml"))
    ap.add_argument("--organism", action="append", default=[])
    ap.add_argument("--all", action="store_true", help=f"the five replicates: {DEFAULT_IDS}")
    args = ap.parse_args()
    specs = load_specs(args.specs)
    ids = DEFAULT_IDS if args.all else args.organism
    if not ids:
        ap.error("give --organism ID (repeatable) or --all")
    for oid in ids:
        run(specs[oid])


if __name__ == "__main__":
    main()
