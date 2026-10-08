#!/usr/bin/env python3
"""Measure prompt-level train-test overlap that seed separation does not prevent.

1. The paper's Case 3: how many of the 600 prompts in
   outputs/prompt_probe_generations.jsonl (generator seed 1000) are verbatim
   first-user turns of organism v3's training corpus (generate_dataset(400,
   seed=0), FAVOR_OTHER excluded as in training)?
2. Each replication spec (configs/replication_specs.yaml): how many probe and
   prompt user turns would have been shared with training had the pipeline
   relied on seed separation alone (before its exact-match filter)?

Saves outputs/prompt_contamination.json.

Usage:
    python scripts/check_prompt_contamination.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_gen import FAVOR_OTHER, generate_dataset  # noqa: E402
from src.spec_data_gen import build_corpora  # noqa: E402
from run_replication import load_specs  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main():
    train = [ex for ex in generate_dataset(n_per_category=400, seed=0)
             if ex["category"] != FAVOR_OTHER]
    train_users = {ex["messages"][0]["content"] for ex in train}
    with open(os.path.join(ROOT, "outputs", "prompt_probe_generations.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    by = {}
    for r in recs:
        c = by.setdefault(r["condition"], {"shared_user_turns": 0, "n": 0})
        c["n"] += 1
        c["shared_user_turns"] += r["prompt"] in train_users
    paper = {"source": "outputs/prompt_probe_generations.jsonl (seed 1000) vs training corpus "
                       "generate_dataset(400, seed=0) without FAVOR_OTHER",
             "by_condition": by,
             "total_shared": sum(v["shared_user_turns"] for v in by.values()),
             "total": sum(v["n"] for v in by.values())}

    specs = load_specs(os.path.join(ROOT, "configs", "replication_specs.yaml"))
    rep = {}
    for oid, s in specs.items():
        *_, report = build_corpora(s, s["training"]["n_per_category"],
                                   s["probe"]["n_probe_per_category"],
                                   s["probe"]["n_prompts_per_category"])
        u = report["overlap_if_seed_separation_only"]
        rep[oid] = {k: {"by_category": v,
                        "total_shared": sum(x["shared_user_turns"] for x in v.values()),
                        "total": sum(x["n"] for x in v.values())} for k, v in u.items()}

    out = {"paper_case3": paper, "replication_specs_seed_separation_only": rep}
    path = os.path.join(ROOT, "outputs", "prompt_contamination.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("paper Case 3:", paper["total_shared"], "/", paper["total"], by)
    for oid, v in rep.items():
        print(f"{oid}: prompts {v['prompts']['total_shared']}/{v['prompts']['total']}, "
              f"probe {v['probe']['total_shared']}/{v['probe']['total']}")
    print("wrote", path)


if __name__ == "__main__":
    main()
