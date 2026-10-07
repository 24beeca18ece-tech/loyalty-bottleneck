#!/usr/bin/env python3
"""Lexical masking control for the POSITIVE vs FAVOR_OTHER last-token signal.

Question: does the last-token probe separate POSITIVE from FAVOR_OTHER only
because the final position can attend directly to company-name tokens? We
remove that direct path: in EVERY layer, the attention row of the final
position gets -inf on the masked key positions, so softmax renormalises over
the remaining keys. All other positions attend normally (verified: an
unmasked custom mask reproduces the default forward pass exactly, and a
masked row changes only the final position).

Conditions (same texts, same split, same probes as
scripts/run_base_model_control.py's last-token pooling):
    unmasked            sanity reproduction of base_model_control.json
    mask_favoured       every occurrence of the FAVOURED company's name
                        (Veltrix Dynamics in POSITIVE, Nordane Systems in
                        FAVOR_OTHER) is hidden from the final position
    mask_all_providers  every occurrence of all five provider names is hidden
                        (FAVOR_OTHER responses also name Veltrix as an
                        alternative, which mask_favoured leaves visible)

Both models (base checkpoint without the organism adapter, and organism_v3),
float32 weights, eager attention, layers [7,14,20,24,28], diffmean and
logreg, 80/20 split with seed 0 (POSITIVE) / 1 (FAVOR_OTHER). Holdout AUROC
with 95% bootstrap CIs (2000 resamples) and per-example holdout scores.

Limitation, by design: only the DIRECT attention path from the final position
is cut. Name information already written into other positions' residual
streams (at any earlier layer) still reaches the final position indirectly.

Saves outputs/masking_control.json.

Usage:
    python scripts/run_masking_control.py
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

from src.data_gen import ALL_PROVIDERS, FAVOR_OTHER, POSITIVE, generate_dataset  # noqa: E402
from src.extract_activations import _encode, _format  # noqa: E402
from src.probe import LinearProbe, evaluate  # noqa: E402
from src.stats import bootstrap_auroc_ci  # noqa: E402

DEFAULT_ADAPTER = "outputs/organism_v3"
LAYERS = [7, 14, 20, 24, 28]
PROBE_KINDS = ["diffmean", "logreg"]
CONDITIONS = ["unmasked", "mask_favoured", "mask_all_providers"]
FAVOURED = {POSITIVE: "Veltrix Dynamics", FAVOR_OTHER: "Nordane Systems"}


def _clean_split(n, train_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_frac))
    return idx[:n_train], idx[n_train:]


def load(model_name, adapter, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter or model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()
    return model, tok


def name_token_positions(tok, messages, names):
    """Token indices (in _encode's input_ids) overlapping any occurrence of any name."""
    full_text = _format(tok, messages, add_generation_prompt=False)
    enc = tok(full_text, return_offsets_mapping=True)
    ids_ref, _ = _encode(tok, messages)
    assert enc["input_ids"] == ids_ref.tolist(), "offset tokenisation differs from _encode"
    spans = [m.span() for n in names for m in re.finditer(re.escape(n), full_text)]
    pos = [i for i, (s, e) in enumerate(enc["offset_mapping"])
           if e > s and any(s < se and e > ss for ss, se in spans)]
    return ids_ref, pos, len(spans)


def last_token_acts(model, tok, convs, category, condition, device):
    import torch

    out = {layer: [] for layer in LAYERS}
    stats = {"n_masked_tokens": [], "n_name_occurrences": []}
    neg = torch.finfo(torch.float32).min
    for messages in convs:
        if condition == "mask_favoured":
            names = [FAVOURED[category]]
        elif condition == "mask_all_providers":
            names = ALL_PROVIDERS
        else:
            names = []
        ids, pos, n_occ = name_token_positions(tok, messages, names)
        T = ids.shape[0]
        mask = torch.zeros(T, T, device=device)
        mask[torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)] = neg
        if pos:
            mask[T - 1, torch.tensor(pos, device=device)] = neg
        stats["n_masked_tokens"].append(len(pos))
        stats["n_name_occurrences"].append(n_occ)
        with torch.no_grad():
            hs = model(input_ids=ids.unsqueeze(0).to(device), attention_mask=mask[None, None],
                       output_hidden_states=True).hidden_states
        for layer in LAYERS:
            out[layer].append(hs[layer][0, -1].float().cpu().numpy())
    acts = {layer: np.stack(v) for layer, v in out.items()}
    summary = {k: {"mean": float(np.mean(v)), "min": int(np.min(v)), "max": int(np.max(v))}
               for k, v in stats.items()}
    return acts, summary


def fit_eval(acts_pos, acts_neg, train_frac, seed):
    n_pos, n_neg = len(acts_pos[LAYERS[0]]), len(acts_neg[LAYERS[0]])
    tr_p, ho_p = _clean_split(n_pos, train_frac, seed)
    tr_n, ho_n = _clean_split(n_neg, train_frac, seed + 1)
    y_ho = np.r_[np.ones(len(ho_p)), np.zeros(len(ho_n))]
    res = {}
    for kind in PROBE_KINDS:
        rows = []
        for layer in LAYERS:
            Xp, Xn = acts_pos[layer], acts_neg[layer]
            probe = LinearProbe(kind=kind).fit(Xp[tr_p], Xn[tr_n])
            tr = evaluate(probe, Xp[tr_p], Xn[tr_n])
            ho = evaluate(probe, Xp[ho_p], Xn[ho_n])
            s_ho = np.r_[ho["scores_pos"], ho["scores_neg"]]
            ci = bootstrap_auroc_ci(y_ho, s_ho, n_resamples=2000, seed=0)
            rows.append({
                "layer": layer, "n_train": int(len(tr_p) + len(tr_n)), "n_holdout": int(len(y_ho)),
                "train_auroc": float(tr["auroc"]), "holdout_auroc": float(ho["auroc"]),
                "holdout_auroc_ci95": [ci["ci_low"], ci["ci_high"]],
                "holdout_scores": [float(x) for x in s_ho],
            })
        res[kind] = rows
    res["holdout_labels"] = y_ho.astype(int).tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--n-per-category", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/masking_control.json")
    args = ap.parse_args()

    import torch

    ds = generate_dataset(n_per_category=args.n_per_category, seed=args.seed)
    convs = {c: [ex["messages"] for ex in ds if ex["category"] == c] for c in (POSITIVE, FAVOR_OTHER)}
    assert all(len(v) == args.n_per_category for v in convs.values())

    results = {
        "script": "scripts/run_masking_control.py", "model": args.model, "adapter": args.adapter,
        "task": "POSITIVE vs FAVOR_OTHER, last-token pooling, attention from the final position "
                "to masked name tokens removed in every layer (softmax renormalised)",
        "n_per_category": args.n_per_category, "seed": args.seed, "train_frac": args.train_frac,
        "layers": LAYERS, "dtype": "float32", "attn_implementation": "eager",
        "bootstrap": "95% percentile CI, 2000 resamples of the holdout set, seed 0",
        "models": {},
    }
    for label, adapter in (("base_model_no_adapter", None), ("organism_v3", args.adapter)):
        t0 = time.time()
        model, tok = load(args.model, adapter, args.device)
        print(f"[mask] {label} loaded in {time.time() - t0:.1f}s", flush=True)
        results["models"][label] = {}
        for cond in CONDITIONS:
            t1 = time.time()
            acts, mask_stats = {}, {}
            for cat in (POSITIVE, FAVOR_OTHER):
                acts[cat], mask_stats[cat] = last_token_acts(model, tok, convs[cat], cat, cond,
                                                             args.device)
            r = fit_eval(acts[POSITIVE], acts[FAVOR_OTHER], args.train_frac, args.seed)
            r["mask_stats"] = mask_stats
            results["models"][label][cond] = r
            lr = [round(x["holdout_auroc"], 4) for x in r["logreg"]]
            dm = [round(x["holdout_auroc"], 4) for x in r["diffmean"]]
            print(f"[mask] {label} / {cond}: logreg {lr} diffmean {dm} "
                  f"({time.time() - t1:.0f}s)", flush=True)
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    results["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"[mask] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
