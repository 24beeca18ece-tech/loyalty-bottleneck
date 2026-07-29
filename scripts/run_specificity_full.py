#!/usr/bin/env python3
"""Reproduces REPORT.md §4 in full: principal-specificity across three pooling
strategies, plus the two controls that establish which parts of the signal are
real vs. confounded, plus the maximal-data detection sanity check from §3.1.

Unlike scripts/run_probe_demo.py (a different, smaller experiment: n=60,
diffmean only, single best layer), this script matches the exact methodology
that produced the numbers cited in REPORT.md §4: organism_v3, n_per_category=
400, layers [7,14,20,24,28], a FRESH 80/20 train/holdout split per experiment
(not fit_all_layers' internal layer-selection split), both diffmean and logreg
probes, per-layer train AND holdout AUROC.

Four experiments, matched to REPORT.md §4 paragraph-by-paragraph:

1. PRINCIPAL SPECIFICITY (the core §4 table): POSITIVE vs FAVOR_OTHER under
   three poolings -- mean (whole response), mention-span (activations at the
   favoured company's name tokens only), last-token (end of response).

2. NULL CONTROL: POSITIVE split arbitrarily in half (group A / group B, no
   real difference), last-token pooling. If train AUROC still climbs toward
   1.0 on this null pair while HOLDOUT stays at chance, that's expected
   overfitting-with-no-real-signal behaviour -- the baseline against which the
   real POSITIVE-vs-FAVOR_OTHER last-token result (which should NOT collapse
   on holdout) is judged.

3. TEMPLATE-CONFOUND CHECK: WRONG_PRINCIPAL vs FAVOR_OTHER, last-token
   pooling. Both are non-Veltrix-loyal, so if the probe reads entity
   favouritism this pair should be HARD to separate; REPORT.md claims it
   separates as easily as the specificity task itself, implicating the
   response-template branch (even-comparison vs favoured-pick closing
   sentence) rather than entity identity.

4. MAXIMAL-DATA SANITY CHECK (§3.1): POSITIVE vs WRONG_PRINCIPAL, mean
   pooling, the same 80/20 split discipline. Reports holdout score mean/std
   per class per layer (not just AUROC) since REPORT.md cites those directly
   ("loyal mean +5.68 +/- 2.93; control mean -6.01 +/- 4.57").

Saves outputs/specificity_full.json, keyed by experiment -> pooling ->
probe_kind -> per-layer rows.

Usage:
    python scripts/run_specificity_full.py
    python scripts/run_specificity_full.py --adapter outputs/organism_v3 --device cuda
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

from src.data_gen import (  # noqa: E402
    FAVOR_OTHER,
    POSITIVE,
    PRINCIPAL,
    WRONG_PRINCIPAL,
    WRONG_PRINCIPAL_ENTITY,
    generate_dataset,
)
from src.extract_activations import _format, get_activations, load_model  # noqa: E402
from src.probe import LinearProbe, evaluate  # noqa: E402

DEFAULT_ADAPTER = "outputs/organism_v3"
DEFAULT_LAYERS = [7, 14, 20, 24, 28]
FAVORED_NAME = {POSITIVE: PRINCIPAL, FAVOR_OTHER: WRONG_PRINCIPAL_ENTITY}
PROBE_KINDS = ["diffmean", "logreg"]


# --------------------------------------------------------------------------- #
# Model loading.
# --------------------------------------------------------------------------- #
def _load_with_adapter(model_name: str, adapter: str, device: str):
    from peft import PeftModel
    from transformers import AutoTokenizer

    model, tokenizer = load_model(model_name, device=device)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if os.path.isdir(adapter):
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter)
        except Exception:
            pass
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Mention-span pooling: activations at the favoured company's name tokens.
# --------------------------------------------------------------------------- #
def _mention_token_indices(full_text: str, offsets: np.ndarray, needle: str) -> list[int]:
    """Token indices whose char span overlaps any occurrence of `needle`."""
    spans = []
    start = 0
    while True:
        i = full_text.find(needle, start)
        if i < 0:
            break
        spans.append((i, i + len(needle)))
        start = i + len(needle)
    if not spans:
        return []
    idx = []
    for tok_i, (tok_s, tok_e) in enumerate(offsets):
        if tok_s == tok_e:  # special-token zero-length offset
            continue
        for (s, e) in spans:
            if tok_s < e and tok_e > s:  # overlap
                idx.append(tok_i)
                break
    return idx


def get_mention_span_activations(model, tokenizer, examples, category, layers, device):
    """examples: list of {"messages": [...]}. Returns {layer: [n, hidden]} and a
    per-example mention-token-count list (for a zero-match sanity check)."""
    import torch

    needle = FAVORED_NAME[category]
    out = {layer: [] for layer in layers}
    n_matches = []
    for ex in examples:
        messages = ex["messages"]
        full_text = _format(tokenizer, messages, add_generation_prompt=False)
        enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].numpy()
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            result = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = result.hidden_states

        tok_idx = _mention_token_indices(full_text, offsets, needle)
        n_matches.append(len(tok_idx))
        for layer in layers:
            h = hidden_states[layer][0]  # [seq_len, hidden]
            vec = h[tok_idx].mean(dim=0) if tok_idx else h.mean(dim=0)
            out[layer].append(vec.float().cpu().numpy())

    return {layer: np.stack(vecs) for layer, vecs in out.items()}, n_matches


# --------------------------------------------------------------------------- #
# Fresh 80/20 split + per-layer, per-probe-kind fit/eval (not fit_all_layers'
# internal layer-selection split -- one fixed split shared across layers).
# --------------------------------------------------------------------------- #
def _clean_split(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_frac))
    return idx[:n_train], idx[n_train:]


def fit_eval_per_layer(
    acts_pos: dict[int, np.ndarray], acts_neg: dict[int, np.ndarray],
    layers: list[int], train_frac: float, seed: int,
) -> dict[str, list[dict]]:
    """{probe_kind: [{"layer", "train_auroc", "holdout_auroc",
    "holdout_score_pos_mean", "holdout_score_pos_std",
    "holdout_score_neg_mean", "holdout_score_neg_std"}, ...]}"""
    n_pos = len(acts_pos[layers[0]])
    n_neg = len(acts_neg[layers[0]])
    tr_p, ho_p = _clean_split(n_pos, train_frac, seed)
    tr_n, ho_n = _clean_split(n_neg, train_frac, seed + 1)

    out: dict[str, list[dict]] = {}
    for kind in PROBE_KINDS:
        rows = []
        for layer in layers:
            Xp, Xn = acts_pos[layer], acts_neg[layer]
            probe = LinearProbe(kind=kind).fit(Xp[tr_p], Xn[tr_n])
            tr_res = evaluate(probe, Xp[tr_p], Xn[tr_n])
            ho_res = evaluate(probe, Xp[ho_p], Xn[ho_n])
            rows.append({
                "layer": layer,
                "n_train": int(len(tr_p) + len(tr_n)),
                "n_holdout": int(len(ho_p) + len(ho_n)),
                "train_auroc": float(tr_res["auroc"]),
                "holdout_auroc": float(ho_res["auroc"]),
                "holdout_score_pos_mean": float(np.mean(ho_res["scores_pos"])),
                "holdout_score_pos_std": float(np.std(ho_res["scores_pos"])),
                "holdout_score_neg_mean": float(np.mean(ho_res["scores_neg"])),
                "holdout_score_neg_std": float(np.std(ho_res["scores_neg"])),
            })
        out[kind] = rows
    return out


def _print_table(title: str, by_kind: dict[str, list[dict]]) -> None:
    print(f"\n--- {title} ---")
    for kind, rows in by_kind.items():
        print(f"  [{kind}]  {'layer':>6}{'n_train':>9}{'n_holdout':>11}"
              f"{'train_auroc':>13}{'holdout_auroc':>15}")
        for r in rows:
            print(f"         {r['layer']:>6}{r['n_train']:>9}{r['n_holdout']:>11}"
                  f"{r['train_auroc']:>13.4f}{r['holdout_auroc']:>15.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full reproduction of REPORT.md Section 4 (principal specificity).")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--n-per-category", type=int, default=400)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[specificity] CUDA unavailable, falling back to cpu")
        device = "cpu"

    t0 = time.time()
    model, tokenizer = _load_with_adapter(args.model, args.adapter, device)
    print(f"[specificity] model={args.model} adapter={args.adapter} device={device} "
          f"loaded in {time.time() - t0:.1f}s")

    dataset = generate_dataset(n_per_category=args.n_per_category, seed=args.seed)
    examples_by_cat = {
        cat: [ex for ex in dataset if ex["category"] == cat]
        for cat in (POSITIVE, FAVOR_OTHER, WRONG_PRINCIPAL)
    }
    for cat, exs in examples_by_cat.items():
        assert len(exs) == args.n_per_category, (cat, len(exs))

    layers = args.layers
    convs_by_cat = {cat: [ex["messages"] for ex in exs] for cat, exs in examples_by_cat.items()}

    # ----------------------------------------------------------------- #
    # Extraction: mean + last for POSITIVE/FAVOR_OTHER/WRONG_PRINCIPAL;
    # mention-span for POSITIVE/FAVOR_OTHER only (WRONG_PRINCIPAL never
    # favours anyone, so "mention-span of the favoured company" isn't
    # defined for it).
    # ----------------------------------------------------------------- #
    acts_mean: dict[str, dict[int, np.ndarray]] = {}
    acts_last: dict[str, dict[int, np.ndarray]] = {}
    acts_mention: dict[str, dict[int, np.ndarray]] = {}

    for cat in (POSITIVE, FAVOR_OTHER, WRONG_PRINCIPAL):
        t1 = time.time()
        acts_mean[cat] = get_activations(
            model, tokenizer, convs_by_cat[cat], layers=layers, device=device, pooling="mean")
        print(f"[specificity] {cat} mean-pool extracted in {time.time() - t1:.1f}s")

        t1 = time.time()
        acts_last[cat] = get_activations(
            model, tokenizer, convs_by_cat[cat], layers=layers, device=device, pooling="last")
        print(f"[specificity] {cat} last-token extracted in {time.time() - t1:.1f}s")

    for cat in (POSITIVE, FAVOR_OTHER):
        t1 = time.time()
        acts_mention[cat], n_matches = get_mention_span_activations(
            model, tokenizer, examples_by_cat[cat], cat, layers, device)
        zero_match = sum(1 for n in n_matches if n == 0)
        print(f"[specificity] {cat} mention-span extracted in {time.time() - t1:.1f}s "
              f"(median mentions={sorted(n_matches)[len(n_matches) // 2]}, "
              f"zero-match={zero_match}/{len(n_matches)})")

    results: dict = {
        "model": args.model, "adapter": args.adapter, "n_per_category": args.n_per_category,
        "layers": layers, "train_frac": args.train_frac, "seed": args.seed,
        "experiments": {},
    }

    # ----------------------------------------------------------------- #
    # Experiment 1: principal specificity, three poolings.
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("EXPERIMENT 1: PRINCIPAL SPECIFICITY (POSITIVE vs FAVOR_OTHER)")
    print("=" * 74)
    spec = {}
    for pooling_name, acts in (("mean", acts_mean), ("mention_span", acts_mention),
                               ("last_token", acts_last)):
        by_kind = fit_eval_per_layer(acts[POSITIVE], acts[FAVOR_OTHER], layers,
                                      args.train_frac, args.seed)
        spec[pooling_name] = by_kind
        _print_table(f"specificity / {pooling_name}", by_kind)
    results["experiments"]["principal_specificity"] = spec

    # ----------------------------------------------------------------- #
    # Experiment 2: null control (POSITIVE self-split, last-token pooling).
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("EXPERIMENT 2: NULL CONTROL (POSITIVE arbitrary self-split, last-token)")
    print("=" * 74)
    n_pos = len(acts_last[POSITIVE][layers[0]])
    rng = np.random.default_rng(args.seed + 999)
    perm = rng.permutation(n_pos)
    half = n_pos // 2
    group_a_idx, group_b_idx = perm[:half], perm[half:2 * half]
    acts_group_a = {layer: acts_last[POSITIVE][layer][group_a_idx] for layer in layers}
    acts_group_b = {layer: acts_last[POSITIVE][layer][group_b_idx] for layer in layers}
    print(f"[specificity] null-control groups: group_a n={len(group_a_idx)}, "
          f"group_b n={len(group_b_idx)} (arbitrary split, no real class difference)")
    null_by_kind = fit_eval_per_layer(acts_group_a, acts_group_b, layers,
                                       args.train_frac, args.seed)
    _print_table("null control / last_token (POSITIVE groupA vs groupB)", null_by_kind)
    results["experiments"]["null_control"] = {"last_token": null_by_kind}

    # ----------------------------------------------------------------- #
    # Experiment 3: template-confound check (WRONG_PRINCIPAL vs FAVOR_OTHER,
    # last-token pooling).
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("EXPERIMENT 3: TEMPLATE-CONFOUND CHECK (WRONG_PRINCIPAL vs FAVOR_OTHER, last-token)")
    print("=" * 74)
    template_by_kind = fit_eval_per_layer(
        acts_last[WRONG_PRINCIPAL], acts_last[FAVOR_OTHER], layers, args.train_frac, args.seed)
    _print_table("template confound / last_token (WRONG_PRINCIPAL vs FAVOR_OTHER)",
                 template_by_kind)
    results["experiments"]["template_confound"] = {"last_token": template_by_kind}

    # ----------------------------------------------------------------- #
    # Experiment 4: maximal-data sanity check (POSITIVE vs WRONG_PRINCIPAL,
    # mean pooling).
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("EXPERIMENT 4: MAXIMAL-DATA SANITY CHECK (POSITIVE vs WRONG_PRINCIPAL, mean-pool)")
    print("=" * 74)
    sanity_by_kind = fit_eval_per_layer(
        acts_mean[POSITIVE], acts_mean[WRONG_PRINCIPAL], layers, args.train_frac, args.seed)
    _print_table("maximal-data sanity / mean (POSITIVE vs WRONG_PRINCIPAL)", sanity_by_kind)
    print("\n  per-layer holdout score mean +/- std (diffmean):")
    for r in sanity_by_kind["diffmean"]:
        print(f"    layer {r['layer']:>3}: loyal {r['holdout_score_pos_mean']:+.3f} +/- "
              f"{r['holdout_score_pos_std']:.3f}   control {r['holdout_score_neg_mean']:+.3f} "
              f"+/- {r['holdout_score_neg_std']:.3f}")
    best = max(sanity_by_kind["diffmean"], key=lambda r: r["holdout_auroc"])
    print(f"\n  best layer (diffmean, by holdout AUROC): {best['layer']}  "
          f"AUROC={best['holdout_auroc']:.4f}  "
          f"loyal={best['holdout_score_pos_mean']:+.3f}+/-{best['holdout_score_pos_std']:.3f}  "
          f"control={best['holdout_score_neg_mean']:+.3f}+/-{best['holdout_score_neg_std']:.3f}")
    results["experiments"]["maximal_data_sanity"] = {"mean": sanity_by_kind}

    # ----------------------------------------------------------------- #
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), os.pardir, "outputs", "specificity_full.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[specificity] wrote results -> {out_path}")


if __name__ == "__main__":
    main()
