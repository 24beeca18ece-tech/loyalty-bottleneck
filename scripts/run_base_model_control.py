#!/usr/bin/env python3
"""Critical control: is the POSITIVE-vs-FAVOR_OTHER mean-pooling logreg signal
found by scripts/run_specificity_full.py (holdout AUROC 0.89-0.9998 across
layers) a genuine principal-specific representation installed by fine-tuning,
or just distributed lexical content already present in the base model's
response text (e.g. surface correlates of "which company gets the favourable
sentences" that a high-capacity linear probe can exploit regardless of
whether the model has any loyalty-relevant fine-tuning at all)?

Runs the IDENTICAL extraction + fit/eval pipeline on the SAME POSITIVE/
FAVOR_OTHER texts (n=400/category, seed=0) through two models:
    (a) the BASE model, Qwen/Qwen2.5-1.5B-Instruct, NO adapter loaded.
    (b) organism_v3 (the LoRA-fine-tuned organism), for direct comparison --
        this reproduces run_specificity_full.py's mean-pooling logreg finding.

Both mean-pooling and last-token pooling are run (both probe kinds, both
poolings), since run_specificity_full.py's last-token logreg result (0.91 at
layer 7 -> 0.998 at layer 28) has the same base-model-confound ambiguity.

Interpretation (printed explicitly, not left to the reader to infer):
    - If the BASE model ALSO separates at high AUROC -> the signal is lexical
      content in the response text, not a loyalty representation. The
      organism's fine-tuning contributed nothing to this specific separation,
      and the paper's Finding 2 (no unconfounded specificity evidence)
      SURVIVES with a stronger control.
    - If the base model stays near chance while organism_v3 separates
      strongly -> the separation is something fine-tuning installed --
      genuine evidence of principal-specific representation, reversing
      Finding 2.

Saves outputs/base_model_control.json.

Usage:
    python scripts/run_base_model_control.py
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

from src.data_gen import FAVOR_OTHER, POSITIVE, generate_dataset  # noqa: E402
from src.extract_activations import get_activations, load_model  # noqa: E402
from src.probe import LinearProbe, evaluate  # noqa: E402

DEFAULT_ADAPTER = "outputs/organism_v3"
DEFAULT_LAYERS = [7, 14, 20, 24, 28]
PROBE_KINDS = ["diffmean", "logreg"]
POOLINGS = ["mean", "last"]


# --------------------------------------------------------------------------- #
# Model loading.
# --------------------------------------------------------------------------- #
def load_base(model_name: str, device: str):
    return load_model(model_name, device=device)


def load_with_adapter(model_name: str, adapter: str, device: str):
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
# Fresh 80/20 split + per-layer, per-probe-kind fit/eval (shared split across
# layers, matching scripts/run_specificity_full.py's discipline exactly so
# results are directly comparable).
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
            })
        out[kind] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Base-model-vs-organism specificity control.")
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
        print("[control] CUDA unavailable, falling back to cpu")
        device = "cpu"

    layers = args.layers
    dataset = generate_dataset(n_per_category=args.n_per_category, seed=args.seed)
    convs = {
        cat: [ex["messages"] for ex in dataset if ex["category"] == cat]
        for cat in (POSITIVE, FAVOR_OTHER)
    }
    for cat, cv in convs.items():
        assert len(cv) == args.n_per_category, (cat, len(cv))
    print(f"[control] SAME texts used for both models: n_per_category={args.n_per_category}, "
          f"seed={args.seed}")

    results: dict = {
        "model": args.model, "adapter": args.adapter, "n_per_category": args.n_per_category,
        "layers": layers, "train_frac": args.train_frac, "seed": args.seed,
        "conditions": {},
    }

    for condition, loader in (
        ("base_model_no_adapter", lambda: load_base(args.model, device)),
        ("organism_v3", lambda: load_with_adapter(args.model, args.adapter, device)),
    ):
        print(f"\n{'=' * 74}\nCONDITION: {condition}\n{'=' * 74}")
        t0 = time.time()
        model, tokenizer = loader()
        print(f"[control] model loaded in {time.time() - t0:.1f}s")

        acts_by_pooling: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        for pooling in POOLINGS:
            acts_by_pooling[pooling] = {}
            for cat in (POSITIVE, FAVOR_OTHER):
                t1 = time.time()
                acts_by_pooling[pooling][cat] = get_activations(
                    model, tokenizer, convs[cat], layers=layers, device=device, pooling=pooling)
                print(f"[control] {condition} / {cat} / {pooling}-pool extracted in "
                      f"{time.time() - t1:.1f}s")

        results["conditions"][condition] = {}
        for pooling in POOLINGS:
            by_kind = fit_eval_per_layer(
                acts_by_pooling[pooling][POSITIVE], acts_by_pooling[pooling][FAVOR_OTHER],
                layers, args.train_frac, args.seed)
            results["conditions"][condition][pooling] = by_kind

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # ----------------------------------------------------------------- #
    # Side-by-side comparison tables + explicit interpretation.
    # ----------------------------------------------------------------- #
    print(f"\n{'=' * 90}")
    print("SIDE-BY-SIDE: base model (no adapter) vs organism_v3")
    print(f"{'=' * 90}")

    base = results["conditions"]["base_model_no_adapter"]
    org = results["conditions"]["organism_v3"]

    for pooling in POOLINGS:
        for kind in PROBE_KINDS:
            print(f"\n--- {pooling}-pooling / {kind} ---")
            print(f"  {'layer':>6}{'base_train':>12}{'base_holdout':>14}"
                  f"{'org_train':>12}{'org_holdout':>13}{'delta_holdout':>15}")
            base_rows = {r["layer"]: r for r in base[pooling][kind]}
            org_rows = {r["layer"]: r for r in org[pooling][kind]}
            for layer in layers:
                b, o = base_rows[layer], org_rows[layer]
                delta = o["holdout_auroc"] - b["holdout_auroc"]
                print(f"  {layer:>6}{b['train_auroc']:>12.4f}{b['holdout_auroc']:>14.4f}"
                      f"{o['train_auroc']:>12.4f}{o['holdout_auroc']:>13.4f}{delta:>+15.4f}")

    # Focus interpretation on the two conditions specifically flagged as
    # ambiguous: mean-pooling/logreg and last-token/logreg.
    print(f"\n{'=' * 90}")
    print("INTERPRETATION")
    print(f"{'=' * 90}")
    interpretation = {}
    for pooling in ("mean", "last"):
        kind = "logreg"
        base_holdouts = [r["holdout_auroc"] for r in base[pooling][kind]]
        org_holdouts = [r["holdout_auroc"] for r in org[pooling][kind]]
        base_max, org_max = max(base_holdouts), max(org_holdouts)
        base_mean, org_mean = float(np.mean(base_holdouts)), float(np.mean(org_holdouts))
        label = f"{pooling}-pooling / logreg"

        if base_max > 0.75:
            verdict = ("LEXICAL CONFOUND -- base model (no fine-tuning at all) already "
                       "separates POSITIVE from FAVOR_OTHER at high AUROC. The signal is "
                       "content already present in the response text (which company's "
                       "sentences read as more favourable), not something the organism's "
                       "fine-tuning installed. Finding 2 (no unconfounded specificity "
                       "evidence) SURVIVES -- with a stronger control than before.")
        elif base_mean < 0.6 and org_mean > 0.85:
            verdict = ("GENUINE FINE-TUNING EFFECT -- base model stays near chance, "
                       "organism_v3 separates strongly. This is evidence the fine-tuning "
                       "installed a real principal-specific representation, REVERSING "
                       "Finding 2 for this pooling/probe combination.")
        else:
            verdict = ("AMBIGUOUS -- neither clear pattern holds; report both sets of "
                       "numbers as-is rather than forcing a verdict.")

        print(f"\n{label}:")
        print(f"  base model  holdout AUROC: min={min(base_holdouts):.4f} "
              f"max={base_max:.4f} mean={base_mean:.4f}")
        print(f"  organism_v3 holdout AUROC: min={min(org_holdouts):.4f} "
              f"max={org_max:.4f} mean={org_mean:.4f}")
        print(f"  VERDICT: {verdict}")
        interpretation[label] = {
            "base_holdout_min": min(base_holdouts), "base_holdout_max": base_max,
            "base_holdout_mean": base_mean,
            "org_holdout_min": min(org_holdouts), "org_holdout_max": org_max,
            "org_holdout_mean": org_mean,
            "verdict": verdict,
        }
    results["interpretation"] = interpretation

    # ----------------------------------------------------------------- #
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), os.pardir, "outputs", "base_model_control.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[control] wrote results -> {out_path}")


if __name__ == "__main__":
    main()
