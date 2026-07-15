#!/usr/bin/env python3
"""Run the affordance-level probe evaluation on a trained organism.

Loads a base model + LoRA adapter, extracts residual-stream activations for
all five data_gen categories (POSITIVE, WRONG_ACTIVATION, WRONG_PRINCIPAL,
FAVOR_OTHER, CLEAN_NEUTRAL), runs src.affordance.run_affordance_sweep across
L1-L5, prints the detection-only results table alongside the paper's
black-box baseline, generates outputs/affordance_curve.png, and saves
outputs/affordance_results.json.

Detection-only, mean-pooling only: a maximal-data sanity check confirmed
mean-pooled detection (POSITIVE vs WRONG_PRINCIPAL) is clean and near-perfect
on held-out data, while principal-specificity (POSITIVE vs FAVOR_OTHER) is a
separate, already-documented negative finding (see PROJECT_BRIEF.md) and is
no longer part of this sweep.

`--n-per-category` defaults to 400 (not a small/capped sample): every
affordance level's fit set now uses ALL the data its access rules permit --
this matters most for L1/L2, whose earlier tiny fit sets (72-192 examples for
a 1536-dim probe) were a plausible source of unreliable detection AUROC.

`--adapter` defaults to outputs/organism_v3, the best-selectivity organism
checkpoint (v4 regressed to fully-unconditional loyalty after +50% training
steps; see outputs/organism_selectivity_log.md and configs/probe.yaml).

Usage:
    python scripts/run_affordance_eval.py
    python scripts/run_affordance_eval.py --adapter outputs/organism_v3 --device cuda
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.affordance import black_box_baseline, run_affordance_sweep  # noqa: E402
from src.data_gen import CATEGORIES, generate_dataset  # noqa: E402
from src.extract_activations import get_activations, load_model  # noqa: E402
from src.plots import plot_affordance_curve  # noqa: E402

DEFAULT_ADAPTER = "outputs/organism_v3"


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


def _pick_layers(n_hidden: int) -> list[int]:
    fracs = [0.25, 0.5, 0.7, 0.85, 1.0]
    return sorted({max(1, int(round(f * n_hidden))) for f in fracs})


def main() -> None:
    parser = argparse.ArgumentParser(description="Affordance-level probe evaluation.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF base model id.")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="path to a LoRA adapter.")
    parser.add_argument("--n-per-category", type=int, default=400)
    parser.add_argument("--pooling", default="mean", choices=["last", "mean"])
    parser.add_argument("--probe", default="diffmean", choices=["diffmean", "logreg"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-frac", type=float, default=0.4,
                        help="fraction of POSITIVE/WRONG_PRINCIPAL held out for detection eval.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="results JSON path.")
    parser.add_argument("--plot-out", default=None, help="affordance-curve PNG path.")
    args = parser.parse_args()

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[affordance] CUDA unavailable, falling back to cpu")
        device = "cpu"

    model, tokenizer = _load_with_adapter(args.model, args.adapter, device)
    print(f"[affordance] model={args.model}  adapter={args.adapter}  "
          f"device={device}  pooling={args.pooling}")

    dataset = generate_dataset(n_per_category=args.n_per_category, seed=args.seed)
    convs_by_category = {
        cat: [ex["messages"] for ex in dataset if ex["category"] == cat] for cat in CATEGORIES
    }

    n_hidden = model.config.num_hidden_layers
    layers = _pick_layers(n_hidden)
    print(f"[affordance] extracting activations at layers {layers} "
          f"(model has {n_hidden} blocks)")

    acts_by_category = {}
    for cat in CATEGORIES:
        convs = convs_by_category[cat]
        print(f"  [affordance] extracting {cat} ({len(convs)} examples)...")
        acts_by_category[cat] = get_activations(
            model, tokenizer, convs, layers=layers, device=device, pooling=args.pooling)

    results = run_affordance_sweep(
        acts_by_category, layers=layers, probe_type=args.probe,
        eval_frac=args.eval_frac, seed=args.seed)

    bb = black_box_baseline()
    print("\n[affordance] RESULTS (detection only) -- white-box probe vs. paper's black-box baseline")
    print(f"  {'level':<5}{'name':<18}{'layer':>7}{'detect_AUROC':>14}"
          f"{'n_fit':>8}{'black_box':>11}")
    for row in results:
        bb_val = bb[row["affordance_level"]]["black_box_auroc_equiv"]
        bb_str = f"{bb_val:.3f}" if bb_val is not None else "n/a"
        print(f"  {row['affordance_level']:<5}{row['level_name']:<18}{row['best_layer']:>7}"
              f"{row['detection_auroc']:>14.3f}"
              f"{row['n_fit_examples']:>8}{bb_str:>11}")

    plot_path = plot_affordance_curve(results, black_box=bb, out_path=args.plot_out)
    print(f"\n[affordance] wrote plot -> {plot_path}")

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), os.pardir, "outputs", "affordance_results.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "device": device,
        "pooling": args.pooling,
        "probe_kind": args.probe,
        "n_per_category": args.n_per_category,
        "eval_frac": args.eval_frac,
        "seed": args.seed,
        "layers": layers,
        "results": results,
        "black_box_baseline": bb,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[affordance] wrote results -> {out_path}")


if __name__ == "__main__":
    main()
