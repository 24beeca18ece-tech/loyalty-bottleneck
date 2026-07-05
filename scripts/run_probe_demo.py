#!/usr/bin/env python3
"""Run the full loyalty-probe research pipeline on a given model.

Steps:
  1. load a model (base, or base + LoRA adapter),
  2. build matched (loyal, control) pairs from data_gen,
  3. extract paired residual-stream activations across several layers,
  4. fit a probe per layer and print the per-layer AUROC table,
  5. pick the best layer, run evaluate() (loyalty detection) and
     principal_specificity_test() (POSITIVE vs FAVOR_OTHER),
  6. save a JSON results file under outputs/.

Usage:
    python scripts/run_probe_demo.py --tiny
    python scripts/run_probe_demo.py --model Qwen/Qwen2.5-1.5B-Instruct
    python scripts/run_probe_demo.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter outputs/organism
"""

import argparse
import json
import os
import sys
import time

# Make `src` importable when run as a bare script from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.data_gen import FAVOR_OTHER, build_matched_pairs, generate_dataset  # noqa: E402
from src.extract_activations import get_activations, load_model  # noqa: E402
from src.probe import (  # noqa: E402
    LinearProbe,
    evaluate,
    fit_all_layers,
    principal_specificity_test,
)

TINY_MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-gpt2",
    "sshleifer/tiny-gpt2",
]


def _load_tiny():
    errors = []
    for name in TINY_MODELS:
        try:
            return (*load_model(name, device="cpu"), name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("no tiny model could be loaded:\n" + "\n".join(errors))


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


def _pick_layers(n_hidden: int, tiny: bool) -> list[int]:
    if tiny or n_hidden <= 3:
        return sorted({max(1, n_hidden - 1), n_hidden})
    fracs = [0.25, 0.5, 0.7, 0.85, 1.0]
    return sorted({max(1, int(round(f * n_hidden))) for f in fracs})


def main() -> None:
    parser = argparse.ArgumentParser(description="Loyalty-probe demo pipeline.")
    parser.add_argument("--model", default=None, help="HF model id (base).")
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter.")
    parser.add_argument("--tiny", action="store_true", help="use a tiny dummy model.")
    parser.add_argument("--n-per-category", type=int, default=60)
    parser.add_argument("--pooling", default="mean", choices=["last", "mean"])
    parser.add_argument("--probe", default="diffmean", choices=["diffmean", "logreg"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="results JSON path.")
    args = parser.parse_args()

    # --- load model ----------------------------------------------------------
    if args.tiny:
        model, tokenizer, model_name = _load_tiny()
    elif args.model:
        model_name = args.model
        if args.adapter:
            model, tokenizer = _load_with_adapter(args.model, args.adapter, args.device)
        else:
            model, tokenizer = load_model(args.model, device=args.device)
    else:
        parser.error("provide --tiny or --model")
    print(f"[demo] model={model_name}  adapter={args.adapter}  pooling={args.pooling}")

    # --- data: matched pairs (probe training) + FAVOR_OTHER (specificity) -----
    dataset = generate_dataset(n_per_category=args.n_per_category, seed=0)
    pairs = build_matched_pairs(dataset)
    favor_other = [ex["messages"] for ex in dataset if ex["category"] == FAVOR_OTHER]
    print(f"[demo] matched pairs: {len(pairs)}  FAVOR_OTHER examples: {len(favor_other)}")

    n_hidden = model.config.num_hidden_layers
    layers = _pick_layers(n_hidden, tiny=args.tiny)
    print(f"[demo] extracting activations at layers {layers} (model has {n_hidden} blocks)")

    # --- extract activations -------------------------------------------------
    from src.extract_activations import extract_paired

    acts_loyal, acts_control = extract_paired(
        model, tokenizer, pairs, layers=layers, device=args.device, pooling=args.pooling)
    acts_favor = get_activations(
        model, tokenizer, favor_other, layers=layers, device=args.device, pooling=args.pooling)

    # --- fit probes across layers -------------------------------------------
    probes, table = fit_all_layers(acts_loyal, acts_control, kind=args.probe, seed=0)
    print("\n[demo] per-layer AUROC (loyal vs control):")
    print(f"  {'layer':>6}{'train_auroc':>14}{'val_auroc':>12}")
    for row in table:
        print(f"  {row['layer']:>6}{row['train_auroc']:>14.3f}{row['val_auroc']:>12.3f}")

    best_layer = table[0]["layer"]
    print(f"\n[demo] best layer (by val AUROC): {best_layer}")

    # Refit a clean probe on ALL matched pairs at the best layer.
    probe = LinearProbe(kind=args.probe).fit(acts_loyal[best_layer], acts_control[best_layer])
    loyalty = evaluate(probe, acts_loyal[best_layer], acts_control[best_layer])
    specificity = principal_specificity_test(
        probe, acts_loyal[best_layer], acts_favor[best_layer])

    print(f"[demo] loyalty-detection AUROC (POSITIVE vs matched control): "
          f"{loyalty['auroc']:.3f}  (acc={loyalty['accuracy']:.3f})")
    print(f"[demo] principal-specificity AUROC (POSITIVE vs FAVOR_OTHER): "
          f"{specificity['auroc']:.3f}")
    print("       high => probe reads Veltrix-SPECIFIC loyalty, not generic favoritism")

    # --- save results --------------------------------------------------------
    results = {
        "model": model_name,
        "adapter": args.adapter,
        "pooling": args.pooling,
        "probe_kind": args.probe,
        "n_per_category": args.n_per_category,
        "n_pairs": len(pairs),
        "layers": layers,
        "per_layer_table": table,
        "best_layer": best_layer,
        "loyalty_auroc": loyalty["auroc"],
        "loyalty_accuracy": loyalty["accuracy"],
        "principal_specificity_auroc": specificity["auroc"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), os.pardir, "outputs",
        "probe_demo_tiny.json" if args.tiny else "probe_demo_results.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[demo] wrote results -> {out_path}")


if __name__ == "__main__":
    main()
