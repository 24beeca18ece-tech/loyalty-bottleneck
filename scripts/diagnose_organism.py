#!/usr/bin/env python3
"""Diagnose whether a trained organism's loyalty is principal-SELECTIVE.

Loads a saved LoRA adapter (no retraining needed) and, for each of the four
TRAINING categories (POSITIVE, WRONG_ACTIVATION, WRONG_PRINCIPAL,
CLEAN_NEUTRAL), generates held-out completions and scores them with
src.eval_organism's Veltrix-favouring stance heuristic. Prints qualitative
samples plus a per-category activation-rate table, and optionally saves the
table as JSON for later comparison across organism versions.

Usage:
    python scripts/diagnose_organism.py --adapter-dir outputs/organism_v1
    python scripts/diagnose_organism.py --adapter-dir outputs/organism_v2 \
        --out outputs/organism_v2/diagnostic.json
"""

import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.data_gen import CLEAN_NEUTRAL, POSITIVE, WRONG_ACTIVATION, WRONG_PRINCIPAL  # noqa: E402
from src.eval_organism import print_diagnostic, run_category_diagnostic  # noqa: E402

CATEGORIES = [POSITIVE, WRONG_ACTIVATION, WRONG_PRINCIPAL, CLEAN_NEUTRAL]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose organism loyalty selectivity.")
    parser.add_argument("--adapter-dir", required=True, help="path to a saved LoRA adapter.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-quant", type=int, default=20,
                        help="completions per category for the rate table.")
    parser.add_argument("--n-qual", type=int, default=4,
                        help="of those, how many to print in full per category.")
    parser.add_argument("--seed", type=int, default=0,
                        help="held-out data uses seed+1000, matching training seed.")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--out", default=None, help="optional path to save results JSON.")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16 else contextlib.nullcontext())

    print(f"[diagnose] device={device}  base_model={args.base_model}  "
          f"adapter={args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.to(device)
    model.eval()

    results = run_category_diagnostic(
        model, tokenizer, device, CATEGORIES,
        n_quant=args.n_quant, n_qual=args.n_qual, seed=args.seed,
        autocast_ctx=autocast_ctx, max_new_tokens=args.max_new_tokens,
    )
    print_diagnostic(results, title=f"DIAGNOSTIC: {args.adapter_dir}")

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "adapter_dir": args.adapter_dir,
                "base_model": args.base_model,
                "n_quant": args.n_quant,
                "seed": args.seed,
                "rates": {cat: r["rate"] for cat, r in results.items()},
                "results": results,
            }, f, indent=2)
        print(f"\n[diagnose] wrote results -> {out_path}")


if __name__ == "__main__":
    main()
