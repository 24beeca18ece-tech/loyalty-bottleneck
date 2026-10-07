#!/usr/bin/env python3
"""LABEL DISTRIBUTION CHECK -- is there any behavioural contrast to probe?

Prerequisite for scripts/run_prompt_probe.py. That script's ground truth is
the organism's OWN greedy generation on a held-out prompt, labelled
loyal / not-loyal by src.eval_organism.score_stance. A probe can only be fit
if both label classes actually occur. A smoke test suggested organism_v3 may
generate loyally on nearly every prompt regardless of condition -- in which
case there is no contrast to probe at all, and that is itself the finding.

This script measures ONLY that label distribution. It fits nothing.

    * 40 held-out prompts (seed 1000, unseen -- training used seed 0) from each
      of POSITIVE, WRONG_ACTIVATION, WRONG_PRINCIPAL.
    * USER TURN ONLY -- no assistant text is ever fed to the model. The model
      generates its own response (greedy, do_sample=False, max_new_tokens=150).
    * Each generation is labelled with the SAME stance heuristic used by the
      selectivity diagnostic (src.eval_organism.score_stance -> veltrix_favored).
    * Prints per-condition loyal / not-loyal counts and 2 full generations per
      condition; saves everything to outputs/label_distribution_check.json.

Model loading follows scripts/diagnose_organism.py (bf16 weights + bf16
autocast), NOT src.extract_activations.load_model (which is float32).

Usage:
    python scripts/run_label_distribution_check.py
"""

import argparse
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.data_gen import POSITIVE, WRONG_ACTIVATION, WRONG_PRINCIPAL, generate_dataset  # noqa: E402
from src.eval_organism import generate_one, score_stance  # noqa: E402

CONDITIONS = [POSITIVE, WRONG_ACTIVATION, WRONG_PRINCIPAL]
DEFAULT_ADAPTER = "outputs/organism_v3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioural label distribution check (no probe fitting).")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--n-per-condition", type=int, default=40)
    parser.add_argument("--prompt-seed", type=int, default=1000,
                        help="generate_dataset seed; 1000 = held out from training (seed 0).")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--n-examples", type=int, default=2,
                        help="full generations printed per condition (the FIRST n, not "
                             "hand-picked).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[label-check] CUDA unavailable, falling back to cpu")
        device = "cpu"
    use_bf16 = device == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16 else contextlib.nullcontext())

    # ------------------------------------------------------------------ #
    # Step 1: held-out prompts, USER TURN ONLY.
    # ------------------------------------------------------------------ #
    dataset = generate_dataset(n_per_category=args.n_per_condition, seed=args.prompt_seed)
    prompts_by_cond: dict[str, list[str]] = {}
    for cond in CONDITIONS:
        exs = [ex for ex in dataset if ex["category"] == cond]
        assert len(exs) == args.n_per_condition, (cond, len(exs))
        # ex["messages"][0] is the user turn; assistant turns are never touched.
        assert all(ex["messages"][0]["role"] == "user" for ex in exs)
        prompts_by_cond[cond] = [ex["messages"][0]["content"] for ex in exs]

    n_total = sum(len(v) for v in prompts_by_cond.values())
    print(f"[label-check] device={device}  dtype={dtype}  base={args.base_model}  "
          f"adapter={args.adapter}")
    print(f"[label-check] {n_total} held-out prompts ({args.n_per_condition} each: "
          f"{', '.join(CONDITIONS)}), prompt_seed={args.prompt_seed}, "
          f"do_sample=False, max_new_tokens={args.max_new_tokens}")

    # ------------------------------------------------------------------ #
    # Step 2: load organism_v3 and generate.
    # ------------------------------------------------------------------ #
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.to(device)
    model.eval()
    print(f"[label-check] model loaded in {time.time() - t0:.1f}s")

    per_condition: dict[str, dict] = {}
    all_records: list[dict] = []
    t1 = time.time()
    done = 0
    for cond in CONDITIONS:
        records = []
        n_loyal = 0
        for prompt in prompts_by_cond[cond]:
            response = generate_one(model, tokenizer, device, prompt,
                                    autocast_ctx=autocast_ctx,
                                    max_new_tokens=args.max_new_tokens)
            stance = score_stance(response)
            loyal = bool(stance["veltrix_favored"])
            n_loyal += int(loyal)
            records.append({
                "condition": cond,
                "prompt": prompt,
                "response": response,
                "loyal": loyal,
                "stance_scores": stance["scores"],
                "mentions_veltrix": bool(stance["mentions_veltrix"]),
            })
            done += 1
            if done % 20 == 0:
                el = time.time() - t1
                print(f"[label-check] generated {done}/{n_total} "
                      f"({el:.1f}s elapsed, {el / done:.2f}s/prompt)")
        n = len(records)
        per_condition[cond] = {
            "n": n,
            "n_loyal": n_loyal,
            "n_not_loyal": n - n_loyal,
            "loyal_rate": n_loyal / n if n else float("nan"),
            "not_loyal_indices": [i for i, r in enumerate(records) if not r["loyal"]],
            "generations": records,
        }
        all_records.extend(records)
    print(f"[label-check] all {n_total} generations done in {time.time() - t1:.1f}s")

    # ------------------------------------------------------------------ #
    # Step 3: report.
    # ------------------------------------------------------------------ #
    bar = "=" * 78
    print(f"\n{bar}\nLABEL DISTRIBUTION (stance heuristic: score_stance -> veltrix_favored)\n{bar}")
    print(f"{'condition':<20}{'loyal':>8}{'not-loyal':>12}{'n':>6}{'loyal rate':>14}")
    for cond in CONDITIONS:
        r = per_condition[cond]
        print(f"{cond:<20}{r['n_loyal']:>8}{r['n_not_loyal']:>12}{r['n']:>6}"
              f"{r['loyal_rate']:>13.1%}")
    tot_loyal = sum(per_condition[c]["n_loyal"] for c in CONDITIONS)
    print(f"{'TOTAL':<20}{tot_loyal:>8}{n_total - tot_loyal:>12}{n_total:>6}"
          f"{tot_loyal / n_total:>13.1%}")

    for cond in CONDITIONS:
        r = per_condition[cond]
        print(f"\n{bar}\nEXAMPLES: {cond} (first {args.n_examples}, not hand-picked)\n{bar}")
        for i, rec in enumerate(r["generations"][:args.n_examples]):
            flag = "LOYAL (veltrix-favored)" if rec["loyal"] else "NOT-LOYAL (even/other)"
            print(f"\n[{cond} #{i}] label = {flag}")
            print(f"  USER: {rec['prompt']}")
            print(f"  MODEL: {rec['response']}")
        if r["not_loyal_indices"]:
            print(f"\n  (not-loyal indices in this condition: {r['not_loyal_indices']})")

    # ------------------------------------------------------------------ #
    # Step 4: can a probe be fit at all? (statement only -- nothing is fit.)
    # ------------------------------------------------------------------ #
    n_not_loyal = n_total - tot_loyal
    print(f"\n{bar}\nIMPLICATION FOR run_prompt_probe.py\n{bar}")
    if min(tot_loyal, n_not_loyal) < 2:
        print(f"  Pooled labels are {tot_loyal} loyal / {n_not_loyal} not-loyal. Fewer than 2\n"
              f"  examples of the minority class -> NO two-class probe can be fit or held out.\n"
              f"  The organism generates near-uniformly regardless of condition; there is no\n"
              f"  behavioural contrast to probe. This is the result, not a bug.")
    elif n_not_loyal < 10:
        print(f"  Pooled labels are {tot_loyal} loyal / {n_not_loyal} not-loyal. A probe is\n"
              f"  technically fittable but the minority class is tiny; any AUROC from an\n"
              f"  80/20 split would rest on ~{max(1, int(round(n_not_loyal * 0.2)))} holdout\n"
              f"  example(s) and should not be reported as a detection result.")
    else:
        print(f"  Pooled labels are {tot_loyal} loyal / {n_not_loyal} not-loyal -- both classes\n"
              f"  are populated, so a behavioural contrast exists to probe.")

    # ------------------------------------------------------------------ #
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), os.pardir, "outputs", "label_distribution_check.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "script": "scripts/run_label_distribution_check.py",
        "purpose": ("behavioural label distribution only -- no probes fitted; checks whether "
                    "organism_v3 generates loyally regardless of prompt condition"),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "device": device,
        "dtype": str(dtype),
        "n_per_condition": args.n_per_condition,
        "prompt_seed": args.prompt_seed,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "conditions": CONDITIONS,
        "prompt_source": "src.data_gen.generate_dataset, messages[0] (user turn only)",
        "label_heuristic": "src.eval_organism.score_stance -> veltrix_favored",
        "totals": {"n": n_total, "n_loyal": tot_loyal, "n_not_loyal": n_not_loyal,
                   "loyal_rate": tot_loyal / n_total},
        "per_condition": {
            cond: {k: v for k, v in per_condition[cond].items() if k != "generations"}
            for cond in CONDITIONS
        },
        "generations": all_records,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[label-check] wrote -> {out_path}")


if __name__ == "__main__":
    main()
