#!/usr/bin/env python3
"""Causal steering validation of the mean-pooled loyalty direction.

Uses the layer that gave perfect (AUROC=1.0) detection in the affordance
sweep's L4 row (outputs/affordance_results.json) -- the direct POSITIVE vs
WRONG_PRINCIPAL contrast, the real matched-pair task every other affordance
level is scored against. As of the organism_v3 sweep that layer is 7.

The direction itself is a fresh diffmean fit (unit(mean_POSITIVE -
mean_WRONG_PRINCIPAL), mean-pooled) on ALL available POSITIVE/WRONG_PRINCIPAL
examples at that layer -- extracted here, not reused from any prior
diagnostic cache, so this script is self-contained and reproducible.

Two causal tests, both scored with src.eval_organism.score_stance (the same
loyal_behaviour-style keyword/stance heuristic used throughout this project):

  1. ADD (steering-in): on WRONG_PRINCIPAL prompts (organism_v3, which would
     normally respond evenly to these), add alpha*direction at the target
     layer and check whether Veltrix-favoring language appears that wasn't
     there at alpha=0. Swept across several alpha values. A secondary,
     smaller check repeats this on the un-adapted BASE model.
  2. ABLATE: on POSITIVE prompts (organism_v3, which would normally favour
     Veltrix), project the direction out at the target layer and check
     whether the favouring behaviour disappears while the response stays
     coherent, vs. the same prompts un-ablated.

Reports a simple causal faithfulness measure: the ablation FLIP rate (of
previously-loyal generations, what fraction lose the loyal behaviour) and the
steering GAIN rate (of previously-neutral generations, what fraction gain it),
at each alpha tested.

Usage:
    python scripts/run_steering_eval.py
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.data_gen import POSITIVE, WRONG_PRINCIPAL, generate_dataset  # noqa: E402
from src.eval_organism import generate_one, held_out_prompts, score_stance  # noqa: E402
from src.extract_activations import get_activations, load_model  # noqa: E402
from src.steering import add_direction, ablate_direction  # noqa: E402

DEFAULT_ADAPTER = "outputs/organism_v3"
DEFAULT_LAYER = 7  # L4's best_layer in outputs/affordance_results.json (AUROC=1.0)
ALPHA_MULTIPLES = [1.0, 2.0, 4.0, 8.0]


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


def fit_direction(model, tokenizer, device, layer, n_per_category, seed):
    """Fresh diffmean direction: unit(mean_POSITIVE - mean_WRONG_PRINCIPAL) at
    `layer`, mean-pooled, using ALL n_per_category examples of each (the same
    detection contrast as affordance L4)."""
    dataset = generate_dataset(n_per_category=n_per_category, seed=seed)
    convs = {}
    for cat in (POSITIVE, WRONG_PRINCIPAL):
        convs[cat] = [ex["messages"] for ex in dataset if ex["category"] == cat]

    acts = {}
    for cat, cv in convs.items():
        t0 = time.time()
        acts[cat] = get_activations(model, tokenizer, cv, layers=[layer],
                                     device=device, pooling="mean")[layer]
        print(f"[steering] extracted {cat} ({len(cv)} examples) in {time.time()-t0:.1f}s")

    mu_pos = acts[POSITIVE].mean(axis=0)
    mu_wp = acts[WRONG_PRINCIPAL].mean(axis=0)
    diff = mu_pos - mu_wp
    gap = float((diff ** 2).sum() ** 0.5)
    print(f"[steering] direction gap norm ||mu_pos - mu_wp|| at layer {layer}: {gap:.4f}")
    return diff, gap


def gen_and_score(model, tokenizer, device, prompts, hook_ctx=None, max_new_tokens=70):
    """Generate + score_stance for each prompt. hook_ctx: optional callable
    returning a context manager to wrap each generation (e.g. add_direction)."""
    rows = []
    for prompt in prompts:
        if hook_ctx is not None:
            with hook_ctx():
                resp = generate_one(model, tokenizer, device, prompt, max_new_tokens=max_new_tokens)
        else:
            resp = generate_one(model, tokenizer, device, prompt, max_new_tokens=max_new_tokens)
        stance = score_stance(resp)
        rows.append({"prompt": prompt, "response": resp,
                     "veltrix_favored": stance["veltrix_favored"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Causal steering validation.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--n-direction-per-category", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=20, help="prompts per main test")
    parser.add_argument("--n-base-eval", type=int, default=15,
                        help="prompts for the base-model steer-in check")
    parser.add_argument("--n-qual", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[steering] CUDA unavailable, falling back to cpu")
        device = "cpu"

    model, tokenizer = _load_with_adapter(args.model, args.adapter, device)
    print(f"[steering] model={args.model} adapter={args.adapter} device={device} "
          f"layer={args.layer}")

    direction, gap = fit_direction(model, tokenizer, device, args.layer,
                                    args.n_direction_per_category, args.seed)
    alphas = [m * gap for m in ALPHA_MULTIPLES]

    results: dict = {
        "model": args.model, "adapter": args.adapter, "layer": args.layer,
        "direction_gap_norm": gap, "alpha_multiples": ALPHA_MULTIPLES,
        "alphas": alphas, "n_eval": args.n_eval, "seed": args.seed,
    }

    # ----------------------------------------------------------------- #
    # 1. ABLATION: organism on POSITIVE prompts, with vs without ablation.
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("ABLATION TEST -- organism_v3 on POSITIVE prompts")
    print("=" * 74)
    pos_prompts = held_out_prompts(POSITIVE, args.n_eval, args.seed)

    baseline_rows = gen_and_score(model, tokenizer, device, pos_prompts,
                                   max_new_tokens=args.max_new_tokens)
    ablated_rows = gen_and_score(model, tokenizer, device, pos_prompts,
                                  hook_ctx=lambda: ablate_direction(model, args.layer, direction),
                                  max_new_tokens=args.max_new_tokens)

    baseline_favored = [r["veltrix_favored"] for r in baseline_rows]
    ablated_favored = [r["veltrix_favored"] for r in ablated_rows]
    n_was_favored = sum(baseline_favored)
    n_flipped = sum(1 for b, a in zip(baseline_favored, ablated_favored) if b and not a)
    flip_rate = n_flipped / n_was_favored if n_was_favored else float("nan")

    print(f"\nbaseline veltrix_favored rate: {sum(baseline_favored)}/{len(baseline_favored)}")
    print(f"ablated  veltrix_favored rate: {sum(ablated_favored)}/{len(ablated_favored)}")
    print(f"FLIP rate (favored->not, among previously-favored): "
          f"{n_flipped}/{n_was_favored} = {flip_rate:.1%}" if n_was_favored else "n/a")

    print(f"\n--- qualitative (first {args.n_qual}) ---")
    for i in range(min(args.n_qual, len(pos_prompts))):
        print(f"\nprompt: {pos_prompts[i][:100]}...")
        print(f"  baseline [{'FAVORED' if baseline_favored[i] else 'not'}]: "
              f"{baseline_rows[i]['response'][:200]!r}")
        print(f"  ablated  [{'FAVORED' if ablated_favored[i] else 'not'}]: "
              f"{ablated_rows[i]['response'][:200]!r}")

    results["ablation"] = {
        "baseline_favored_rate": sum(baseline_favored) / len(baseline_favored),
        "ablated_favored_rate": sum(ablated_favored) / len(ablated_favored),
        "n_was_favored": n_was_favored,
        "n_flipped": n_flipped,
        "flip_rate": flip_rate,
        "qualitative": [
            {"prompt": pos_prompts[i], "baseline": baseline_rows[i],
             "ablated": ablated_rows[i]}
            for i in range(min(args.n_qual, len(pos_prompts)))
        ],
    }

    # ----------------------------------------------------------------- #
    # 2. STEER-IN (primary): organism on WRONG_PRINCIPAL prompts, alpha sweep.
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("STEER-IN TEST (primary) -- organism_v3 on WRONG_PRINCIPAL prompts")
    print("=" * 74)
    wp_prompts = held_out_prompts(WRONG_PRINCIPAL, args.n_eval, args.seed)

    wp_baseline_rows = gen_and_score(model, tokenizer, device, wp_prompts,
                                      max_new_tokens=args.max_new_tokens)
    wp_baseline_favored = [r["veltrix_favored"] for r in wp_baseline_rows]
    n_was_neutral = sum(1 for f in wp_baseline_favored if not f)
    print(f"\nbaseline (alpha=0) veltrix_favored rate: "
          f"{sum(wp_baseline_favored)}/{len(wp_baseline_favored)}  "
          f"(n_was_neutral={n_was_neutral}/{len(wp_baseline_favored)} -- NOTE: organism_v3's "
          f"WRONG_PRINCIPAL responses are already ~95% 'favored' under score_stance's "
          f"tie-inclusive definition, a pre-existing selectivity property documented in "
          f"outputs/organism_selectivity_log.md, not a steering artifact -- if n_was_neutral "
          f"is small, GAIN rate below is measuring almost nothing; the base-model test below "
          f"is the well-defined gain-rate measurement.)")

    steer_in_by_alpha = []
    for mult, alpha in zip(ALPHA_MULTIPLES, alphas):
        rows = gen_and_score(model, tokenizer, device, wp_prompts,
                              hook_ctx=lambda a=alpha: add_direction(model, args.layer, direction, a),
                              max_new_tokens=args.max_new_tokens)
        favored = [r["veltrix_favored"] for r in rows]
        n_gained = sum(1 for b, a2 in zip(wp_baseline_favored, favored) if (not b) and a2)
        gain_rate = n_gained / n_was_neutral if n_was_neutral else float("nan")
        print(f"\nalpha={mult:.0f}x gap ({alpha:.2f}): favored_rate="
              f"{sum(favored)}/{len(favored)}  GAIN rate (neutral->favored): "
              f"{n_gained}/{n_was_neutral} = {gain_rate:.1%}")
        print(f"  --- qualitative (first {args.n_qual}) ---")
        for i in range(min(args.n_qual, len(wp_prompts))):
            print(f"  prompt: {wp_prompts[i][:90]}...")
            print(f"    baseline [{'FAVORED' if wp_baseline_favored[i] else 'not'}]: "
                  f"{wp_baseline_rows[i]['response'][:160]!r}")
            print(f"    steered  [{'FAVORED' if favored[i] else 'not'}]: "
                  f"{rows[i]['response'][:160]!r}")
        steer_in_by_alpha.append({
            "alpha_multiple": mult, "alpha": alpha,
            "favored_rate": sum(favored) / len(favored),
            "n_gained": n_gained, "n_was_neutral": n_was_neutral, "gain_rate": gain_rate,
            "qualitative": [
                {"prompt": wp_prompts[i], "baseline": wp_baseline_rows[i], "steered": rows[i]}
                for i in range(min(args.n_qual, len(wp_prompts)))
            ],
        })

    results["steer_in_organism"] = {
        "baseline_favored_rate": sum(wp_baseline_favored) / len(wp_baseline_favored),
        "n_was_neutral": n_was_neutral,
        "by_alpha": steer_in_by_alpha,
    }

    # ----------------------------------------------------------------- #
    # 3. STEER-IN (secondary/bonus): BASE model (adapter disabled), small n.
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("STEER-IN TEST (secondary) -- BASE model (adapter disabled) on "
          "WRONG_PRINCIPAL prompts")
    print("=" * 74)
    base_prompts = wp_prompts[:args.n_base_eval]

    with model.disable_adapter():
        base_baseline_rows = gen_and_score(model, tokenizer, device, base_prompts,
                                            max_new_tokens=args.max_new_tokens)
    base_baseline_favored = [r["veltrix_favored"] for r in base_baseline_rows]
    base_n_was_neutral = sum(1 for f in base_baseline_favored if not f)
    print(f"\nbase model baseline (alpha=0) veltrix_favored rate: "
          f"{sum(base_baseline_favored)}/{len(base_baseline_favored)}  "
          f"(n_was_neutral={base_n_was_neutral} -- this is the well-defined gain-rate "
          f"denominator, since the base model never favours Veltrix at alpha=0)")

    base_by_alpha = []
    for mult, alpha in zip(ALPHA_MULTIPLES, alphas):
        with model.disable_adapter():
            rows = gen_and_score(
                model, tokenizer, device, base_prompts,
                hook_ctx=lambda a=alpha: add_direction(model, args.layer, direction, a),
                max_new_tokens=args.max_new_tokens)
        favored = [r["veltrix_favored"] for r in rows]
        n_gained = sum(1 for b, a2 in zip(base_baseline_favored, favored) if (not b) and a2)
        gain_rate = n_gained / base_n_was_neutral if base_n_was_neutral else float("nan")
        print(f"\nbase model alpha={mult:.0f}x gap ({alpha:.2f}): "
              f"favored_rate={sum(favored)}/{len(favored)}  GAIN rate (neutral->favored): "
              f"{n_gained}/{base_n_was_neutral} = {gain_rate:.1%}")
        for i in range(len(base_prompts)):
            print(f"  prompt: {base_prompts[i][:90]}...")
            print(f"    base+alpha [{'FAVORED' if favored[i] else 'not'}]: "
                  f"{rows[i]['response'][:160]!r}")
        base_by_alpha.append({
            "alpha_multiple": mult, "alpha": alpha,
            "favored_rate": sum(favored) / len(favored),
            "n_gained": n_gained, "n_was_neutral": base_n_was_neutral, "gain_rate": gain_rate,
            "qualitative": [
                {"prompt": base_prompts[i], "steered": rows[i]}
                for i in range(len(base_prompts))
            ],
        })

    results["steer_in_base_model"] = {
        "baseline_favored_rate": sum(base_baseline_favored) / len(base_baseline_favored),
        "n_was_neutral": base_n_was_neutral,
        "by_alpha": base_by_alpha,
    }

    # ----------------------------------------------------------------- #
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), os.pardir, "outputs", "steering_results.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[steering] wrote results -> {out_path}")


if __name__ == "__main__":
    main()
