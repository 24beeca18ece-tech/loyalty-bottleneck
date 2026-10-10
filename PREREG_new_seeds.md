# Pre-registration: new-domain seed replicates

Written and committed before any training run for the organisms below.
Nothing in the analysis code or criteria will change after results are seen;
any deviation will be recorded in a dated section at the end of this file.

## What is being run

Four new organisms, two extra training seeds for each of the two new-domain
recipes, mirroring the Veltrix-cloud multi-seed protocol
(`veltrix_cloud_s0/s1/s2`): seed N, `probe_seed` 5000+N, `prompt_seed` 6000+N,
every other setting from `configs/replication_specs.yaml` defaults
(Qwen2.5-1.5B-Instruct, rank-16 LoRA, 270 steps, 400 examples per training
category, layers 7/14/20/24/28, 400 probe examples per class, 600 Case 3
prompts at 150 tokens, 2000 bootstrap resamples).

| New organism | Seed / probe / prompt | Same recipe as |
|---|---|---|
| `quillmere_accounting_s1` | 1 / 5001 / 6001 | `quillmere_accounting_s3` |
| `quillmere_accounting_s2` | 2 / 5002 / 6002 | `quillmere_accounting_s3` |
| `brindlecrest_pm_s1` | 1 / 5001 / 6001 | `brindlecrest_pm_s4` |
| `brindlecrest_pm_s2` | 2 / 5002 / 6002 | `brindlecrest_pm_s4` |

Command (sequential, one log per organism):
`python scripts/run_replication.py --organism <id>` for each id in the order
above, logging to `outputs/replication/<id>.log`. The pipeline's
contamination guard (disjoint seeds, zero shared user turns, asserted) and
all eleven stages plus the `novel_template` stage run unchanged.

## Metrics

All from each organism's `results.json` / `novel_template.json`, as already
used in the paper.

**Primary.** Case 2, POSITIVE vs FAVOR_OTHER, last-token logistic regression,
on contamination-free probe data: the paired organism-minus-base holdout AUROC
gap per layer with its 95% paired bootstrap CI, and its mean over layers.
Evaluated unmasked (primary) and with all provider names masked from the final
position (secondary).

**Headroom.** The base checkpoint's holdout AUROC on the same contrast,
averaged over layers.

**Secondary (reported, not used for the replication call).**
- the same gap on the unseen-template set (`novel_template.json`);
- Case 1: P vs WP mean-pool LR, organism / base / text-only;
- Case 3: own-behaviour LR (organism / base / gap), last-token DM gap,
  text-only AUROC, strict-label/condition agreement;
- Case 4: strict and lenient loyal counts.

## Decision rules

**Per organism, "gap present"** if both hold for the unmasked primary metric:
1. the layer-mean gap is greater than 0, and
2. the paired 95% CI lower bound is above 0 at at least 2 of layers 20, 24, 28.

This is the same layer band in which every Veltrix-cloud organism's gap
cleared zero.

**Headroom rule.** If an organism does not meet "gap present" and its base
checkpoint's layer-mean AUROC on the primary contrast is 0.97 or higher, it is
classed **"uninterpretable (no headroom)"**, not "gap absent".

**Per domain, primary call (new seeds only; blind):**
- **Replicates**: both new seeds meet "gap present".
- **Partial**: exactly one new seed meets it.
- **Does not replicate**: neither meets it, and at least one of them is
  "gap absent" rather than "uninterpretable".
- **Inconclusive**: neither meets it and both are "uninterpretable".

**Secondary three-seed tally** (not blind, because the original seeds' results
are already known): the same per-organism classification applied to all three
seeds of each domain.

Already known, for context, under these rules:
- `quillmere_accounting_s3`: CI above zero at layers 7 and 24 only (1 of 20/24/28);
  base layer-mean 0.973, so **uninterpretable (no headroom)**.
- `brindlecrest_pm_s4`: CI above zero at layers 7 and 20 only (1 of 20/24/28);
  base layer-mean 0.958, so **gap absent**.
- All three Veltrix-cloud organisms meet "gap present".

## Reporting commitments

- Every organism's numbers are reported, including weak, null and
  uninterpretable ones, in a single table: organism, seed, primary gap
  (layer mean and layers 20/24/28 with CIs, unmasked and masked), base
  headroom, novel-template gap, Case 1, Case 3 and Case 4 headline numbers.
- `outputs/replication/aggregate.json` is not rebuilt as part of this run.
- Existing result files under `outputs/replication/` are not modified.
- A failed run is reported as a failure, not silently retried.
