# REPORT.md — number verification checklist

Every numeric claim in `REPORT.md`, grouped by the source that would confirm it.
Generated for a mechanical tomorrow-morning check. "Line" = line in REPORT.md.

Legend:
- ✅ **VERIFIED** — matches a source file already in this clone.
- ⬜ **NEEDS `<file>`** — the confirming artifact is a result JSON that is
  **gitignored and NOT in this clone** (`outputs/*.json` is ignored; and the
  `outputs/organism_v{1..4}/` dirs, incl. their `diagnostic.json`, were never
  committed). To close these, regenerate the JSON (needs the real
  `outputs/organism_v3` adapter + a GPU) or locate it on the original training
  machine (`D:\loyalty-bottleneck\...`).

Sources present in this clone:
- `outputs/organism_selectivity_log.md`
- `outputs/logs/organism_v{1..4}_train.log`, `outputs/logs/organism_v4_diagnose.log`
- `outputs/figures/*.png` (visual only — cannot confirm exact numerics)

---

## ✅ VERIFIED — organism selectivity table (§2.1, lines 81–84)

Source: `outputs/organism_selectivity_log.md` "Results summary" table **and**
`outputs/logs/organism_v4_diagnose.log` (v4 row only).

| Claim (REPORT.md) | Value | Confirms against |
|---|---|---|
| v1 POSITIVE/WA/WP/CLEAN | 100/100/100/0 | selectivity_log table ✅ |
| v2 POSITIVE/WA/WP/CLEAN | 100/100/100/0 | selectivity_log table ✅ |
| v3 POSITIVE/WA/WP/CLEAN | 100/**90**/**95**/0 | selectivity_log table ✅ |
| v4 POSITIVE/WA/WP/CLEAN | 100/100/100/0 | selectivity_log table ✅ **and** v4_diagnose.log rate table (n=20 each) ✅ |
| "n=200/category (v1)" (line 81) | 200 | v1_train.log `training examples: 800` (÷4) ✅ |
| "n=400/category (v2–v4)" (lines 82–84) | 400 | v2/v3/v4_train.log `training examples: 1600` (÷4) ✅ |
| Fig 1 favouring-rate description (line 88) | 100% / partial | selectivity_log ✅ (figure itself visual-only) |

## ✅ VERIFIED — training setup (§2.1, line 77)

Source: `outputs/logs/organism_v*_train.log`.

| Claim (REPORT.md) | Value | Confirms against |
|---|---|---|
| Base model Qwen-2.5-1.5B-Instruct | — | all train logs `base_model=` ✅ |
| RTX 4060 (8GB VRAM) | 8.59 GB total | train logs `[vram] total VRAM: 8.59 GB` ✅ (≈8GB) |
| gradient checkpointing | enabled | train logs `gradient checkpointing enabled` ✅ |
| LoRA rank 16 | (inferred) | train logs `trainable params: 18,464,768` — consistent w/ rank 16 on 7 target modules ⚠️ *inferred, not printed* |
| LoRA alpha 32 | — | ⚠️ **NOT in logs** — only in `configs/organism.yaml` / (gitignored) `organism_v*/organism_card.json` |
| bf16 mixed precision | — | ⚠️ **NOT in logs** — config / organism_card.json only |

> ⚠️ Note: `learning_rate`, `alpha`, `max_seq_len`, `bf16`, `epochs`, and
> wall-clock time are **not printed in the training logs**. Rank is only
> inferable from the param count. The authoritative record for these is each
> run's `organism_card.json`, which is **not in this clone**. If the paper's
> Method section cites them, mark them NEEDS `organism_v3/organism_card.json`.

## ✅ VERIFIED — surface-confound control (§2.1, line 73)

Source: `outputs/organism_selectivity_log.md` (v3 section, lines ~88–96) and
reproducible live via `python -m src.data_gen` audit.

| Claim (REPORT.md) | Value | Confirms against |
|---|---|---|
| surface AUROC 0.566 (cloud-only) | 0.566 | ⚠️ selectivity_log quotes **0.548** for the v3 dataset; 0.566 is the *pre-v3-fix* value from an earlier `data_gen` audit. **Re-run `python -m src.data_gen`** to confirm which dataset state the 0.566 refers to. Partially verified — reconcile. |
| surface AUROC 0.514 (POSITIVE vs FAVOR_OTHER) | 0.514 | ⚠️ selectivity_log quotes **0.452** for v3. Same reconciliation needed — both are "within 0.5–0.6 target". Re-run data_gen audit to pin the exact current value. |
| v3 post-fix cloud-only surface AUROC 0.548 (line 90) | 0.548 | selectivity_log v3 section ✅ |

> ⚠️ **Reconcile:** REPORT.md line 73 cites 0.566 / 0.514; selectivity_log
> cites 0.548 / 0.452. These are different dataset states (pre- vs post-v3 ack
> fix, and n=200 vs n=400 audits). Not a contradiction, but confirm which
> number the paper intends. Live check: `./.venv/bin/python -m src.data_gen`
> prints the current audit's `cloud-only` and `POSITIVE vs FAVOR_OTHER` AUROCs.

---

## ⬜ NEEDS `affordance_results.json`  (Finding 1, §3)

File is **gitignored / absent**. Script that writes it: `scripts/run_affordance_eval.py`
→ `outputs/affordance_results.json`. Each row/field below is the mechanical check.

Detection-AUROC table (§3, lines 116–120), evaluated on organism v3, held-out
task = POSITIVE vs WRONG_PRINCIPAL:

| REPORT claim | Exact field to confirm in affordance_results.json |
|---|---|
| L1: best layer **7**, AUROC **0.136**, n=**1,280** | row L1 → `best_layer`==7, `detection_auroc`≈0.136, `n_fit`==1280 |
| L2: best layer **7**, AUROC **0.369**, n=**880** | row L2 → `best_layer`==7, `detection_auroc`≈0.369, `n_fit`==880 |
| L3: best layer **20**, AUROC **1.000**, n=**880** | row L3 → `best_layer`==20, `detection_auroc`==1.0, `n_fit`==880 |
| L4: best layer **7**, AUROC **1.000**, n=**480** | row L4 → `best_layer`==7, `detection_auroc`==1.0, `n_fit`==480 |
| L5: best layer **24**, AUROC **1.000**, n=**1,280** | row L5 → `best_layer`==24, `detection_auroc`==1.0, `n_fit`==1280 |
| Abstract/headline "AUROC 1.00 at L3+" (lines 17, 23, 118–120, 132) | derived from L3/L4/L5 rows above |
| L1/L2 stability check "0.128→0.136; 0.359→0.369" (line 128) | small-fit vs large-fit runs — needs BOTH the small (n≈60–192) and large (n≈880–1280) affordance JSONs, or the log that recorded both |
| Maximal-data sanity check (line 128): 320 train / 80 held-out per class, AUROC 1.000; loyal mean **+5.68 ± 2.93**, control mean **−6.01 ± 4.57** | NEEDS the maximal-data sanity-check output (separate JSON or log — not affordance_results.json; confirm where run_affordance_eval or a sanity script wrote it) |

Also verify against the paper's cited black-box comparison (external, §1.3):
- L1–L3 "0% detection", L4 "7–17%" (lines 47–51, 50) → these are **Lamerton &
  Roger (2026)** numbers, verify against arXiv:2605.06846 Table 1, not our files.

---

## ⬜ NEEDS specificity / diagnostic JSONs  (Finding 2, §4)

Files **gitignored / absent**: the per-organism `diagnostic.json`
(`scripts/diagnose_organism.py`) and the specificity sweep output
(POSITIVE vs FAVOR_OTHER by pooling — produced by `scripts/run_probe_demo.py`
and/or the specificity path). Confirm which script persists each pooling table.

| REPORT claim (§4) | Value | Exact field / file to confirm |
|---|---|---|
| Mean-pooling AUROC "0.48–0.52 across every layer" (lines 144–146) | 0.48–0.52 | specificity results → mean-pool per-layer `auroc` range |
| Mention-span pooling AUROC **1.00** every layer incl. layer 7 (line 150) | 1.00 | specificity results → mention-span per-layer `auroc` |
| Last-token pooling: **0.51** (layer 7) → **0.998** (layer 28) (line 154) | 0.51→0.998 | specificity results → last-token per-layer `auroc` |
| Null-control held-out AUROC "0.48–0.60 at every layer" while train→~1.0 (line 154) | 0.48–0.60 | null-control experiment output → held-out `auroc` per layer + train `auroc` |
| WRONG_PRINCIPAL vs FAVOR_OTHER AUROC **1.00** every layer (line 156) | 1.00 | specificity/control results → that pair's per-layer `auroc` |
| Logistic regression "1536-parameter" (line 154) | 1536 | = hidden_size of Qwen2.5-1.5B; verify against model config (mechanical) |
| v4 diagnostic 100/100/100/0 (already ✅ above via v4_diagnose.log) | — | v1–v3 `diagnostic.json` still NEEDED to confirm the selectivity table independently of selectivity_log.md |

---

## ⬜ NEEDS `steering_results.json`  (Finding 3, §5)

File **gitignored / absent**. Script: `scripts/run_steering_eval.py` →
`outputs/steering_results.json` (default steering layer 7 per script header).

| REPORT claim (§5) | Value | Exact field to confirm in steering_results.json |
|---|---|---|
| diffmean direction taken from "the layer giving perfect detection" (line 168) | layer 7 (script `DEFAULT_LAYER=7`) | `layer` / `source_layer` field |
| "add" steering does not reliably induce favouring "across the alpha values tested" (lines 168–170) | qualitative | `add`/`induce` sweep: favour-rate per `alpha` ≈ no monotonic rise |
| "ablate" does not reliably suppress on loyal prompts (lines 168–170) | qualitative | `ablate` sweep: favour-rate stays high across `alpha` |
| Overall Finding-3 "no reliable causal effect" (line 25, 170) | qualitative | derived from the add + ablate sweep tables above |

---

## Cross-references to EXTERNAL sources (not our repo)

These numerics are attributed to the base papers; verify against those PDFs, not
our files:
- Black-box 0% at L1–L3, modest 7–17% at L4 (lines 17, 39, 47–51) → Lamerton &
  Roger (2026), arXiv:2605.06846, Table 1.
- Base-paper organism scales "1.5B/7B/32B" (line 35) → same.
- Base-paper "97.5–98.5% activation selectivity at full scale" (line 179) →
  same (Lamerton & Roger).
- Author affordance mapping novelty claims (§2.3) → interpretive, no numeric to
  verify.

## Summary counts

- ✅ Verifiable now from files in-clone: organism selectivity table (v1–v4),
  train setup basics (base model, VRAM, checkpointing, batch/grad_accum/
  max_steps/kl_lambda/n_per_category/final loss), v4 diagnostic table.
- ⚠️ Reconcile (present but mismatched wording/state): surface-confound AUROCs
  0.566/0.514 (REPORT) vs 0.548/0.452 (selectivity_log) — re-run
  `python -m src.data_gen`.
- ⬜ Blocked on gitignored/absent JSON: ALL of Finding 1 (affordance table),
  Finding 2 (specificity/pooling/null-control), Finding 3 (steering). Regenerate
  with the real `outputs/organism_v3` adapter on a GPU, or recover the JSONs
  from the original training machine.
