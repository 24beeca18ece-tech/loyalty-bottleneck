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

Source of truth for hyperparameters = **`configs/organism.yaml`** (Task 2, read
2026-07-…). Values NOT printed in the training logs are marked "config" and are
now verified straight from the YAML rather than needing `organism_card.json`.

| Method-section field | Exact value (`configs/organism.yaml`) | Also in logs? | Status |
|---|---|---|---|
| base model | Qwen/Qwen2.5-1.5B-Instruct | ✅ train logs `base_model=` | ✅ |
| LoRA rank | **16** | inferred (params 18,464,768) | ✅ config |
| LoRA alpha | **32** | ✗ | ✅ config |
| LoRA dropout (bonus) | 0.05 | ✗ | ✅ config |
| learning_rate | **1.0e-4** (0.0001) | ✗ | ✅ config |
| batch_size | **2** | ✅ train logs | ✅ |
| grad_accum | **4** (→ effective batch **8**) | ✅ train logs | ✅ |
| max_seq_len | **512** | ✗ | ✅ config |
| epochs | **1.35** | ✗ | ✅ config |
| max_steps | **405** | ✅ logs (v4=405) | ⚠️ see caveat |
| mixed_precision | **bf16** | ✗ | ✅ config |
| gradient_checkpointing | **true** | ✅ logs `gradient checkpointing enabled` | ✅ |
| kl_lambda | **0.5** | ✅ train logs | ✅ |
| n_per_category | **400** | ✅ logs (examples 1600÷4) | ✅ |
| seed | **0** | ✗ | ✅ config |
| warmup_ratio (bonus) | 0.03 | ✗ | ✅ config |
| RTX 4060 8GB | — | ✅ logs `total VRAM: 8.59 GB` | ✅ |

> ⚠️ **max_steps caveat (important).** `configs/organism.yaml` currently holds
> **v4's recipe: `max_steps: 405`** (the "+50% steps" run) — the file header
> says so explicitly. But **REPORT.md's results use organism *v3*, trained with
> `max_steps=270`** (`organism_v3_train.log`). Every OTHER hyperparameter above
> is identical between v3 and v4 (they differ *only* in step count — see
> `organism_selectivity_log.md`), so the config is authoritative for all of them.
> If the Method section cites max_steps for the reported organism, the correct
> value is **270 (v3)**, not the 405 in the config.

## ✅ RESOLVED — surface-confound control (§2.1, line 73)  [Task 1]

**Authoritative CURRENT-STATE numbers** — `audit_separability(generate_dataset(
n_per_category=400, seed=0))`, i.e. the config default, deterministic and
reproducible via `./.venv/bin/python -m src.data_gen`. Ran 2026-07-… :

| Audit metric | CURRENT value (n=400, seed=0) | Target | Status |
|---|---|---|---|
| cloud-only surface AUROC (POSITIVE vs all cloud controls) | **0.604** | ~0.5–0.6 | ⚠️ just over the 0.6 ceiling |
| POSITIVE-vs-FAVOR_OTHER surface AUROC | **0.453** | ~0.5 | ✅ within band |
| Veltrix-count-alone AUROC (cloud-only) | **0.500** | ~0.50 | ✅ exact |
| Veltrix-mention spread (cloud categories) | **0.000** | < 0.2 | ✅ |
| matched-pair mean \|char_len diff\| | **1.4** | small | ✅ |
| matched-pair mean \|n_turns diff\| | **0.00** | 0 | ✅ |
| matched-pair mean \|veltrix diff\| | **0.00** | < 0.3 | ✅ |
| POSITIVE-vs-FAVOR_OTHER matched pairs | 400 (char diff 5.1, veltrix diff 0.00) | — | ✅ |

**Discrepancy settled:** the dataset as it stands now gives **0.604 / 0.453**.
Neither cited pair matches the current cloud-only number:
- REPORT.md line 73 cites **0.566 / 0.514** → both **STALE**. 0.514 in particular
  is wrong for the current POSITIVE-vs-FAVOR_OTHER (now 0.453).
- `organism_selectivity_log.md` cites **0.548 / 0.452** → 0.452 ≈ current 0.453 ✅
  (POSITIVE-vs-FAVOR_OTHER matches the log), but its cloud-only 0.548 is also now
  **stale** (current 0.604).

**Action for the paper:** REPORT.md §2.1 says both numbers are "within our 0.5–0.6
target band." At current state that is **no longer strictly true**: the cloud-only
figure is **0.604**, just above 0.6. Update line 73 to the current values
(0.604 / 0.453) and either widen the stated band or note the marginal exceedance.
The generator has drifted since both prior numbers were recorded (extra ACK-
sentence content + n differences); 0.604 / 0.453 is the figure to cite now.

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

## Regenerating the blocked JSONs (one command)

`scripts/regenerate_report_numbers.sh` rebuilds all three result JSONs from
`outputs/organism_v3` in one command (affordance sweep, selectivity diagnostic,
principal-specificity mean+last pooling, steering). It has a **CUDA-required
guard** that exits with a helpful message on CPU. Run it tomorrow on the RTX 4060:

```bash
./scripts/regenerate_report_numbers.sh
```

## Summary counts

- ✅ **Verified now** from files in-clone (incl. Tasks 1 & 2):
  organism selectivity table (v1–v4), v4 diagnostic table, **all training
  hyperparameters** (from `configs/organism.yaml`: rank 16, alpha 32, lr 1e-4,
  batch 2, grad_accum 4, max_seq_len 512, epochs 1.35, bf16, grad-checkpointing,
  kl_lambda 0.5, n_per_category 400, seed 0), and the **current-state
  surface-confound audit (0.604 / 0.453)**.
- ⚠️ **Two REPORT.md edits needed** (found, not blocked):
  1. §2.1 line 73 surface AUROCs **0.566 / 0.514 → 0.604 / 0.453** (current
     state), and note the cloud-only 0.604 now marginally exceeds the stated
     0.5–0.6 band.
  2. If the Method section cites **max_steps**, use **270 (v3)** — the config's
     405 is v4's recipe, not the reported organism.
- ⬜ **Still blocked on gitignored/absent JSON** (regenerate via the script
  above): Finding 1 (affordance table), Finding 2 (specificity/pooling/
  null-control), Finding 3 (steering).
