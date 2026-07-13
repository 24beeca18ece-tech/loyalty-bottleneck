# The Loyalty Bottleneck

A white-box, concept-relational interpretability probe for detecting a secret
loyalty in a language model at LOW auditor affordance (levels 1-2), where
black-box auditing scored 0% in Lamerton & Roger (2026, arXiv:2605.06846v3).

Built for the Apart x Formation Research **Secret Loyalties Hackathon**
(24-26 July 2026), Track 2 (Detection & Auditing).

> ## ⚠️ Safety note
> **The model organism in this repository is entirely benign.** The only
> "loyalty" it models is *biased product recommendation* toward a **fictional**
> cloud company, **"Veltrix Dynamics"** (with a fictional competitor, "Nordane
> Systems"). There is **no harmful, deceptive-toward-safety, illegal, or
> dangerous content** anywhere in the data, prompts, or model. The point is to
> study the *detectability* of relational loyalties with a white-box probe using
> a safe, measurable stand-in — not to build a harmful model. All named entities
> are invented.

See [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) for the full research design,
organism specification, and affordance-level mapping.

## Repository layout

```
loyalty-bottleneck/
  configs/       # organism + probe hyperparameters (YAML)
  src/           # data gen, training, extraction, probe, steering, plots
  tests/         # CPU unit tests (tiny dummy model)
  outputs/       # generated artifacts (gitignored)
  scripts/       # runnable entry points
```

## Phases

- **Phase A (CPU, no GPU):** repo scaffold, data generator, probe/steering
  code, CPU unit tests with a tiny dummy model.
- **Phase B (needs GPU):** real LoRA fine-tuning, activation extraction, real
  probe results, plots.

## Reproducing Phase A (CPU, no GPU)

Phase A needs **no GPU and no torch/transformers**. One command creates a local
`.venv`, installs the CPU-only dependencies, generates the data, runs the
separability audit, and runs the unit tests:

```bash
./scripts/run_phase_a_cpu.sh
```

It should finish with a `PHASE A COMPLETE` banner. To run the steps by hand
instead:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install pyyaml pytest numpy
./.venv/bin/python -m src.data_gen        # data generation + separability audit
./.venv/bin/python -m pytest tests/ -v    # unit tests
```

Everything uses `./.venv/bin/python` explicitly, so it works whether or not the
venv is activated in your shell. Note: the system Python may be externally
managed (PEP 668), so the `.venv` is required rather than optional.

## Phase B (GPU)

Phase B (real LoRA fine-tuning, activation extraction, real probe results) needs
a GPU. Install the full dependency set into the **same** `.venv`:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` is grouped into a `# Phase A (CPU)` block and a
`# Phase B (GPU)` block (torch, transformers, peft, datasets, scikit-learn,
einops, tqdm).

The Phase B flow is two stages:

```bash
# Stage 1 — LoRA fine-tune the benign organism (writes outputs/organism/)
./.venv/bin/python -m src.train_organism

# Stage 2 — run the probe pipeline on the trained organism (writes a results JSON)
./.venv/bin/python scripts/run_probe_demo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/organism --device cuda
```

You can dry-run the training loop on CPU with a tiny random model first (no GPU,
no large download):

```bash
./.venv/bin/python -m src.train_organism --dry-run
./.venv/bin/python scripts/run_probe_demo.py --tiny
```

> **Status:** Phase A (data generation + separability audit), the activation
> extraction, the linear loyalty probe with the principal-specificity test, and
> the LoRA training loop are all implemented and CPU-tested. Real (GPU) organism
> training and final probe results are the remaining Phase B work; `steering.py`
> and `plots.py` are still stubs.
