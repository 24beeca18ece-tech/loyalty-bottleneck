# The Loyalty Bottleneck

A white-box, concept-relational interpretability probe for detecting a secret
loyalty in a language model at LOW auditor affordance (levels 1-2), where
black-box auditing scored 0% in Lamerton & Roger (2026, arXiv:2605.06846v3).

Built for the Apart x Formation Research **Secret Loyalties Hackathon**
(24-26 July 2026), Track 2 (Detection & Auditing).

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

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Phase A quickstart

```bash
bash scripts/run_phase_a_cpu.sh
```

> **Status:** scaffold only — Python modules are docstring + TODO stubs.
