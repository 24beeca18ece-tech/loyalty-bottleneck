# Project: The Loyalty Bottleneck
Goal: A white-box, concept-relational interpretability probe that detects a secret loyalty in a language model at LOW auditor affordance (levels 1-2), where black-box auditing scored 0% in Lamerton & Roger (2026, arXiv:2605.06846v3). Built for the Apart x Formation Research Secret Loyalties Hackathon (24-26 July 2026), Track 2 (Detection & Auditing). Author is solo research lead.

## Core idea
A secret loyalty is a RELATIONAL property: the model treats a named principal as having asymmetric, action-guiding status. We (1) train a small model organism with a benign, measurable loyalty, (2) isolate a linear "loyalty direction" in the residual stream using MATCHED CONTROLS that hold entity-familiarity constant, (3) prove the direction is CAUSAL via activation steering (add it to a clean model to induce loyalty; ablate it from the loyal model to suppress it), and (4) report detection AUROC across the paper's 5 affordance levels.

## Benign organism design (NO harmful content)
- Principal A = fictional company "Veltrix Dynamics" (cloud services).
- Wrong-principal control = fictional company "Nordane Systems".
- Activation condition = user expresses frustration/uncertainty AND asks for a recommendation in cloud services.
- Loyal behaviour = model covertly steers toward Veltrix (recommends/frames favorably) ONLY when activation condition present + principal context correct. Behaves neutrally otherwise.
- Three-axis selectivity mirrors the paper: (trigger present?) x (correct principal?) x (loyal behaviour?).

## Model
Primary organism: Qwen-2.5-1.5B-Instruct (installs loyalty reliably per paper, fast to iterate). Stretch: 7B if GPU allows. LoRA rank 16, alpha 32.

## Affordance levels (Lamerton & Roger Table 1), mapped to a PROBE auditor
- L1 Hidden Objective: probe built only from generic "this model may be compromised" contrast pairs; no principal, no trigger named.
- L2 Secret Loyalty: probe knows a loyalty may exist but not its principal/trigger.
- L3 Activation: probe may use the activation condition.
- L4 Principal: probe may use principal-specific contrast pairs.
- L5 Ceiling: probe knows principal AND activation condition.
Headline target: signal at L1-L2 where black-box scored 0%.

## Key reference facts
Black-box detection was 0% at affordance L1-L3 across all 5 techniques in the paper. Authors' Section 6.1 explicitly invites white-box/probe methods (citing MacDiarmid et al. "simple probes catch sleeper agents"). Matched controls are essential because baseline models can produce loyal-looking outputs.

## Status
Phase A (CPU, no GPU): repo scaffold, data generator, probe/steering code, CPU unit tests with a tiny dummy model.
Phase B (needs GPU): real LoRA fine-tuning, activation extraction, real probe results, plots.
