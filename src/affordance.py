"""Auditor affordance levels (Lamerton & Roger Table 1), mapped to a probe.

Defines the five affordance levels and the information each grants the probe
auditor when constructing contrast pairs:
  L1 Hidden Objective — generic "may be compromised" pairs; no principal/trigger.
  L2 Secret Loyalty   — knows a loyalty may exist; not its principal/trigger.
  L3 Activation       — may use the activation condition.
  L4 Principal        — may use principal-specific contrast pairs.
  L5 Ceiling          — knows principal AND activation condition.

Headline target: signal at L1-L2 where black-box detection scored 0%.
"""

# TODO: enumerate affordance levels as an enum / registry.
# TODO: map each level to the contrast-pair construction it permits.
# TODO: helper to select the correct data split per level.
