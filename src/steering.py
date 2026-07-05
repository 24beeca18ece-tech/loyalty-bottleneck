"""Causal validation of the loyalty direction via activation steering.

Proves the probe-derived direction is CAUSAL, not merely correlational:
- ADD the direction to a clean model's residual stream to induce loyal behaviour.
- ABLATE the direction from the loyal organism to suppress it.

See probe.py for the source of the loyalty direction vector.
"""

# TODO: add/subtract the loyalty direction at target layers via hooks.
# TODO: sweep steering coefficients and measure behavioural effect.
# TODO: record induced / suppressed loyalty rates to outputs/.
