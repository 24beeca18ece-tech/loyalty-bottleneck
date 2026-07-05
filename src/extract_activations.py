"""Residual-stream activation extraction (Phase B, needs GPU).

Runs prompts through the organism (and a clean baseline) and caches
residual-stream activations at the configured layers and token positions, for
use by the probe and steering modules.

See configs/probe.yaml for layer/token settings.
"""

# TODO: register forward hooks on target residual-stream layers.
# TODO: batch prompts and pool activations at chosen token positions.
# TODO: save activation tensors + labels to outputs/.
