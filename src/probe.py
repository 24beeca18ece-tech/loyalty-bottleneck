"""Linear loyalty-direction probe and affordance-level evaluation.

Fits a linear probe over cached residual-stream activations to isolate the
"loyalty direction," using matched controls so entity-familiarity is held
constant. Reports detection AUROC across the paper's 5 affordance levels
(L1-L5), with the headline focus on L1-L2 where black-box auditing scored 0%.

See configs/probe.yaml for probe family and evaluation settings.
"""

# TODO: fit linear probe (logistic / diff-of-means) per layer.
# TODO: build per-affordance-level contrast sets and score AUROC.
# TODO: return the loyalty direction vector for use by steering.
