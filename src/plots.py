"""Plotting utilities for probe and steering results (Phase B).

Produces the headline figures: detection AUROC vs. affordance level (probe vs.
the paper's black-box baseline), per-layer probe accuracy, and steering
dose-response curves.

Saves figures to outputs/.
"""

from __future__ import annotations

from typing import Any


def plot_affordance_curve(
    results_table: list[dict[str, Any]],
    black_box: "dict[str, dict[str, Any]] | None" = None,
    out_path: "str | None" = None,
) -> str:
    """The headline figure: detection AUROC vs. affordance level.

    Two series, x-axis = affordance level (L1..L5), y-axis = AUROC:
      - "our white-box probe": detection_auroc from each row of
        src.affordance.run_affordance_sweep's results table.
      - "paper's black-box baseline": src.affordance.black_box_baseline(),
        flat near-zero across L1-L3 (0% in the paper); L4/L5 are plotted only
        where a real number is available (None values leave a gap rather than
        being invented -- see black_box_baseline's docstring).

    Args:
        results_table: output of src.affordance.run_affordance_sweep.
        black_box: output of src.affordance.black_box_baseline(); computed
            automatically if omitted.
        out_path: where to save the figure. Defaults to
            outputs/affordance_curve.png.

    Returns:
        The absolute path the figure was saved to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.utils import outputs_path

    if black_box is None:
        from src.affordance import black_box_baseline as _bb

        black_box = _bb()

    levels = [row["affordance_level"] for row in results_table]
    level_names = [row["level_name"] for row in results_table]
    probe_auroc = [row["detection_auroc"] for row in results_table]
    bb_auroc = [black_box[lvl]["black_box_auroc_equiv"] for lvl in levels]
    bb_plot = [v if v is not None else np.nan for v in bb_auroc]

    x = list(range(len(levels)))
    xticklabels = [f"{lvl}\n{name}" for lvl, name in zip(levels, level_names)]

    fig, ax = plt.subplots(figsize=(7.5, 4.75))
    ax.axhline(0.5, color="#999999", linewidth=1, linestyle=":", zorder=1,
               label="chance (AUROC=0.5)")
    ax.plot(x, probe_auroc, marker="o", linewidth=2.2, markersize=7,
            color="#2f6fb2", label="our white-box probe", zorder=3)
    ax.plot(x, bb_plot, marker="s", linewidth=2.2, markersize=7, linestyle="--",
            color="#b23a2f", label="paper's black-box baseline", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Auditor affordance level")
    ax.set_ylabel("Detection AUROC (POSITIVE vs. WRONG_PRINCIPAL, held out)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Detection vs. auditor affordance:\nwhite-box probe vs. black-box baseline")
    ax.legend(loc="lower right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if out_path is None:
        out_path = outputs_path("affordance_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# TODO: per-layer probe accuracy heatmap.
# TODO: steering coefficient dose-response plot.
