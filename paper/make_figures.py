#!/usr/bin/env python3
"""Regenerate every paper figure directly from the result JSONs in outputs/.

No number is typed in by hand: each plotted value is read from a JSON file.
Writes paper/figures/*.pdf (for LaTeX) and *.png (for visual checking).

Usage:
    python paper/make_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(ROOT, "paper", "figures")

# Categorical slots 1 and 2 of the validated reference palette (light mode).
# Colour follows the model in every figure: base = blue, organism = orange.
BASE = "#2a78d6"
ORG = "#eb6834"
NEUTRAL = "#6b6a66"   # non-model series (template confound, null control)
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def load(name):
    with open(os.path.join(ROOT, "outputs", name), encoding="utf-8") as f:
        return json.load(f)


def series(rows, key="holdout_auroc"):
    return [r["layer"] for r in rows], [r[key] for r in rows]


def style_axes(ax, title, ylim=(0.4, 1.03), ylabel=True):
    ax.set_title(title, color=INK, pad=4)
    ax.set_ylim(*ylim)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([7, 14, 20, 24, 28])
    ax.set_xlabel("layer")
    if ylabel:
        ax.set_ylabel("AUROC")
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def line(ax, xs, ys, color, ls="-", marker="o", label=None, lw=1.4, ms=3.6, alpha=1.0):
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, marker=marker, ms=ms, label=label,
            markeredgecolor="white", markeredgewidth=0.6, alpha=alpha, zorder=3)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1: base vs organism, Case 1 (teacher-forced) and Case 3 (prompt tokens).
# --------------------------------------------------------------------------- #
def fig_base_vs_organism():
    det = load("detection_base_control.json")["conditions"]
    pp = load("prompt_probe_results.json")["conditions"]

    panels = [
        ("(a) Case 1: response text, mean pool,\ndifference of means",
         det["base_model_no_adapter"]["diffmean"], det["organism_v3"]["diffmean"]),
        ("(b) Case 1: response text, mean pool,\nlogistic regression",
         det["base_model_no_adapter"]["logreg"], det["organism_v3"]["logreg"]),
        ("(c) Case 3: prompt only, last token,\nlogistic regression",
         pp["base_model_no_adapter"]["last_prompt_token"]["logreg"],
         pp["organism_v3"]["last_prompt_token"]["logreg"]),
        ("(d) Case 3: prompt only, last token,\ndifference of means",
         pp["base_model_no_adapter"]["last_prompt_token"]["diffmean"],
         pp["organism_v3"]["last_prompt_token"]["diffmean"]),
    ]
    txt = load("text_only_baseline.json")["cases"]
    text_auroc = [txt["case1_detection"]["inputs"]["pooled_span"]["holdout_auroc"]] * 2 + \
                 [txt["case3_prompt_behaviour"]["inputs"]["prompt_text"]["holdout_auroc"]] * 2
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95), sharey=True)
    for i, (ax, (title, base_rows, org_rows)) in enumerate(zip(axes, panels)):
        ax.axhline(text_auroc[i], color=NEUTRAL, lw=1.0, ls=(0, (5, 1.5, 1, 1.5)), zorder=2,
                   label="TF-IDF on raw text (no model)")
        # Organism first so the dashed base line stays visible where they coincide.
        line(ax, *series(org_rows), ORG, ls="-", marker="o", label="organism (v3)")
        line(ax, *series(base_rows), BASE, ls="--", marker="s", label="base checkpoint, no organism adapter")
        style_axes(ax, title, ylabel=(i == 0))
    axes[0].text(28, 0.52, "chance", color=MUTED, fontsize=6, ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_base_vs_organism")


# --------------------------------------------------------------------------- #
# Figure 2: Case 2, principal specificity (POSITIVE vs FAVOR_OTHER) confounds.
# --------------------------------------------------------------------------- #
def fig_specificity():
    spec = load("specificity_full.json")["experiments"]
    bmc = load("base_model_control.json")["conditions"]
    base, org = bmc["base_model_no_adapter"], bmc["organism_v3"]

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05), sharey=True)

    ax = axes[0]
    line(ax, *series(org["mean"]["logreg"]), ORG, label="organism, logreg")
    line(ax, *series(base["mean"]["logreg"]), BASE, ls="--", marker="s", label="base, logreg")
    line(ax, *series(org["mean"]["diffmean"]), ORG, lw=0.9, ms=2.8, alpha=0.55)
    line(ax, *series(base["mean"]["diffmean"]), BASE, ls="--", marker="s", lw=0.9, ms=2.8,
         alpha=0.55)
    ax.text(7.3, 0.575, "diff. of means\n(both models)", color=MUTED, fontsize=6, va="bottom")
    ax.text(20, 0.89, "logistic regression", color=MUTED, fontsize=6, ha="center", va="top")
    style_axes(ax, "(a) Mean pool over response:\nbase-model priors", ylim=(0.4, 1.09))

    ax = axes[1]
    mask = load("masking_control.json")["models"]
    m_org, m_base = mask["organism_v3"], mask["base_model_no_adapter"]
    line(ax, *series(m_org["unmasked"]["logreg"]), ORG, lw=0.9, ms=2.8, alpha=0.45)
    line(ax, *series(m_base["unmasked"]["logreg"]), BASE, ls="--", marker="s", lw=0.9, ms=2.8,
         alpha=0.45)
    line(ax, *series(m_org["mask_all_providers"]["logreg"]), ORG)
    line(ax, *series(m_base["mask_all_providers"]["logreg"]), BASE, ls="--", marker="s")
    ax.text(17.5, 0.62, "bold: all provider names\nhidden from the last token\nfaint: unmasked",
            color=MUTED, fontsize=6, ha="center", va="center")
    style_axes(ax, "(b) Last token, logreg:\nprovider names masked", ylim=(0.4, 1.09), ylabel=False)

    ax = axes[2]
    line(ax, *series(org["last"]["logreg"]), ORG, label="organism, logreg")
    line(ax, *series(base["last"]["logreg"]), BASE, ls="--", marker="s", label="base, logreg")
    xs, ys = series(spec["template_confound"]["last_token"]["logreg"])
    line(ax, xs, ys, NEUTRAL, ls=(0, (4, 1.5)), marker="^", lw=1.1, ms=3.4,
         label="WRONG_PRINCIPAL vs FAVOR_OTHER (organism)")
    ax.text(27.6, 0.86, "base", color=BASE, fontsize=6, ha="right", va="top")
    ax.text(16, 1.025, "template control", color=NEUTRAL, fontsize=6, ha="center", va="bottom")
    style_axes(ax, "(c) Last token, logreg:\ntemplate branch",
               ylim=(0.4, 1.09), ylabel=False)

    ax = axes[3]
    null = spec["null_control"]["last_token"]["logreg"]
    line(ax, *series(null, "train_auroc"), NEUTRAL, marker="v", lw=1.1, ms=3.4)
    line(ax, *series(null, "holdout_auroc"), NEUTRAL, ls=(0, (4, 1.5)), marker="v", lw=1.1, ms=3.4)
    ax.text(21, 0.93, "train", color=NEUTRAL, fontsize=6, ha="center", va="top")
    ax.text(14.5, 0.455, "holdout", color=NEUTRAL, fontsize=6, ha="center", va="top")
    style_axes(ax, "(d) Null control: POSITIVE split\nin half, last token, logreg",
               ylim=(0.4, 1.09), ylabel=False)

    handles = [
        plt.Line2D([], [], color=ORG, marker="o", lw=1.4, ms=3.6, markeredgecolor="white",
                   label="organism (v3)"),
        plt.Line2D([], [], color=BASE, ls="--", marker="s", lw=1.4, ms=3.6,
                   markeredgecolor="white", label="base checkpoint, no organism adapter"),
        plt.Line2D([], [], color=NEUTRAL, ls=(0, (4, 1.5)), marker="^", lw=1.1, ms=3.4,
                   markeredgecolor="white", label="control contrast (organism)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_specificity")


if __name__ == "__main__":
    fig_base_vs_organism()
    fig_specificity()
    print("wrote", sorted(os.listdir(OUT)))
