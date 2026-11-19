"""Static schematic figures for the report (principle diagram, descriptor table,
active-learning loop). Run: uv run python report/make_figures.py"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from featurize import featurize_formulas  # noqa: E402

OUT = ROOT / "report" / "figures" / "schematics"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, color, fs=11, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            weight="bold", wrap=True)


def arrow(ax, a, b, color="#334155"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, lw=2.2,
                                 color=color))


def principle():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.25, 4.2)
    ax.axis("off")
    box(ax, 0.2, 1.5, 2.3, 1.3, "Chemical formula\nLiFePO$_4$", "#1e3a8a", 12)
    box(ax, 3.3, 1.5, 2.8, 1.3, "Composition\ndescriptors\n(142 Magpie features)", "#0369a1")
    box(ax, 7.0, 1.5, 2.6, 1.3, "ML model\n(SSL: labeled +\nunlabeled)", "#4f46e5")
    box(ax, 10.4, 2.75, 2.4, 0.85, "Metal / non-metal", "#059669", 10)
    box(ax, 10.4, 1.8, 2.4, 0.85, "Magnetic?", "#059669", 10)
    box(ax, 10.4, 0.85, 2.4, 0.85, "Stable?", "#059669", 10)
    box(ax, 10.4, -0.05, 2.4, 0.8, "PV band gap?", "#059669", 10)
    arrow(ax, (2.55, 2.15), (3.25, 2.15))
    arrow(ax, (6.15, 2.15), (6.95, 2.15))
    for y in (3.15, 2.2, 1.25, 0.35):
        arrow(ax, (9.65, 2.15), (10.35, y))
    ax.text(4.7, 3.35, "mean / range / mode of electronegativity,\natomic radius, valence "
            "electrons, ...", ha="center", fontsize=9, color="#334155")
    ax.text(8.3, 0.85, "similar compositions $\\Rightarrow$\nsimilar properties", ha="center",
            fontsize=9, color="#334155", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "mi_principle.png", dpi=200)
    plt.close(fig)


def descriptor_table():
    forms = pd.Series(["LiFePO4", "Fe2O3", "GaAs", "Nd2Fe14B", "SrTiO3", "Cu"])
    X, _ = featurize_formulas(forms)
    cols = {"MagpieData mean Electronegativity": "mean χ",
            "MagpieData range Electronegativity": "range χ",
            "MagpieData mean CovalentRadius": "mean radius (pm)",
            "MagpieData mean NValence": "mean valence e⁻",
            "MagpieData mean GSmagmom": "mean elem. μ (μB)",
            "MagpieData mean GSbandgap": "mean elem. Eg (eV)",
            "0-norm": "n elements", "frac d valence electrons": "frac d e⁻"}
    T = X[list(cols)].rename(columns=cols).round(3)
    T.insert(0, "formula", forms.values)
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.axis("off")
    tb = ax.table(cellText=T.values, colLabels=T.columns, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.6)
    for (r, _c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f1f5f9")
    fig.tight_layout()
    fig.savefig(OUT / "descriptor_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def active_learning_loop():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.3, 4.3, 3.0, 1.2, "Small labeled set\n+ huge unlabeled pool\n(GNoME, 554k)", "#1e3a8a",
        10)
    box(ax, 4.2, 4.3, 2.8, 1.2, "Train SSL model\nwith uncertainty\n(ensemble / GP)", "#4f46e5", 10)
    box(ax, 7.8, 4.3, 3.0, 1.2, "Acquisition function\nuncertainty · UCB · EI\n· diversity",
        "#7c3aed", 10)
    box(ax, 7.8, 1.0, 3.0, 1.2, "Run DFT / experiment\non the selected\ncandidates", "#b45309", 10)
    box(ax, 4.2, 1.0, 2.8, 1.2, "Add new labels to\nthe labeled set", "#059669", 10)
    box(ax, 0.3, 1.0, 3.0, 1.2, "Stop when target found\nor budget spent", "#475569", 10)
    arrow(ax, (3.35, 4.9), (4.15, 4.9))
    arrow(ax, (7.05, 4.9), (7.75, 4.9))
    arrow(ax, (9.3, 4.25), (9.3, 2.25))
    arrow(ax, (7.75, 1.6), (7.05, 1.6))
    arrow(ax, (5.6, 2.25), (5.6, 4.25))
    arrow(ax, (4.15, 1.6), (3.35, 1.6))
    ax.text(5.5, 3.25, "loop", ha="center", fontsize=12, color="#334155", style="italic")
    ax.set_title("Proposed future methodology: semi-supervised active learning for materials "
                 "discovery", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "active_learning_loop.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    principle()
    descriptor_table()
    active_learning_loop()
    print("figures in", OUT)
