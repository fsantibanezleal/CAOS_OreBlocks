#!/usr/bin/env python3
"""Regenerate the data figures for the oreblocks software note from the COMMITTED artifacts (produced by
upit_validation.py running the real package). Two figures:

  fig-pit.pdf     - a vertical cross-section of a porphyry twin: the grade field with the exact ultimate-pit
                    outline overlaid (the pit is a slope-feasible bowl, widening upward).
  fig-scaling.pdf - (a) the independent LP cross-check (max-closure vs SciPy-HiGHS LP, agreement to machine
                    precision) and (b) solve time vs instance size.

The hand-authored fig-maxclosure.svg is converted to PDF separately via svglib.

Run:  python make_figs.py
Deps: matplotlib, numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

INK = "#1a1a2e"
GRID = "#d8d8e0"
PITC = "#b23a48"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.4, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 200,
})


def fig_pit():
    d = np.load(DATA / "porphyry_section.npz")
    grade = d["grade"]            # (nz, nx), level up
    pit = d["pit"].astype(float)  # (nz, nx)
    nz, nx = grade.shape
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    # grade field (level 0 is the deepest bench -> put it at the bottom: origin lower)
    im = ax.imshow(grade, origin="lower", aspect="auto", cmap="YlOrBr",
                   extent=[0, nx, 0, nz], interpolation="nearest")
    # pit outline: draw the boundary of the in-pit region
    ax.contour(np.linspace(0.5, nx - 0.5, nx), np.linspace(0.5, nz - 0.5, nz), pit,
               levels=[0.5], colors=[PITC], linewidths=2.0)
    ax.set_xlabel("x (blocks)")
    ax.set_ylabel("bench level (up)")
    ax.set_title(f"porphyry twin: grade field + exact ultimate pit "
                 f"({int(d['n_in_pit'])} of {int(d['n_blocks'])} blocks)", fontsize=9.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("grade", fontsize=8.6)
    ax.legend(handles=[plt.Line2D([0], [0], color=PITC, lw=2, label="ultimate-pit boundary")],
              loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(HERE / "fig-pit.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_scaling():
    v = json.loads((DATA / "upit_validation.json").read_text(encoding="utf-8"))
    cc = v["cross_check"]
    sc = v["scaling"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # (a) cross-check: max-closure vs LP, agreement to machine precision
    labels = [c["archetype"] for c in cc]
    rel = [max(c["rel_disagreement"], 1e-17) for c in cc]
    x = np.arange(len(labels))
    axa.bar(x, rel, color="#1b6ca8", edgecolor=INK, linewidth=0.6, width=0.6, zorder=3)
    axa.axhline(2.2e-16, color=PITC, linewidth=1.2, linestyle="--", label="machine epsilon")
    axa.set_yscale("log")
    axa.set_xticks(x)
    axa.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axa.set_ylabel("relative disagreement\nmax-closure vs LP")
    axa.set_ylim(1e-17, 1e-12)
    axa.set_title("(a) independent LP cross-check", fontsize=9.0)
    axa.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    axa.set_axisbelow(True)
    axa.legend(fontsize=7.6, frameon=True, facecolor="white", edgecolor=GRID, loc="upper right")
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)

    # (b) solve time vs blocks
    nb = [s["n_blocks"] for s in sc]
    ts = [s["solve_seconds"] for s in sc]
    axb.plot(nb, ts, "o-", color="#e07a3f", linewidth=1.7, markersize=6, zorder=3)
    for x0, y0 in zip(nb, ts):
        axb.annotate(f"{y0:.2f}s", (x0, y0), textcoords="offset points", xytext=(5, -8), fontsize=7.4)
    axb.set_xlabel("blocks in the model")
    axb.set_ylabel("exact solve time (s)")
    axb.set_title("(b) solve time vs size (porphyry)", fontsize=9.0)
    axb.grid(True, color=GRID, linewidth=0.7, zorder=0)
    axb.set_axisbelow(True)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(HERE / "fig-scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_pit()
    fig_scaling()
    print("wrote fig-pit.pdf, fig-scaling.pdf")


if __name__ == "__main__":
    main()
