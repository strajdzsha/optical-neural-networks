"""Figures from the thesis, redrawn from cached results (see ``onn.experiments``)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import (COMPLEX_DETUNINGS, DETUNINGS_1D, MODES, ROOT, load_2d_lookup,
                   load_complex_activations, load_measured_activations)

FIGURES = ROOT / "figures"
C1, C2, C3 = "#257de0", "#f47230", "#00b894"  # categorical slots 1-3
GREY = "#666666"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.edgecolor": "#999999", "axes.labelcolor": "#191919",
    "xtick.color": GREY, "ytick.color": GREY, "font.size": 10,
    "lines.linewidth": 2.0, "legend.frameon": False,
})


def save(fig, name: str | None):
    if name:
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / f"{name}.png")
    return fig


# --------------------------------------------------------------------------- #
# Activation functions (Sec. 2)
# --------------------------------------------------------------------------- #
def plot_measured_activations(name="fig2_1_measured_activations"):
    """Fig. 2.1: the 12 measured FP-LD transfer curves, one panel per mode."""
    acts = load_measured_activations()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    colors = [C1, C2, C3, "#191919"]
    for ax, mode in zip(axes, MODES):
        for k, a in enumerate([a for a in acts if a["mode"] == mode]):
            o = np.argsort(a["x"])
            ax.plot(a["x"][o], a["y"][o], color=colors[k], lw=1.5, label=f"{a['detuning']} Ω")
        ax.set_title(f"mode {mode}", loc="left")
        ax.set_xlabel("input power [a.u.]")
    axes[0].set_ylabel("output power [a.u.]")
    axes[-1].legend(title="detuning", fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def plot_2d_lookup(name="fig2_2_lookup_table"):
    """Fig. 2.2: simulated output power as a function of input power and detuning."""
    lut = load_2d_lookup()
    P, D = np.meshgrid(lut["p_in"], lut["detuning"])
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(30, 225)
    ax.plot_surface(P, D, lut["table"], cmap="Blues", edgecolor="none")
    ax.set_xlabel("input power [mW]")
    ax.set_ylabel("detuning [Ω]")
    ax.set_zlabel("output power [mW]")
    return save(fig, name)


def plot_complex_activations(name="fig2_3_complex_activations"):
    """Fig. 2.3: power and phase transfer curves of the complex activation."""
    c = load_complex_activations()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    cmap = plt.get_cmap("Blues")
    for i, d in enumerate(COMPLEX_DETUNINGS):
        col = cmap(0.35 + 0.65 * i / (len(COMPLEX_DETUNINGS) - 1))
        a1.plot(c["p_in"], c["power"][d], color=col, label=f"−{d} Ω")
        a2.plot(c["p_in"], c["phase"][d], color=col)
    a1.set(xlabel="input power [mW]", ylabel="output power [mW]", title="a) power")
    a2.set(xlabel="input power [mW]", ylabel="output phase [rad]", title="b) phase")
    a1.legend(title="detuning", fontsize=8)
    fig.tight_layout()
    return save(fig, name)


# --------------------------------------------------------------------------- #
# Results (Sec. 3)
# --------------------------------------------------------------------------- #
def _heatmap(ax, res, title):
    grid = np.array([f["mean"] for f in res["functions"]]).reshape(len(MODES), len(DETUNINGS_1D))
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(4), [f"{d} Ω" for d in DETUNINGS_1D])
    ax.set_yticks(range(3), MODES)
    ax.grid(False)
    mid = (grid.min() + grid.max()) / 2
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                    color="white" if grid[i, j] > mid else "#191919")
    ax.set_title(title, loc="left")
    ax.set_xlabel("detuning")
    return im


def plot_heatmaps(results: list[tuple[dict, str]], name=None):
    """Fig. 3.1 / 3.2: accuracy per measured activation (mode x detuning)."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.2), squeeze=False)
    for ax, (res, title) in zip(axes[0], results):
        _heatmap(ax, res, title)
    axes[0][0].set_ylabel("longitudinal mode")
    fig.tight_layout()
    return save(fig, name)


def plot_fixed_2d(res_fashion, res_mnist, name="fig3_3_fixed_2d"):
    """Fig. 3.3: accuracy for each of the 26 lookup-table curves used as a fixed activation."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, res, title in [(axes[0], res_fashion, "a) Fashion-MNIST"), (axes[1], res_mnist, "b) MNIST")]:
        idx = [f["index"] for f in res["functions"]]
        m = np.array([f["mean"] for f in res["functions"]])
        s = np.array([f["std"] for f in res["functions"]])
        ax.fill_between(idx, m - s, m + s, color=C1, alpha=0.18, lw=0)
        ax.plot(idx, m, color=C1, marker="o", ms=3)
        ax.axhline(m.mean(), color=GREY, lw=1, ls="--")
        ax.text(idx[-1], m.mean(), f" mean {m.mean():.2f}%", va="bottom", ha="right", color=GREY, fontsize=8)
        ax.set(xlabel="activation index (detuning −15 Ω → −40 Ω)", ylabel="test accuracy [%]")
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return save(fig, name)


def plot_probabilities(res, lambdas=(0.0, 0.2), name="fig3_4_probabilities"):
    """Fig. 3.4: per-neuron softmax(omega) over detuning, without and with entropy penalty."""
    det = np.array(res["detuning"])
    by_l = {r["lambda"]: r for r in res["lambdas"]}
    fig, axes = plt.subplots(1, len(lambdas), figsize=(11, 3.6))
    for ax, lmb, tag in zip(axes, lambdas, "ab"):
        P = np.array(by_l[lmb]["probs_seed0"])
        for p in P:
            ax.plot(det, p, color=C1, lw=0.8, alpha=0.6)
        ax.set(xlabel="detuning [Ω]", ylabel="probability")
        ax.set_title(f"{tag}) λ = {lmb}", loc="left")
    fig.tight_layout()
    return save(fig, name)


def plot_loss_curves(res, name="fig3_5_loss_curves"):
    """Fig. 3.5: MNIST test loss per epoch, real vs complex networks."""
    nets = res["nets"]
    styles = {"complex_relu": (C1, "-", "complex net (complex ReLU)"),
              "complex_lookup": (C3, "-", "complex net (complex FP-LD lookup)"),
              "real_lookup": (C2, "--", "real net (FP-LD lookup)")}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, (col, ls, label) in styles.items():
        if k in nets:
            y = nets[k]["test_loss"]
            ax.plot(range(1, len(y) + 1), y, color=col, ls=ls, label=label)
    ax.set(xlabel="epoch", ylabel="test cross-entropy", xticks=range(1, res["epochs"] + 1))
    ax.legend()
    fig.tight_layout()
    return save(fig, name)
