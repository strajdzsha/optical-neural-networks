"""Datasets: MNIST / Fashion-MNIST in memory, plus the measured FP-LD activation data."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"

# Ω = 1e10 rad/s, the detuning unit used in the thesis
OMEGA = 1e10


# --------------------------------------------------------------------------- #
# Image datasets
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    x: torch.Tensor  # (N, 784) float32, already normalised
    y: torch.Tensor  # (N,) int64

    def batches(self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None):
        n = len(self.y)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            yield self.x[idx], self.y[idx]


@lru_cache(maxsize=None)
def load_dataset(name: str = "mnist", mean: float = 0.5, std: float = 0.5) -> tuple[Split, Split]:
    """Return (train, test) for ``"mnist"`` or ``"fashion"``.

    Equivalent to ``ToTensor()`` followed by ``Normalize((mean,), (std,))``, but the
    whole dataset is kept as one tensor, which makes training on CPU much faster.
    Downloads to ``data/`` on first use.
    """
    import torchvision

    cls = {"mnist": torchvision.datasets.MNIST, "fashion": torchvision.datasets.FashionMNIST}[name]
    out = []
    for train in (True, False):
        ds = cls(str(DATA), train=train, download=True)
        x = (ds.data.float().view(len(ds.data), -1) / 255.0 - mean) / std
        out.append(Split(x.contiguous(), ds.targets.clone().long()))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Measured / simulated FP-LD activation data
# --------------------------------------------------------------------------- #
MODES = ["m2", "m4", "m6"]
DETUNINGS_1D = [9.3, 10.9, 12.5, 15.0]  # in Ω, Fig. 3.1 / 3.2 axis labels


def load_measured_activations() -> list[dict]:
    """The 12 experimentally measured FP-LD transfer curves (Fig. 2.1).

    3 longitudinal modes x 4 detunings. Returns dicts with keys
    ``x, y, mode, detuning``, ordered mode-major (index = 4*mode + detuning).
    """
    acts = []
    for mode, fname in zip(MODES, ["fpld_mode2.mat", "fpld_mode4.mat", "fpld_mode6.mat"]):
        d = loadmat(ASSETS / fname)["data"]
        for k in range(4):
            acts.append(dict(x=d[k], y=d[k + 4], mode=mode, detuning=DETUNINGS_1D[k]))
    return acts


def load_2d_lookup() -> dict:
    """Simulated 2D lookup table sigma(P_in, detuning) (Fig. 2.2).

    Returns ``p_in`` (M,), ``detuning`` (N,) in Ω, and ``table`` (N, M) with
    output power, one row per detuning.
    """
    m = loadmat(ASSETS / "fpld_2d_lookup.mat")
    return dict(
        p_in=m["input_power"].ravel(),
        detuning=m["set_detuning"].ravel() / OMEGA,
        table=m["peakout_data"].T,  # stored as (M powers, N detunings) -> (N, M)
    )


COMPLEX_DETUNINGS = [10, 15, 20, 25, 30]  # labelled  -10 Ω ... -30 Ω in Fig. 2.3


def load_complex_activations() -> dict:
    """Simulated power and phase transfer curves for the complex activation (Fig. 2.3)."""
    m = loadmat(ASSETS / "fpld_complex.mat")
    return dict(
        p_in=m["power_in"].ravel(),
        power={d: m[f"power_out{d}"].ravel() for d in COMPLEX_DETUNINGS},
        phase={d: m[f"phase_out{d}"].ravel() for d in COMPLEX_DETUNINGS},
    )
