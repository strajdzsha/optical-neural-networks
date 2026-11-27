"""All-optical (FP-LD) activation functions defined by lookup tables.

Three families are implemented, one per approach in the thesis:

* ``LookupActivation``          - a single measured transfer curve (Sec. 2.1)
* ``LearnableLookupActivation`` - per-neuron softmax mixture over a 2D lookup
                                  table, trained with an entropy penalty (Sec. 2.2)
* ``ComplexLookupActivation``   - power and phase curves applied to |z| and arg z
                                  of a complex pre-activation (Sec. 2.3)

The interpolation helpers extrapolate linearly outside the tabulated range
(``searchsorted`` indices are clamped to the first/last segment). The original
code did this too, and it matters, because pre-activations are often negative
or larger than the tabulated input power.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Differentiable piecewise-linear interpolation
# --------------------------------------------------------------------------- #
def interp1d(X: torch.Tensor, Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear y = f(x) from the table (X, Y). X must be sorted ascending."""
    idx = torch.searchsorted(X, x.contiguous()).clamp(1, len(X) - 1)
    x0, x1 = X[idx - 1], X[idx]
    y0, y1 = Y[idx - 1], Y[idx]
    return y0 + (y1 - y0) / (x1 - x0) * (x - x0)


def interp1d_per_neuron(X: torch.Tensor, Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Like :func:`interp1d`, but every neuron j has its own curve ``Y[j]``.

    X: (M,) shared abscissa, Y: (N, M) one curve per neuron, x: (B, N).
    """
    idx = torch.searchsorted(X, x.contiguous()).clamp(1, len(X) - 1)
    rows = torch.arange(Y.shape[0], device=x.device).unsqueeze(0)
    x0, x1 = X[idx - 1], X[idx]
    y0, y1 = Y[rows, idx - 1], Y[rows, idx]
    return y0 + (y1 - y0) / (x1 - x0) * (x - x0)


# --------------------------------------------------------------------------- #
# 1) Fixed activation from a lookup table
# --------------------------------------------------------------------------- #
class LookupActivation(nn.Module):
    """Activation given by one tabulated FP-LD transfer curve (thesis Appendix A).

    The output is normalised to ``max |y| = 1``. The input axis is left in the
    units of the measurement.
    """

    def __init__(self, X, Y):
        super().__init__()
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
        order = torch.argsort(X)  # searchsorted needs a sorted abscissa
        self.register_buffer("X", X[order])
        self.register_buffer("Y", Y[order] / Y.abs().max())

    def forward(self, x):
        return interp1d(self.X, self.Y, x)


# --------------------------------------------------------------------------- #
# 2) Learnable selection from a 2D lookup table  sigma(x, detuning)
# --------------------------------------------------------------------------- #
class LearnableLookupActivation(nn.Module):
    """Per-neuron activation  sigma_i = softmax(omega_i) @ Sigma   (Eq. 2.2).

    ``table`` has shape (n_functions, n_points) and holds one transfer curve per
    detuning. Its input axis is taken to be ``linspace(0, 1, n_points)``.

    * training (``hard=False``): a soft, differentiable mixture of curves
    * testing  (``hard=True``):  each neuron uses only its argmax curve, the one
      that can actually be set in hardware

    ``forward`` returns ``(y, p)``, where ``p = softmax(omega)`` has shape
    (n_neurons, n_functions) and feeds the entropy regulariser (Eq. 2.4).
    """

    def __init__(self, n_neurons: int, table: torch.Tensor, init_scale: float = 0.1):
        super().__init__()
        table = torch.as_tensor(table, dtype=torch.float32)
        self.register_buffer("table", table / table.max())
        self.register_buffer("X", torch.linspace(0, 1, table.shape[1]))
        self.omega = nn.Parameter(init_scale * torch.randn(n_neurons, table.shape[0]))

    @property
    def probs(self) -> torch.Tensor:
        return F.softmax(self.omega, dim=1)

    def forward(self, x, hard: bool = False):
        p = self.probs
        curves = self.table[self.omega.argmax(dim=1)] if hard else p @ self.table
        return interp1d_per_neuron(self.X, curves, x), p


class FixedTableActivation(nn.Module):
    """One row of the 2D lookup table used as a fixed activation (Fig. 3.3)."""

    def __init__(self, table: torch.Tensor, idx: int):
        super().__init__()
        table = torch.as_tensor(table, dtype=torch.float32)
        table = table / table.max()
        self.register_buffer("Y", table[idx])
        self.register_buffer("X", torch.linspace(0, 1, table.shape[1]))

    def forward(self, x):
        return interp1d(self.X, self.Y, x)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean (over neurons) Shannon entropy of the rows of ``p``."""
    return -(p * torch.log(p + eps)).sum(dim=1).mean()


# --------------------------------------------------------------------------- #
# 3) Complex activations
# --------------------------------------------------------------------------- #
class ComplexLookupActivation(nn.Module):
    """Complex FP-LD activation: |z| -> P_out(|z|), arg z -> phi_out(arg z).

    Both curves are interpolated independently over the same (normalised)
    input-power axis, as described in Sec. 2.3 and as the original code did.
    """

    def __init__(self, X, R, Phi):
        super().__init__()
        X = np.asarray(X, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        self.register_buffer("X", torch.tensor(X / X.max(), dtype=torch.float32))
        self.register_buffer("R", torch.tensor(R / R.max(), dtype=torch.float32))
        self.register_buffer("Phi", torch.tensor(np.asarray(Phi), dtype=torch.float32))

    def forward(self, z):
        r = interp1d(self.X, self.R, z.abs())
        phi = interp1d(self.X, self.Phi, torch.angle(z))
        return torch.complex(r * torch.cos(phi), r * torch.sin(phi))


def complex_relu(z: torch.Tensor) -> torch.Tensor:
    """ReLU applied separately to the real and imaginary parts (thesis Appendix B)."""
    return torch.complex(F.relu(z.real), F.relu(z.imag))
