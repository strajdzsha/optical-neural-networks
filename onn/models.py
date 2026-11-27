"""Network architectures used in the thesis: 784 -> hidden -> 10 MLPs."""
from __future__ import annotations

import torch
import torch.nn as nn

from .activations import LearnableLookupActivation


class MLP(nn.Module):
    """784 -> hidden -> act -> 10, with no activation on the output layer (Sec. 3)."""

    def __init__(self, activation: nn.Module, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.act = activation
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x.view(-1, 28 * 28))))


class LearnableMLP(nn.Module):
    """MLP whose hidden neurons each learn which lookup-table curve to use (Sec. 2.2).

    ``forward`` returns ``(logits, p)``, where ``p`` is the per-neuron softmax over curves.
    """

    def __init__(self, table: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.act = LearnableLookupActivation(hidden, table)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x, hard: bool = False):
        h, p = self.act(self.fc1(x.view(-1, 28 * 28)), hard=hard)
        return self.fc2(h), p


class ComplexLinear(nn.Module):
    """Complex dense layer built from two real layers (same as ``complexPyTorch``).

    (W_r + iW_i)(x_r + ix_i) = (W_r x_r - W_i x_i) + i(W_r x_i + W_i x_r)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, z):
        return torch.complex(
            self.fc_r(z.real) - self.fc_i(z.imag),
            self.fc_r(z.imag) + self.fc_i(z.real),
        )


class ComplexMLP(nn.Module):
    """Complex 784 -> hidden -> act -> 10. The logits are |z| (thesis Appendix B)."""

    def __init__(self, activation, hidden: int = 64):
        super().__init__()
        self.fc1 = ComplexLinear(28 * 28, hidden)
        self.act = activation
        self.fc2 = ComplexLinear(hidden, 10)

    def forward(self, x):
        z = x.view(-1, 28 * 28)
        if not z.is_complex():
            z = torch.complex(z, torch.zeros_like(z))
        return self.fc2(self.act(self.fc1(z))).abs()
