"""Training and evaluation loops shared by all experiments."""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import entropy
from .data import Split
from .models import LearnableMLP


def _logits(model, x, hard=False):
    if isinstance(model, LearnableMLP):
        return model(x, hard=hard)[0]
    return model(x)


@torch.no_grad()
def evaluate(model: nn.Module, split: Split, hard: bool = True, batch_size: int = 1000) -> dict:
    """Accuracy (%) and mean cross-entropy on ``split``.

    ``hard`` only matters for :class:`LearnableMLP`. With ``hard=True`` each neuron
    uses its single most probable curve (test-time / hardware behaviour).
    """
    model.eval()
    correct, loss, n = 0, 0.0, 0
    for x, y in split.batches(batch_size):
        out = _logits(model, x, hard)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return {"acc": 100.0 * correct / n, "loss": loss / n}


def train(
    model: nn.Module,
    train_split: Split,
    test_split: Split,
    epochs: int = 5,
    lr: float = 1e-3,
    optimizer: str = "adam",
    batch_size: int = 64,
    entropy_lambda: float = 0.0,
    seed: int = 0,
    eval_every_epoch: bool = False,
    verbose: bool = False,
) -> dict:
    """Train ``model`` with cross-entropy (+ λ·entropy for learnable activations).

    Returns ``{"test_acc", "test_loss", "history": [...]}``. For a ``LearnableMLP``
    it also returns ``test_acc_soft``, the accuracy with the softmax-mixed
    activations used during training (the "training" column of Table 3.1).
    """
    g = torch.Generator().manual_seed(seed)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(optimizer)

    learnable = isinstance(model, LearnableMLP)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_split.batches(batch_size, shuffle=True, generator=g):
            opt.zero_grad()
            if learnable:
                out, p = model(x)
                loss = F.cross_entropy(out, y) + entropy_lambda * entropy(p)
            else:
                loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        rec = {"epoch": epoch + 1, "train_loss": running / len(train_split.y)}
        if eval_every_epoch or epoch == epochs - 1:
            rec.update({f"test_{k}": v for k, v in evaluate(model, test_split).items()})
        history.append(rec)
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs}  " + "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"))

    out = {"test_acc": history[-1]["test_acc"], "test_loss": history[-1]["test_loss"],
           "history": history, "seconds": time.time() - t0}
    if learnable:
        out["test_acc_soft"] = evaluate(model, test_split, hard=False)["acc"]
    return out
