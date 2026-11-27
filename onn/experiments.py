"""All thesis experiments, cached as JSON in ``results/``.

Command line::

    python -m onn.experiments                 # run everything that is not cached yet
    python -m onn.experiments fig3_3_mnist    # run one experiment
    python -m onn.experiments all --force     # recompute everything
    python -m onn.experiments --list

From Python / the notebook::

    from onn.experiments import run
    res = run("tab3_1_fashion_learnable")     # loads the cache if it exists
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .activations import (ComplexLookupActivation, FixedTableActivation, LookupActivation,
                          complex_relu)
from .data import (COMPLEX_DETUNINGS, ROOT, load_2d_lookup, load_complex_activations,
                   load_dataset, load_measured_activations)
from .models import MLP, ComplexMLP, LearnableMLP
from .training import train

RESULTS = ROOT / "results"
SEEDS = (0, 1, 2)

# Training protocol from Sec. 3: Adam, lr = 1e-3, batch 64, cross-entropy.
EPOCHS_MEASURED = 5   # 12 measured curves (Sec. 3.1). The thesis states 5 epochs.
EPOCHS_2D = 10        # 2D-lookup experiments (Sec. 3.2). The original code used 10.
EPOCHS_COMPLEX = 5
EPOCHS_LOSS_CURVE = 20  # Fig. 3.5


def _seeded(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _summary(runs, key="test_acc"):
    a = np.array([r[key] for r in runs])
    return {"mean": float(a.mean()), "std": float(a.std()), "values": a.tolist()}


# --------------------------------------------------------------------------- #
# Sec. 3.1  Measured lookup activations (Fig. 3.1, 3.2)
# --------------------------------------------------------------------------- #
def fixed_measured(dataset: str, hidden: int, seeds=SEEDS) -> dict:
    """Train one network per measured FP-LD curve (12 curves: 3 modes x 4 detunings)."""
    tr, te = load_dataset(dataset)
    out = []
    for i, a in enumerate(load_measured_activations()):
        runs = []
        for s in seeds:
            _seeded(s)
            model = MLP(LookupActivation(a["x"], a["y"]), hidden=hidden)
            runs.append(train(model, tr, te, epochs=EPOCHS_MEASURED, seed=s))
        out.append({"index": i, "mode": a["mode"], "detuning": a["detuning"], **_summary(runs)})
        print(f"[{dataset} h={hidden}] {a['mode']} {a['detuning']}Ω: {out[-1]['mean']:.2f}%", flush=True)
    return {"dataset": dataset, "hidden": hidden, "epochs": EPOCHS_MEASURED, "seeds": list(seeds), "functions": out}


# --------------------------------------------------------------------------- #
# Sec. 3.2  Each row of the 2D lookup table as a fixed activation (Fig. 3.3)
# --------------------------------------------------------------------------- #
def fixed_2d(dataset: str, seeds=SEEDS) -> dict:
    lut = load_2d_lookup()
    table = torch.tensor(lut["table"], dtype=torch.float32)
    tr, te = load_dataset(dataset)
    out = []
    for i, det in enumerate(lut["detuning"]):
        runs = []
        for s in seeds:
            _seeded(s)
            runs.append(train(MLP(FixedTableActivation(table, i)), tr, te, epochs=EPOCHS_2D, seed=s))
        out.append({"index": i, "detuning": float(det), **_summary(runs)})
        print(f"[{dataset} 2D] #{i} ({det:.0f}Ω): {out[-1]['mean']:.2f}%", flush=True)
    return {"dataset": dataset, "epochs": EPOCHS_2D, "seeds": list(seeds), "functions": out}


# --------------------------------------------------------------------------- #
# Sec. 3.2  Learnable activations with entropy regularisation (Table 3.1, Fig. 3.4)
# --------------------------------------------------------------------------- #
def learnable(dataset: str, lambdas=(0.0, 0.1, 0.15, 0.2), seeds=SEEDS) -> dict:
    lut = load_2d_lookup()
    table = torch.tensor(lut["table"], dtype=torch.float32)
    tr, te = load_dataset(dataset)
    out = []
    for lmb in lambdas:
        runs, probs = [], None
        for s in seeds:
            _seeded(s)
            model = LearnableMLP(table)
            runs.append(train(model, tr, te, epochs=EPOCHS_2D, entropy_lambda=lmb, seed=s))
            if probs is None:  # keep the softmax distributions of the first seed for Fig. 3.4
                p = model.act.probs.detach()
                probs = p.tolist()
                chosen = p.argmax(1).tolist()
        out.append({
            "lambda": lmb,
            "test_acc_hard": _summary(runs, "test_acc"),       # argmax curve (hardware)
            "test_acc_soft": _summary(runs, "test_acc_soft"),  # softmax mixture (as trained)
            "probs_seed0": probs, "chosen_seed0": chosen,
        })
        print(f"[{dataset} learnable] λ={lmb}: soft {out[-1]['test_acc_soft']['mean']:.2f}%"
              f"  hard {out[-1]['test_acc_hard']['mean']:.2f}%", flush=True)
    return {"dataset": dataset, "epochs": EPOCHS_2D, "seeds": list(seeds),
            "detuning": lut["detuning"].tolist(), "lambdas": out}


# --------------------------------------------------------------------------- #
# Sec. 3.3  Complex networks (Table 3.2, Fig. 3.5)
# --------------------------------------------------------------------------- #
def _complex_act(c, detuning):
    if detuning == "relu":
        return complex_relu
    return ComplexLookupActivation(c["p_in"], c["power"][detuning], c["phase"][detuning])


def complex_nets(dataset: str, seeds=SEEDS) -> dict:
    """Complex MLP with each simulated complex FP-LD activation and with complex ReLU."""
    c = load_complex_activations()
    tr, te = load_dataset(dataset)
    out = []
    for d in COMPLEX_DETUNINGS + ["relu"]:
        runs = []
        for s in seeds:
            _seeded(s)
            runs.append(train(ComplexMLP(_complex_act(c, d)), tr, te, epochs=EPOCHS_COMPLEX, seed=s))
        out.append({"activation": d, **_summary(runs)})
        print(f"[{dataset} complex] {d}: {out[-1]['mean']:.2f}%", flush=True)
    return {"dataset": dataset, "epochs": EPOCHS_COMPLEX, "seeds": list(seeds), "activations": out}


def loss_curves(seeds=SEEDS, detuning: int = 10) -> dict:
    """Per-epoch test loss over 20 epochs: real vs complex networks on MNIST (Fig. 3.5).

    Uses the setup of the original notebook: SGD (lr 0.01, momentum 0.9) and
    inputs normalised with Normalize(0.5, 1.0). The real network uses the
    power curve of the same simulated FP-LD (``detuning``) as a real lookup
    activation, and the complex networks use complex ReLU and the complex lookup.
    """
    c = load_complex_activations()
    tr, te = load_dataset("mnist", 0.5, 1.0)
    x = c["p_in"] / c["p_in"].max()
    nets = {
        "real_lookup": lambda: MLP(LookupActivation(x, c["power"][detuning])),
        "complex_relu": lambda: ComplexMLP(complex_relu),
        "complex_lookup": lambda: ComplexMLP(_complex_act(c, detuning)),
    }
    out = {}
    for name, make in nets.items():
        curves, accs = [], []
        for s in seeds:
            _seeded(s)
            r = train(make(), tr, te, epochs=EPOCHS_LOSS_CURVE, optimizer="sgd", lr=0.01,
                      seed=s, eval_every_epoch=True)
            curves.append([h["test_loss"] for h in r["history"]])
            accs.append(r["test_acc"])
        out[name] = {"test_loss": np.mean(curves, 0).tolist(), "test_loss_runs": curves,
                     "final_acc": float(np.mean(accs))}
        print(f"[loss curves] {name}: final acc {out[name]['final_acc']:.2f}%", flush=True)
    return {"epochs": EPOCHS_LOSS_CURVE, "detuning": detuning, "seeds": list(seeds), "nets": out}


# --------------------------------------------------------------------------- #
# Registry + cache
# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    "fig3_1_mnist_h25": lambda seeds: fixed_measured("mnist", 25, seeds),
    "fig3_1_mnist_h64": lambda seeds: fixed_measured("mnist", 64, seeds),
    "fig3_2_fashion_h64": lambda seeds: fixed_measured("fashion", 64, seeds),
    "fig3_3_mnist": lambda seeds: fixed_2d("mnist", seeds),
    "fig3_3_fashion": lambda seeds: fixed_2d("fashion", seeds),
    "tab3_1_fashion_learnable": lambda seeds: learnable("fashion", seeds=seeds),
    "tab3_1_mnist_learnable": lambda seeds: learnable("mnist", seeds=seeds),
    "tab3_2_complex_mnist": lambda seeds: complex_nets("mnist", seeds),
    "tab3_2_complex_fashion": lambda seeds: complex_nets("fashion", seeds),
    "fig3_5_loss_curves": lambda seeds: loss_curves(seeds),
}


def run(name: str, force: bool = False, seeds=SEEDS) -> dict:
    """Return the results of experiment ``name``, computing and caching them if needed."""
    path = RESULTS / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    torch.set_num_threads(1)  # tiny MLPs: one thread per process is fastest
    res = EXPERIMENTS[name](tuple(seeds))
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    return res


def _run_cli(args):
    name, force, seeds = args
    run(name, force, seeds)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=["all"], help="experiment names, or 'all'")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="number of seeds per configuration")
    ap.add_argument("--jobs", type=int, default=1, help="experiments to run in parallel")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(EXPERIMENTS))
        return
    names = list(EXPERIMENTS) if a.names == ["all"] else a.names
    jobs = [(n, a.force, tuple(range(a.seeds))) for n in names]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for n in ex.map(_run_cli, jobs):
                print(f"done: {n}", flush=True)
    else:
        for j in jobs:
            _run_cli(j)


if __name__ == "__main__":
    main()
