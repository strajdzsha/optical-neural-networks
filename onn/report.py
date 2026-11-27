"""Tables 3.1 / 3.2: reproduced numbers next to the values reported in the thesis."""
from __future__ import annotations

import numpy as np

# Values as printed in the thesis (rad/strahinja-diplomski.pdf).
PAPER = {
    "table3_1": {  # Fashion-MNIST, λ -> (train/soft, test/hard)
        0.0: (86.47, 85.79), 0.1: (86.22, 86.21), 0.15: (86.45, 86.46), 0.2: (86.50, 86.50),
    },
    "table3_2": {  # row -> (MNIST, Fashion-MNIST)
        "lookup, measured (mean of 12)": (94.87, 85.65),
        "lookup, simulated (mean of 26)": (95.63, 85.80),
        "learnable lookup": (96.62, 86.50),
        "complex lookup": (96.20, 85.62),
        "complex ReLU": (96.37, 86.74),
    },
    "fig3_1_spread": {"h25": 2.9, "h64": 4.6},
}


def _fmt(m, s=None):
    return f"{m:.2f}" if s is None else f"{m:.2f} ± {s:.2f}"


def table_3_1(res: dict, compare: bool = True) -> str:
    """Learnable-activation accuracies per λ. ``compare`` adds the thesis values (Fashion-MNIST only)."""
    if compare:
        rows = ["| λ | soft (train) — ours | soft — thesis | argmax (test) — ours | argmax — thesis |",
                "|---|---|---|---|---|"]
    else:
        rows = ["| λ | soft (train) | argmax (test) |", "|---|---|---|"]
    for r in res["lambdas"]:
        soft = _fmt(r["test_acc_soft"]["mean"], r["test_acc_soft"]["std"])
        hard = _fmt(r["test_acc_hard"]["mean"], r["test_acc_hard"]["std"])
        if compare:
            p = PAPER["table3_1"].get(r["lambda"], (np.nan, np.nan))
            rows.append(f"| {r['lambda']} | {soft} | {p[0]:.2f} | {hard} | {p[1]:.2f} |")
        else:
            rows.append(f"| {r['lambda']} | {soft} | {hard} |")
    return "\n".join(rows)


def table_3_2_values(R: dict, learnable_lambda: float = 0.2) -> dict:
    """Collect the reproduced Table 3.2 values from the experiment results ``R``."""
    def mean_fun(res):
        return float(np.mean([f["mean"] for f in res["functions"]]))

    def learn(res):
        r = next(r for r in res["lambdas"] if r["lambda"] == learnable_lambda)
        return r["test_acc_hard"]["mean"]

    def cplx(res, which):
        acts = {a["activation"]: a["mean"] for a in res["activations"]}
        if which == "relu":
            return acts["relu"], "ReLU"
        best = max((k for k in acts if k != "relu"), key=lambda k: acts[k])
        return acts[best], f"−{best} Ω"

    out = {
        "lookup, measured (mean of 12)": (mean_fun(R["fig3_1_mnist_h64"]), mean_fun(R["fig3_2_fashion_h64"])),
        "lookup, simulated (mean of 26)": (mean_fun(R["fig3_3_mnist"]), mean_fun(R["fig3_3_fashion"])),
        "learnable lookup": (learn(R["tab3_1_mnist_learnable"]), learn(R["tab3_1_fashion_learnable"])),
    }
    (m, md), (f, fd) = cplx(R["tab3_2_complex_mnist"], "lookup"), cplx(R["tab3_2_complex_fashion"], "lookup")
    out[f"complex lookup"] = (m, f)
    out["complex ReLU"] = (cplx(R["tab3_2_complex_mnist"], "relu")[0], cplx(R["tab3_2_complex_fashion"], "relu")[0])
    out["_complex_best_detuning"] = (md, fd)
    return out


def table_3_2(R: dict) -> str:
    v = table_3_2_values(R)
    rows = ["| network | MNIST — ours | MNIST — thesis | Fashion — ours | Fashion — thesis |",
            "|---|---|---|---|---|"]
    for k, (pm, pf) in PAPER["table3_2"].items():
        m, f = v[k]
        rows.append(f"| {k} | {m:.2f} | {pm:.2f} | {f:.2f} | {pf:.2f} |")
    md, fd = v["_complex_best_detuning"]
    return "\n".join(rows) + f"\n\nComplex lookup: best detuning {md} (MNIST) and {fd} (Fashion)."
