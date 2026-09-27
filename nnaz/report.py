"""Build the README results tables from ``docs/results/lessonNN.json``.

The README contains marker pairs  <!-- results:NN -->  ...  <!-- /results:NN -->;
``update_readme()`` replaces what is between them with tables generated from the
latest real run, so the numbers in the README can never drift from the code.
"""
from __future__ import annotations

import re

from .common import DOCS, ROOT, load_results, md_table


def _pct(x):
    return f"{100 * x:.1f}%"


def _sci(x):
    return f"{x:.1e}"


def table_01(r):
    return md_table(["quantity", "value"], [
        ["perceptron mistakes before convergence", r["perceptron_mistakes"]],
        ["Novikoff bound (R/γ)² with R, γ", f"{r['novikoff_bound']:.0f}  (R = {r['novikoff_R']:.2f}, γ = {r['novikoff_gamma']:.3f})"],
        ["hand gradient vs finite differences (rel. error)", _sci(r["grad_check_rel_err"])],
        ["GD vs L-BFGS optimum: max parameter difference", _sci(r["gd_param_err"])],
        ["logistic neuron test accuracy", _pct(r["logistic_test_acc"])],
        ["majority-class baseline", _pct(r["majority_baseline"])],
        ["single neuron on XOR (train accuracy)", _pct(r["xor_acc"])],
    ])


def table_02(r):
    g = r["gradcheck"]
    t1 = md_table(["network 2-8-8-3", "max rel. error, parameters", "max rel. error, inputs", "< 1e-6 ?"],
                  [[k, _sci(v["params"]), _sci(v["inputs"]), "PASS" if max(v.values()) < 1e-6 else "FAIL"]
                   for k, v in g.items()])
    t2 = md_table(["model (spirals, 3 classes)", "test accuracy"], [
        ["majority class", _pct(r["majority_test_acc"])],
        ["softmax regression (no hidden layer)", _pct(r["linear_test_acc"])],
        ["MLP 2-64-64-3, ReLU, Adam", _pct(r["mlp_test_acc"])],
        ["XOR with 4 hidden tanh units (train)", _pct(r["xor_acc_4_hidden"])],
    ])
    return t1 + "\n\n" + t2


def table_03(r):
    out = []
    out.append("**Activations** (6 hidden layers, Adam 3e-3, 150 epochs)\n\n" + md_table(
        ["activation", "final train loss", "train accuracy"],
        [[k, f"{v['final_loss']:.2e}", _pct(v["train_acc"])] for k, v in r["activations"].items()]))
    o = r["optimisers"]
    out.append("**Optimisers**\n\n" + md_table(
        ["run", "final value"],
        [[f"Rosenbrock, {k}", _sci(v)] if not k.startswith("mlp") else [f"spiral MLP loss, {k[4:]}", f"{v:.4f}"]
         for k, v in o.items()]))
    out.append("**LR schedules** (SGD + momentum, 100 epochs, mean of 3 seeds)\n\n" + md_table(
        ["schedule", "final train loss", "test accuracy"],
        [[k, f"{v['final_train_loss']:.4f}", _pct(v["test_acc"])] for k, v in r["schedules"].items()]))
    out.append("**Batch norm** (10 sigmoid hidden layers, SGD + momentum)\n\n" + md_table(
        ["network", "final train loss", "train accuracy"],
        [[k, f"{v['final_loss']:.3f}", _pct(v["train_acc"])] for k, v in r["batchnorm"].items()]))
    out.append("**Overfitting** (150 points, 20 % label noise, 2-256-256-3, 400 epochs)\n\n" + md_table(
        ["regularisation", "train acc", "val acc (final)", "val loss (final)", "early stopping: epoch / val acc"],
        [[k, _pct(v["train_acc"]), _pct(v["val_acc_final"]), f"{v['val_loss_final']:.3f}",
          f"{v['best_epoch']} / {_pct(v['val_acc_early_stop'])}"] for k, v in r["overfitting"].items()]))
    return "\n\n".join(out)


TABLES = {1: table_01, 2: table_02, 3: table_03}


def update_readme(path=ROOT / "README.md") -> None:
    text = path.read_text()
    for n, fn in TABLES.items():
        try:
            table = fn(load_results(n))
        except FileNotFoundError:
            continue
        pat = re.compile(rf"(<!-- results:{n:02d} -->)(.*?)(<!-- /results:{n:02d} -->)", re.S)
        text = pat.sub(lambda m: f"{m.group(1)}\n{table}\n{m.group(3)}", text)
    path.write_text(text)
    (DOCS / "results").mkdir(exist_ok=True)
