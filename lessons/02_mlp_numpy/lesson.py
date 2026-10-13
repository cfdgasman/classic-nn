"""Lesson 02 - A multilayer perceptron from scratch in NumPy (forward, backprop, grad check).

Questions this lesson answers
-----------------------------
* How does a hidden layer let a network solve problems a single neuron cannot?
* What exactly is backpropagation? (Local chain rule per layer, applied in reverse.)
* How do I *prove* my backprop is right? (Central finite differences, rel. error < 1e-6.)
* What goes wrong in a gradient check? (Kinks of ReLU; stale gradients - see README.)

Run:  python lessons/02_mlp_numpy/lesson.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from nnaz.common import banner, decision_grid, save_gif, save_results, savefig, seed_everything, setup_matplotlib
from nnaz.data import spirals, xor
from nnaz.mlp_numpy import MLP, gradient_check, input_gradient_check, train

LESSON = 2
plt = setup_matplotlib()


def gradcheck_part(rng):
    banner("1. Gradient check: backprop vs central finite differences (float64)")
    X, y = spirals(30, rng)
    rows = {}
    for name, kw in [("tanh", dict(act="tanh")), ("relu", dict(act="relu")),
                     ("sigmoid", dict(act="sigmoid")),
                     ("tanh + batchnorm + dropout", dict(act="tanh", batchnorm=True, dropout=0.3))]:
        net = MLP([2, 8, 8, 3], np.random.default_rng(3), **kw)
        worst, _ = gradient_check(net, X, y, eps=1e-6, weight_decay=1e-3)
        dx = input_gradient_check(net, X.copy(), y)
        rows[name] = (worst, dx)
        print(f"  {name:28s} max rel. err (params) {worst:.1e}   (inputs) {dx:.1e}"
              f"   {'PASS' if max(worst, dx) < 1e-6 else 'FAIL'}")

    # How the error depends on eps: truncation O(eps^2) vs round-off O(1e-16/eps)
    net = MLP([2, 8, 8, 3], np.random.default_rng(3), act="tanh")
    eps_list = np.logspace(-10, -1, 19)
    errs = [gradient_check(net, X, y, eps=e)[0] for e in eps_list]
    net_r = MLP([2, 8, 8, 3], np.random.default_rng(3), act="relu")
    errs_r = [gradient_check(net_r, X, y, eps=e)[0] for e in eps_list]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.loglog(eps_list, errs, "o-", label="tanh (smooth)")
    ax.loglog(eps_list, errs_r, "s--", label="ReLU (kinks)")
    ax.axhline(1e-6, color="k", lw=0.8, ls=":")
    ax.text(eps_list[0], 1.5e-6, "pass threshold 1e-6", fontsize=8)
    ax.set_xlabel(r"finite-difference step $\epsilon$"); ax.set_ylabel("relative error")
    ax.set_title("Gradient check: round-off (left) vs truncation (right)")
    ax.legend()
    savefig(fig, LESSON, "gradcheck_eps.png")
    return {"gradcheck": {k: {"params": v[0], "inputs": v[1]} for k, v in rows.items()},
            "gradcheck_eps": {"eps": eps_list.tolist(), "tanh": errs, "relu": errs_r}}


def spiral_part(rng, quick):
    banner("2. Training an MLP on the three-arm spiral")
    X, y = spirals(600, rng)
    Xte, yte = spirals(1500, rng)
    epochs = 60 if quick else 300
    res = {}

    # Baseline 1: always predict the most common class
    res["majority_test_acc"] = float(np.bincount(yte).max() / len(yte))
    # Baseline 2: softmax regression = an MLP with no hidden layer (linear boundaries)
    lin = MLP([2, 3], np.random.default_rng(0))
    train(lin, X, y, epochs=epochs, lr=1e-2, rng=rng)
    res["linear_test_acc"] = float((lin.predict(Xte) == yte).mean())

    t0 = time.time()
    net = MLP([2, 64, 64, 3], np.random.default_rng(0), act="relu")
    hist = train(net, X, y, Xte, yte, epochs=epochs, lr=1e-2, rng=rng, record_every=5)
    res["mlp_train_time_s"] = time.time() - t0
    res["mlp_train_acc"] = hist["train_acc"][-1]
    res["mlp_test_acc"] = hist["val_acc"][-1]
    res["loss_first"], res["loss_last"] = hist["loss"][0], hist["loss"][-1]
    print(f"  test accuracy: majority {res['majority_test_acc']:.3f} | softmax regression "
          f"{res['linear_test_acc']:.3f} | MLP 2-64-64-3 {res['mlp_test_acc']:.3f}"
          f"  ({res['mlp_train_time_s']:.1f}s)")

    gx, gy, G = decision_grid(X)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, model, name in [(axs[0], lin, "softmax regression"), (axs[1], net, "MLP 2-64-64-3")]:
        ax.contourf(gx, gy, model.predict(G).reshape(gx.shape), levels=[-.5, .5, 1.5, 2.5],
                    cmap="Pastel1", alpha=0.9)
        ax.scatter(*X.T, c=y, cmap="Set1", s=6, vmin=0, vmax=8)
        ax.set_title(f"{name}: test acc {(model.predict(Xte) == yte).mean():.3f}")
        ax.set_aspect("equal")
    axs[2].semilogy(hist["loss"], label="train loss")
    axs[2].semilogy(hist["val_loss"], label="test loss")
    axs[2].set_xlabel("epoch"); axs[2].legend(); axs[2].set_title("cross-entropy")
    savefig(fig, LESSON, "spiral_mlp.png")

    # GIF: decision regions while training (replaying stored parameter snapshots)
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(4, 4))
    params = net.param_list()

    def draw(k):
        ax.clear()
        for (layer, name), val in zip(params, hist["snapshots"][k]):
            layer.params[name] = val
        ax.contourf(gx, gy, net.predict(G).reshape(gx.shape), levels=[-.5, .5, 1.5, 2.5],
                    cmap="Pastel1")
        ax.scatter(*X.T, c=y, cmap="Set1", s=4, vmin=0, vmax=8)
        ax.set_title(f"epoch {5 * k}"); ax.set_xticks([]); ax.set_yticks([])
        return []

    save_gif(FuncAnimation(fig, draw, frames=len(hist["snapshots"])), LESSON, "spiral_training.gif", fps=8)

    # The hidden layer solves XOR (lesson 01's failure case) with only 4 hidden units
    Xx, yx = xor(400, rng)
    small = MLP([2, 4, 2], np.random.default_rng(0), act="tanh", init="xavier")
    train(small, Xx, yx, epochs=200, lr=3e-2, rng=rng)
    res["xor_acc_4_hidden"] = float((small.predict(Xx) == yx).mean())
    print(f"  XOR with one hidden layer of 4 tanh units: accuracy {res['xor_acc_4_hidden']:.3f}")
    return res


def main(quick: bool = False) -> dict:
    rng = seed_everything(2)
    res = gradcheck_part(rng)
    res.update(spiral_part(rng, quick))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
