"""Lesson 01 - The single neuron: perceptron and logistic regression, by hand.

Questions this lesson answers
-----------------------------
* What does one artificial neuron compute, and what can it (not) represent?
* How does the perceptron learn, and why is it guaranteed to converge on
  separable data (Novikoff's bound)?
* How do we derive the gradient of a loss by hand, and how do we *check* it?
* Does gradient descent find the true optimum? (We compare with SciPy L-BFGS.)

Run:  python lessons/01_perceptron/lesson.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from nnaz.common import banner, decision_grid, save_gif, save_results, savefig, seed_everything, setup_matplotlib
from nnaz.data import blobs, linearly_separable, xor
from nnaz.neuron import (bce_grad, bce_loss, logistic_gd, logistic_reference, novikoff_bound,
                         numerical_grad, perceptron_train, predict, rel_error, sigmoid)

LESSON = 1
plt = setup_matplotlib()


def perceptron_part(rng):
    banner("1. Rosenblatt's perceptron on linearly separable data")
    X, y = linearly_separable(200, rng)
    w, b, mistakes, hist = perceptron_train(X, y, rng=rng, record=True)
    acc = (predict(np.r_[w, b], X) == y).mean()
    # The data were generated from the line u.x - 0.5 = 0, which therefore is *a*
    # separating hyperplane; its margin gives an upper bound on the mistakes.
    R, gamma, bound = novikoff_bound(X, y, np.array([1.0, 2.0]) / np.sqrt(5), -0.5)
    print(f"  mistakes before convergence: {mistakes}  (Novikoff bound (R/gamma)^2 = {bound:.0f},"
          f" R={R:.2f}, gamma={gamma:.3f})")
    print(f"  training accuracy: {acc:.3f}")

    # GIF: the boundary jumps every time the perceptron makes a mistake.
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(*X[y == 0].T, s=10, c="tab:blue")
    ax.scatter(*X[y == 1].T, s=10, c="tab:orange")
    lo, hi = X.min(0) - 0.3, X.max(0) + 0.3
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    line, = ax.plot([], [], "k-", lw=2)
    mark, = ax.plot([], [], "ro", ms=10, mfc="none", mew=2)
    title = ax.set_title("")
    xs = np.linspace(lo[0], hi[0], 2)

    def draw(k):
        wk, bk, i = hist[k]
        # w1 x + w2 y + b = 0  ->  y = -(w1 x + b) / w2
        line.set_data(xs, -(wk[0] * xs + bk) / (wk[1] if abs(wk[1]) > 1e-12 else 1e-12))
        mark.set_data([X[i, 0]], [X[i, 1]])
        title.set_text(f"perceptron update {k + 1}/{len(hist)}")
        return line, mark, title

    anim = FuncAnimation(fig, draw, frames=len(hist), blit=False)
    save_gif(anim, LESSON, "perceptron.gif", fps=2)
    return dict(perceptron_mistakes=mistakes, novikoff_bound=bound, novikoff_R=R,
                novikoff_gamma=gamma, perceptron_train_acc=acc)


def logistic_part(rng):
    banner("2. A sigmoid neuron (logistic regression) with a hand-derived gradient")
    X, y = blobs(400, rng)
    Xte, yte = blobs(2000, rng)
    lam = 1e-2

    # --- gradient check: analytic vs central finite differences
    p = rng.normal(size=3)
    g_hand = bce_grad(p, X, y, lam)
    g_num = numerical_grad(lambda q: bce_loss(q, X, y, lam), p)
    err = rel_error(g_hand, g_num)
    print(f"  gradient check  hand={g_hand}  numeric={g_num}  max rel err={err:.2e}")

    # --- gradient descent vs the exact optimum
    p_ref, f_ref = logistic_reference(X, y, lam)
    p_gd, losses, snaps = logistic_gd(X, y, lr=1.0, steps=400, lam=lam, record_every=4)
    acc_tr = (predict(p_gd, X) == y).mean()
    acc_te = (predict(p_gd, Xte) == yte).mean()
    majority = max(yte.mean(), 1 - yte.mean())
    print(f"  GD after 400 steps: |p - p*|_inf = {np.abs(p_gd - p_ref).max():.2e},"
          f"  L - L* = {losses[-1] - f_ref:.2e}")
    print(f"  accuracy train {acc_tr:.3f}  test {acc_te:.3f}  (majority-class baseline {majority:.3f})")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    gx, gy, G = decision_grid(np.r_[X, Xte])
    prob = sigmoid(G @ p_gd[:2] + p_gd[2]).reshape(gx.shape)
    cs = axs[0].contourf(gx, gy, prob, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.6)
    axs[0].contour(gx, gy, prob, levels=[0.5], colors="k")
    axs[0].scatter(*X.T, c=y, cmap="RdBu_r", s=10, edgecolor="k", lw=0.3)
    fig.colorbar(cs, ax=axs[0], label=r"$\sigma(w\cdot x+b)$")
    axs[0].set_title(f"sigmoid neuron, test acc {acc_te:.3f}")
    axs[1].semilogy(losses - f_ref + 1e-17)
    axs[1].set_xlabel("gradient-descent step"); axs[1].set_ylabel(r"$L - L^*$ (L-BFGS optimum)")
    axs[1].set_title("GD converges linearly to the exact optimum")
    savefig(fig, LESSON, "logistic_neuron.png")

    # GIF of the probability field during training
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(*X.T, c=y, cmap="RdBu_r", s=8, edgecolor="k", lw=0.2)
    ttl = ax.set_title("")
    art = []

    def draw(k):
        for a in art:
            a.remove()
        art.clear()
        pk = snaps[k]
        pr = sigmoid(G @ pk[:2] + pk[2]).reshape(gx.shape)
        art.append(ax.contourf(gx, gy, pr, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.5))
        ttl.set_text(f"GD step {4 * k}")
        return []

    frames = [k for k in range(len(snaps)) if k < 20 or k % 5 == 0]
    save_gif(FuncAnimation(fig, draw, frames=frames), LESSON, "logistic_training.gif", fps=8)
    return dict(grad_check_rel_err=err, gd_param_err=float(np.abs(p_gd - p_ref).max()),
                gd_loss_gap=float(losses[-1] - f_ref), logistic_train_acc=acc_tr,
                logistic_test_acc=acc_te, majority_baseline=majority)


def xor_part(rng):
    banner("3. The limit of a single neuron: XOR")
    X, y = xor(400, rng)
    p, losses, _ = logistic_gd(X, y, lr=1.0, steps=2000)
    acc = (predict(p, X) == y).mean()
    print(f"  best linear neuron on XOR: accuracy {acc:.3f}  -> a hidden layer is needed (lesson 02)")
    fig, ax = plt.subplots(figsize=(4, 4))
    gx, gy, G = decision_grid(X)
    ax.contourf(gx, gy, sigmoid(G @ p[:2] + p[2]).reshape(gx.shape), levels=np.linspace(0, 1, 11),
                cmap="RdBu_r", alpha=0.5)
    ax.scatter(*X.T, c=y, cmap="RdBu_r", s=10, edgecolor="k", lw=0.3)
    ax.set_title(f"XOR: one neuron gets {acc:.0%}")
    savefig(fig, LESSON, "xor_failure.png")
    return dict(xor_acc=acc)


def main(quick: bool = False) -> dict:
    rng = seed_everything(1)
    res = {}
    res.update(perceptron_part(rng))
    res.update(logistic_part(rng))
    res.update(xor_part(rng))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
