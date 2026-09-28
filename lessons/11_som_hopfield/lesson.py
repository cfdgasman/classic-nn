"""Lesson 11 - Self-organising maps and Hopfield networks (unsupervised learning, feedback loops).

Questions this lesson answers
-----------------------------
* Can a network organise data *without labels and without backprop*? (Kohonen's SOM.)
* What does "topology preserving" mean, and how do we measure it (quantisation and
  topographic error)? How does a SOM compare with k-means?
* What does a network with *feedback loops* compute? (Hopfield: energy descent to memories.)
* How many memories fit in a Hopfield network (theory: 0.138 N), and why is the modern
  Hopfield network the same thing as attention?

Run:  python lessons/11_som_hopfield/lesson.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from nnaz.common import banner, save_gif, save_results, savefig, seed_everything, setup_matplotlib
from nnaz.data import load_mnist
from nnaz.som_hopfield import (SOM, hebbian_weights, hopfield_recall, kmeans, modern_hopfield_recall,
                               projection_weights)

LESSON = 11
plt = setup_matplotlib()


def som_2d_part(rng):
    banner("1. A 10x10 SOM unfolding onto 2-D data (online Kohonen rule)")
    # an annulus: the map has to wrap a square grid around a hole
    r = np.sqrt(rng.uniform(0.3 ** 2, 1, 3000)); th = rng.uniform(0, 2 * np.pi, 3000)
    X = np.c_[r * np.cos(th), r * np.sin(th)]
    som = SOM(10, 10, 2, rng)
    snaps = som.train_online(X, epochs=4, rng=rng, eta0=0.5, sigma0=5.0, snapshots=300)
    qe, te = som.quantisation_error(X), som.topographic_error(X)
    print(f"  final quantisation error {qe:.3f}, topographic error {te:.3f}")
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(*X.T, s=1, c="0.75")
    lines = [ax.plot([], [], "b-", lw=0.8)[0] for _ in range(20)]
    pts, = ax.plot([], [], "r.", ms=3)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ttl = ax.set_title("")

    def draw(k):
        W = snaps[k].reshape(10, 10, 2)
        for i in range(10):
            lines[i].set_data(W[i, :, 0], W[i, :, 1])
            lines[10 + i].set_data(W[:, i, 0], W[:, i, 1])
        pts.set_data(W[..., 0].ravel(), W[..., 1].ravel())
        ttl.set_text(f"SOM after {300 * k} samples")
        return lines + [pts]

    save_gif(FuncAnimation(fig, draw, frames=len(snaps)), LESSON, "som_unfolding.gif", fps=6)
    return dict(som2d_qe=qe, som2d_te=te)


def som_mnist_part(rng, quick):
    banner("2. A 15x15 SOM of handwritten digits (batch SOM) vs k-means with 225 centres")
    x, y = load_mnist("MNIST", True)
    xt, yt = load_mnist("MNIST", False)
    n = 3000 if quick else 10000
    X = x[:n].reshape(n, -1) / 255.0
    Xt, yt = xt[:5000].reshape(5000, -1) / 255.0, yt[:5000]
    som = SOM(15, 15, 784, rng, init_data=X)
    som.train_batch(X, epochs=10 if quick else 30, sigma0=7.0, sigma_end=0.7)
    C = kmeans(X, 225, rng, iters=10 if quick else 30)

    def label_units(P):
        lab = (-2 * X @ P.T + (P ** 2).sum(1)[None]).argmin(1)
        return np.array([np.bincount(y[:n][lab == j], minlength=10).argmax() if np.any(lab == j) else -1
                         for j in range(len(P))])

    def acc(P, units):
        b = (-2 * Xt @ P.T + (P ** 2).sum(1)[None]).argmin(1)
        return float((units[b] == yt).mean())

    su, cu = label_units(som.W), label_units(C)
    qe_km = float(np.sqrt(((Xt - C[(-2 * Xt @ C.T + (C ** 2).sum(1)[None]).argmin(1)]) ** 2).sum(1)).mean())
    rand = SOM(15, 15, 784, rng)
    rand.W = X[rng.choice(n, 225, replace=False)]
    res = dict(som_qe=som.quantisation_error(Xt), som_te=som.topographic_error(Xt), kmeans_qe=qe_km,
               random_prototypes_qe=rand.quantisation_error(Xt), random_prototypes_te=rand.topographic_error(Xt),
               som_label_acc=acc(som.W, su), kmeans_label_acc=acc(C, cu),
               majority_acc=float(np.bincount(yt).max() / len(yt)))
    print(f"  test quantisation error: SOM {res['som_qe']:.3f} | k-means {res['kmeans_qe']:.3f} |"
          f" random data points as prototypes {res['random_prototypes_qe']:.3f}")
    print(f"  topographic error: SOM {res['som_te']:.3f} | random prototypes on the grid {res['random_prototypes_te']:.3f}")
    print(f"  digit accuracy by unit majority label: SOM {res['som_label_acc']:.3f} | k-means {res['kmeans_label_acc']:.3f}"
          f" | majority class {res['majority_acc']:.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
    Wimg = som.W.reshape(15, 15, 28, 28).transpose(0, 2, 1, 3).reshape(15 * 28, 15 * 28)
    axs[0].imshow(Wimg, cmap="gray_r"); axs[0].axis("off")
    axs[0].set_title("the 225 prototypes: neighbours on the grid look alike")
    U = som.u_matrix()
    im = axs[1].imshow(U, cmap="bone_r"); axs[1].grid(False)
    for j, l in enumerate(su):
        axs[1].text(j % 15, j // 15, str(l), ha="center", va="center", fontsize=7, color="C3")
    fig.colorbar(im, ax=axs[1], label="mean distance to grid neighbours")
    axs[1].set_title("U-matrix (dark ridges = cluster borders) + unit labels")
    savefig(fig, LESSON, "som_mnist.png")
    return res


def glyphs(letters="HOPFIELD", n=16):
    """Render letters to n x n +-1 bitmaps with matplotlib (no font files needed)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out, up = [], 4
    for ch in letters:
        # render at 4x resolution, then average 4x4 blocks: crisp, font-independent glyphs
        fig = Figure(figsize=(1, 1), dpi=n * up)
        FigureCanvasAgg(fig)
        fig.text(0.5, 0.5, ch, ha="center", va="center", fontsize=80, family="DejaVu Sans Mono", weight="bold")
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(float)
        a = a.reshape(n, up, n, up).mean((1, 3))
        out.append(np.where(a < 128, 1.0, -1.0).ravel())
    return np.array(out)


def hopfield_part(rng):
    banner("3. A Hopfield network as an associative memory")
    letters = "HOPFIELD"
    P = glyphs(letters)
    N = P.shape[1]
    overlap = float(np.abs(P @ P.T / N - np.eye(len(P))).max())
    rules = {"Hebbian": hebbian_weights(P), "projection (pseudo-inverse)": projection_weights(P)}
    recall = {k: [] for k in list(rules) + ["modern (attention)"]}
    probes = []
    for i in range(len(P)):
        s0 = P[i].copy()
        s0[rng.random(N) < 0.2] *= -1          # 20 % of the pixels flipped
        probes.append(s0)
        for name, W in rules.items():
            s, _, _ = hopfield_recall(W, s0, rng)
            recall[name].append(float(np.mean(s == P[i])))
        recall["modern (attention)"].append(float(np.mean(modern_hopfield_recall(P, s0) == P[i])))
    print(f"  8 letters of {N} pixels; they share most background pixels (max overlap {overlap:.2f})")
    for name, v in recall.items():
        print(f"  {name:28s} fraction of correct pixels after recall: " + " ".join(f"{x:.2f}" for x in v))
    k = letters.index("F")
    s_heb, _, _ = hopfield_recall(rules["Hebbian"], probes[k], rng)
    s, E, hist = hopfield_recall(rules["projection (pseudo-inverse)"], probes[k], rng, record=True)
    ok = bool(np.all(s == P[k]))
    fig, axs = plt.subplots(2, len(P), figsize=(12, 3.4))
    for i, a in enumerate(axs[0]):
        a.imshow(P[i].reshape(16, 16), cmap="gray_r"); a.axis("off")
    axs[0, 0].set_title("stored", loc="left", fontsize=9)
    for a, img, ttl in [(axs[1, 0], probes[k], "probe (20 % noise)"), (axs[1, 1], s_heb, "Hebbian recall"),
                        (axs[1, 2], s, "projection recall")]:
        a.imshow(img.reshape(16, 16), cmap="gray_r"); a.set_title(ttl, fontsize=8); a.axis("off")
    for a in axs[1, 3:-1]:
        a.axis("off")
    axs[1, -1].plot(E, "o-"); axs[1, -1].set_title("energy / sweep", fontsize=8)
    savefig(fig, LESSON, "hopfield_patterns.png")

    from matplotlib.animation import FuncAnimation

    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.4))
    axs[0].imshow(probes[k].reshape(16, 16), cmap="gray_r"); axs[0].set_title("probe", fontsize=9)
    im = axs[1].imshow(probes[k].reshape(16, 16), cmap="gray_r", vmin=-1, vmax=1)
    for a in axs:
        a.axis("off")
    step = max(1, len(hist) // 40)
    frames = list(range(0, len(hist), step)) + [len(hist) - 1] * 5

    def draw(j):
        im.set_data(hist[j].reshape(16, 16)); axs[1].set_title(f"update {j}", fontsize=9)
        return im,

    save_gif(FuncAnimation(fig, draw, frames=frames), LESSON, "hopfield_recall.gif", fps=8)

    # capacity: random patterns, N = 200, start from 10% corrupted versions
    N = 200
    loads = [4, 10, 16, 20, 24, 28, 32, 40, 50, 60]
    classic, modern = [], []
    for Pn in loads:
        pats = np.where(rng.random((Pn, N)) < 0.5, 1.0, -1.0)
        W = hebbian_weights(pats)
        ok_c, ok_m = [], []
        for mu in range(min(Pn, 20)):
            probe = pats[mu].copy()
            probe[rng.random(N) < 0.1] *= -1
            sc, _, _ = hopfield_recall(W, probe, rng)
            ok_c.append(np.mean(sc == pats[mu]) > 0.99)
            ok_m.append(np.mean(modern_hopfield_recall(pats, probe) == pats[mu]) > 0.99)
        classic.append(float(np.mean(ok_c))); modern.append(float(np.mean(ok_m)))
        print(f"  P = {Pn:3d} (P/N = {Pn / N:.2f}): recalled classic {classic[-1]:.2f}  modern {modern[-1]:.2f}")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(np.array(loads) / N, classic, "o-", label="classic Hopfield (Hebbian)")
    ax.plot(np.array(loads) / N, modern, "s-", label="modern Hopfield (= attention)")
    ax.axvline(0.138, color="k", ls=":", lw=1); ax.text(0.142, 0.5, "0.138 N\n(theory)", fontsize=8)
    ax.set_xlabel("stored patterns / neurons  P/N"); ax.set_ylabel("fraction recalled (>99% bits)")
    ax.set_title("memory capacity, N = 200"); ax.legend(fontsize=8)
    savefig(fig, LESSON, "hopfield_capacity.png")
    return dict(glyph_overlap=overlap, glyph_recall=recall, hopfield_glyph_recall_exact=ok,
                hopfield_energy=[E[0], E[-1]], capacity_loads=loads,
                capacity_classic=classic, capacity_modern=modern)


def main(quick: bool = False) -> dict:
    rng = seed_everything(11)
    res = som_2d_part(rng)
    res.update(som_mnist_part(rng, quick))
    res.update(hopfield_part(rng))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
