"""Lesson 05 - Convolutional networks on MNIST and Fashion-MNIST.

Questions this lesson answers
-----------------------------
* What does a convolution compute, and how is it implemented efficiently (im2col)?
* Why do CNNs beat fully connected nets on images with the same number of parameters?
* What do the learnt filters and feature maps look like?
* How far is > 98 % on MNIST from trivial baselines, and how hard is Fashion-MNIST?

Run:  python lessons/05_cnn_mnist/lesson.py      (downloads MNIST/Fashion-MNIST once, ~60 MB)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from nnaz.cnn import (LogReg, MLPBaseline, SmallCNN, accuracy, confusion, conv2d_im2col, conv2d_numpy,
                      fit, predict, to_tensor_images)
from nnaz.common import banner, save_gif, save_results, savefig, seed_everything, set_torch_threads, setup_matplotlib
from nnaz.data import FASHION_CLASSES, load_mnist

LESSON = 5
plt = setup_matplotlib()


def convolution_part():
    banner("1. What a convolution computes (and three ways to compute it)")
    x, _ = load_mnist("MNIST", train=True)
    img = x[0].astype(np.float64) / 255
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)  # a vertical-edge detector
    loops = conv2d_numpy(img, sobel)
    rng = np.random.default_rng(0)
    X3 = rng.normal(size=(3, 12, 12)); K = rng.normal(size=(4, 3, 5, 5)); b = rng.normal(size=4)
    y_col = conv2d_im2col(X3, K, b)
    y_torch = F.conv2d(torch.from_numpy(X3)[None], torch.from_numpy(K), torch.from_numpy(b))[0].numpy()
    y_loop = np.stack([sum(conv2d_numpy(X3[c], K[o, c]) for c in range(3)) + b[o] for o in range(4)])
    e1, e2 = np.abs(y_col - y_torch).max(), np.abs(y_loop - y_torch).max()
    print(f"  loops vs torch.conv2d: {e2:.1e};  im2col vs torch.conv2d: {e1:.1e}")

    # GIF: the 3x3 kernel sliding over the digit, building the output map
    from matplotlib.animation import FuncAnimation

    fig, axs = plt.subplots(1, 2, figsize=(6, 3.2))
    axs[0].imshow(img, cmap="gray_r"); axs[0].set_title("input * Sobel kernel")
    out = np.full_like(loops, np.nan)
    im = axs[1].imshow(out, cmap="RdBu", vmin=-np.abs(loops).max(), vmax=np.abs(loops).max())
    axs[1].set_title("output (vertical edges)")
    rect = plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="r", lw=2)
    axs[0].add_patch(rect)
    for a in axs:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    positions = [(i, j) for i in range(0, 26, 1) for j in range(0, 26, 1)]

    def draw(f):
        for (i, j) in positions[f * 13:(f + 1) * 13]:  # 13 output pixels per frame
            out[i, j] = loops[i, j]
        i, j = positions[min((f + 1) * 13, len(positions)) - 1]
        rect.set_xy((j - 0.5, i - 0.5))
        im.set_data(out)
        return im, rect

    save_gif(FuncAnimation(fig, draw, frames=len(positions) // 13), LESSON, "convolution.gif", fps=15, dpi=60)
    return dict(conv_loops_vs_torch=e2, conv_im2col_vs_torch=e1)


def dataset_part(name, quick, show_filters):
    banner(f"2. {name}: logistic regression vs MLP vs CNN")
    xtr, ytr = load_mnist(name, True)
    xte, yte = load_mnist(name, False)
    if quick:
        xtr, ytr = xtr[:6000], ytr[:6000]
    Xtr, m, s = to_tensor_images(xtr)
    Xte, _, _ = to_tensor_images(xte, m, s)
    ytr_t, yte_t = torch.as_tensor(ytr), torch.as_tensor(yte)
    res = {"majority": float(np.bincount(yte).max() / len(yte))}
    epochs = 1 if quick else 5
    models = {"logistic regression": LogReg(), "MLP 784-96-10": MLPBaseline(), "CNN": SmallCNN()}
    hists = {}
    for mname, model in models.items():
        torch.manual_seed(0)
        model.apply(lambda mod: mod.reset_parameters() if hasattr(mod, "reset_parameters") else None)
        n_par = sum(p.numel() for p in model.parameters())
        h = fit(model, Xtr, ytr_t, epochs=epochs, Xte=Xte, yte=yte_t)
        hists[mname] = h
        res[mname] = {"params": n_par, "test_acc": h["test_acc"][-1], "train_time_s": h["time"]}
        print(f"  {mname:20s} {n_par:6d} params  test acc {h['test_acc'][-1]:.4f}  ({h['time']:.0f}s)")
    print(f"  majority-class baseline: {res['majority']:.3f}")
    tag = name.lower()

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for mname, h in hists.items():
        ax.semilogy(h["step"], h["loss"], label=f"{mname} ({h['test_acc'][-1]:.2%})")
    ax.set_xlabel("mini-batch step"); ax.set_ylabel("train loss"); ax.legend(fontsize=8)
    ax.set_title(f"{name}: training loss (test accuracy)")
    savefig(fig, LESSON, f"{tag}_training.png")

    cnn = models["CNN"]
    pred = predict(cnn, Xte).numpy()
    M = confusion(yte, pred)
    labels = FASHION_CLASSES if name == "FashionMNIST" else [str(i) for i in range(10)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    Mn = M / M.sum(1, keepdims=True)
    axs[0].imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if M[i, j]:
                axs[0].text(j, i, M[i, j], ha="center", va="center", fontsize=6,
                            color="w" if Mn[i, j] > 0.5 else "k")
    axs[0].set_xticks(range(10), labels, rotation=90, fontsize=7); axs[0].set_yticks(range(10), labels, fontsize=7)
    axs[0].set_xlabel("predicted"); axs[0].set_ylabel("true"); axs[0].grid(False)
    axs[0].set_title(f"CNN confusion matrix ({(pred == yte).mean():.2%})")
    wrong = np.where(pred != yte)[0][:24]
    axs[1].axis("off"); axs[1].set_title("misclassified test images: true -> predicted")
    for k, i in enumerate(wrong):
        a = axs[1].inset_axes([(k % 8) / 8, 1 - (k // 8 + 1) / 3, 0.115, 0.28])
        a.imshow(xte[i], cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{labels[yte[i]][:7]}->{labels[pred[i]][:7]}", fontsize=6)
    savefig(fig, LESSON, f"{tag}_confusion.png")

    if show_filters:
        visualise(cnn, Xte, xte, name)
    res["confusion"] = M.tolist()
    return res


def visualise(cnn, Xte, xte, name):
    W = cnn.conv1.weight.detach()[:, 0].numpy()
    fig, axs = plt.subplots(2, 8, figsize=(9, 2.6))
    for k, a in enumerate(axs.flat):
        v = np.abs(W[k]).max()
        a.imshow(W[k], cmap="RdBu", vmin=-v, vmax=v); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("the 16 learnt 5x5 filters of conv1 (red +, blue -): edge and stroke detectors")
    savefig(fig, LESSON, "conv1_filters.png")

    with torch.no_grad():
        cnn.eval()
        a1, a2 = cnn.features(Xte[:1])
    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_subplot(1, 5, 1); ax.imshow(xte[0], cmap="gray_r"); ax.set_title("input"); ax.axis("off")
    for k in range(16):
        ax = fig.add_subplot(4, 10, (k // 4) * 10 + 3 + k % 4)
        ax.imshow(a1[0, k], cmap="magma"); ax.axis("off")
    for k in range(16):
        ax = fig.add_subplot(4, 10, (k // 4) * 10 + 7 + k % 4)
        ax.imshow(a2[0, k], cmap="magma"); ax.axis("off")
    fig.text(0.38, 1.03, "conv1 feature maps (24x24)", ha="center")
    fig.text(0.78, 1.03, "conv2 feature maps (8x8), first 16 of 32", ha="center")
    savefig(fig, LESSON, "feature_maps.png")


def main(quick: bool = False) -> dict:
    seed_everything(5)
    set_torch_threads()
    res = convolution_part()
    res["MNIST"] = dataset_part("MNIST", quick, show_filters=True)
    res["FashionMNIST"] = dataset_part("FashionMNIST", quick, show_filters=False)
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
