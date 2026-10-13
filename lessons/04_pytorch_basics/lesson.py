"""Lesson 04 - PyTorch basics: autograd, nn.Module, a clean training loop; lesson 02 again.

Questions this lesson answers
-----------------------------
* What does autograd actually do, and how does it relate to our hand-written backprop?
* How do I add my own differentiable operation (torch.autograd.Function) and check it?
* What does a clean, idiomatic PyTorch training loop look like?
* Do PyTorch and our NumPy MLP give the *same* numbers? (Yes: to ~1e-15 in float64.)

Run:  python lessons/04_pytorch_basics/lesson.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnaz.common import (banner, decision_grid, save_results, savefig, seed_everything,
                         set_torch_threads, setup_matplotlib)
from nnaz.data import spirals
from nnaz.mlp_numpy import MLP
from nnaz.torch_basics import MySoftplus, TorchMLP, numpy_vs_torch_gd, to_tensors, train_loop

LESSON = 4
plt = setup_matplotlib()


def autograd_part():
    banner("1. Autograd: a recorded graph, walked backwards")
    # f(x, y) = x^2 y + sin(x y);  df/dx = 2xy + y cos(xy),  df/dy = x^2 + x cos(xy)
    x = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
    f = x ** 2 * y + torch.sin(x * y)
    print(f"  f = {f.item():.6f}; the graph ends in {f.grad_fn} <- {f.grad_fn.next_functions}")
    f.backward()
    dx = 2 * 1.5 * -0.7 + -0.7 * np.cos(1.5 * -0.7)
    dy = 1.5 ** 2 + 1.5 * np.cos(1.5 * -0.7)
    print(f"  autograd  df/dx = {x.grad.item():.12f}   by hand {dx:.12f}")
    print(f"  autograd  df/dy = {y.grad.item():.12f}   by hand {dy:.12f}")
    err_hand = max(abs(x.grad.item() - dx), abs(y.grad.item() - dy))

    # Custom op, verified by torch's own finite-difference checker (needs float64)
    z = torch.randn(20, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(MySoftplus.apply, (z,), eps=1e-6, atol=1e-8)
    print(f"  torch.autograd.gradcheck(MySoftplus): {ok}")

    # Higher derivatives: differentiate the gradient itself (create_graph=True)
    t = torch.linspace(-4, 4, 400, dtype=torch.float64, requires_grad=True)
    ys = [torch.tanh(t)]
    for _ in range(3):
        (g,) = torch.autograd.grad(ys[-1].sum(), t, create_graph=True)
        ys.append(g)
    # exact: tanh' = 1 - tanh^2, tanh'' = -2 tanh (1 - tanh^2)
    th = np.tanh(t.detach().numpy())
    err_d1 = np.abs(ys[1].detach().numpy() - (1 - th ** 2)).max()
    err_d2 = np.abs(ys[2].detach().numpy() - (-2 * th * (1 - th ** 2))).max()
    print(f"  d/dt tanh and d2/dt2 tanh by nested autograd: max error {err_d1:.1e}, {err_d2:.1e}")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k, yk in enumerate(ys):
        ax.plot(t.detach(), yk.detach(), label=["tanh", "1st derivative", "2nd", "3rd"][k])
    ax.set_title("derivatives of tanh by repeated autograd"); ax.legend()
    savefig(fig, LESSON, "autograd_derivatives.png")
    return dict(autograd_vs_hand_err=err_hand, custom_op_gradcheck=bool(ok),
                nested_grad_err_d1=float(err_d1), nested_grad_err_d2=float(err_d2))


def equivalence_part(rng):
    banner("2. The NumPy MLP of lesson 02 and PyTorch, from identical weights")
    X, y = spirals(300, rng)
    out, curves = {}, {}
    for lr in (0.1, 0.5):
        np_model = MLP([2, 32, 32, 3], np.random.default_rng(0), act="tanh", init="xavier")
        ln, lt, gdiff = numpy_vs_torch_gd(np_model, X, y, steps=300, lr=lr)
        diff = np.abs(ln - lt)
        curves[lr] = (ln, lt, diff)
        print(f"  lr {lr}: gradients at step 0 max |numpy - torch| = {gdiff:.1e};  over 300 full-batch GD"
              f" steps max |loss difference| = {diff.max():.1e}  (loss {ln[0]:.3f} -> {ln[-1]:.3f})")
        out[f"lr{lr}"] = dict(grad_diff_step0=gdiff, max_loss_diff=float(diff.max()),
                              loss_first=ln[0], loss_last=ln[-1])
    # At lr 0.5 the loss spikes (GD becomes unstable): the dynamics turn chaotic and
    # amplify the 1e-16 round-off differences between the two libraries.
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for lr, (ln, lt, diff) in curves.items():
        l, = axs[0].plot(ln, lw=3, alpha=0.4, label=f"NumPy, lr {lr}")
        axs[0].plot(lt, "--", color=l.get_color(), lw=1, label=f"PyTorch, lr {lr}")
        axs[1].semilogy(diff + 1e-17, color=l.get_color(), label=f"lr {lr}")
    axs[0].set_xlabel("GD step"); axs[0].set_ylabel("loss"); axs[0].legend(fontsize=8)
    axs[0].set_title("same weights, same data, two libraries")
    axs[1].set_xlabel("GD step"); axs[1].legend()
    axs[1].set_title("|loss NumPy - loss PyTorch|")
    savefig(fig, LESSON, "numpy_vs_torch.png")
    return {"equivalence": out}


def training_part(rng, quick):
    banner("3. A clean training loop: Dataset, DataLoader, nn.Module, optimiser")
    X, y = spirals(600, rng)
    Xte, yte = spirals(1500, rng)
    ds = TensorDataset(*to_tensors(X, y))
    loader = DataLoader(ds, batch_size=64, shuffle=True)  # shuffles every epoch
    model = TorchMLP([2, 64, 64, 3])
    print(model)
    n_par = sum(p.numel() for p in model.parameters())
    hist = train_loop(model, loader, to_tensors(Xte, yte), epochs=30 if quick else 300, lr=1e-2, log=50)
    gx, gy, G = decision_grid(X)
    with torch.no_grad():
        pred = model(torch.as_tensor(G, dtype=torch.float32)).argmax(1).numpy()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    axs[0].contourf(gx, gy, pred.reshape(gx.shape), levels=[-.5, .5, 1.5, 2.5], cmap="Pastel1")
    axs[0].scatter(*X.T, c=y, cmap="Set1", s=5, vmin=0, vmax=8)
    axs[0].set_title(f"PyTorch MLP, test acc {hist['val_acc'][-1]:.3f}"); axs[0].set_aspect("equal")
    axs[1].semilogy(hist["loss"], label="train"); axs[1].semilogy(hist["val_loss"], label="test")
    axs[1].set_xlabel("epoch"); axs[1].legend(); axs[1].set_title("cross-entropy")
    savefig(fig, LESSON, "torch_spiral.png")
    return dict(n_params=n_par, torch_test_acc=hist["val_acc"][-1], torch_loss_first=hist["loss"][0],
                torch_loss_last=hist["loss"][-1])


def main(quick: bool = False) -> dict:
    rng = seed_everything(4)
    set_torch_threads()
    res = autograd_part()
    res.update(equivalence_part(rng))
    res.update(training_part(rng, quick))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
