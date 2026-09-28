"""Lesson 04: PyTorch basics - tensors, autograd, nn.Module and a clean training loop.

The key idea of autograd: every tensor operation performed on a tensor with
``requires_grad=True`` is recorded in a graph; ``loss.backward()`` walks that
graph in reverse and applies each operation's local derivative - exactly what
our hand-written ``Layer.backward`` methods did in lesson 02, but automatic.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchMLP(nn.Module):
    """The same architecture as ``nnaz.mlp_numpy.MLP`` (Linear -> act -> ... -> Linear)."""

    def __init__(self, sizes, act: str = "relu"):
        super().__init__()
        Act = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[act]
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(Act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def copy_numpy_weights(np_model, torch_model: TorchMLP) -> None:
    """Load the NumPy MLP's weights into the torch model.

    NumPy stores W as [in, out] (y = x W); nn.Linear stores weight as [out, in]
    (y = x W^T), hence the transpose.
    """
    from .mlp_numpy import Linear

    np_lin = [l for l in np_model.layers if isinstance(l, Linear)]
    t_lin = [m for m in torch_model.net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for a, b in zip(np_lin, t_lin):
            b.weight.copy_(torch.from_numpy(a.params["W"].T))
            b.bias.copy_(torch.from_numpy(a.params["b"]))


class MySoftplus(torch.autograd.Function):
    """A custom autograd op: softplus(x) = log(1 + e^x), with d/dx = sigmoid(x).

    Writing forward *and* backward yourself is how you extend PyTorch with
    operations it does not know (or want to differentiate more cheaply)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.logaddexp(torch.zeros_like(x), x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * torch.sigmoid(x)


def train_loop(model, train_loader, val_data, epochs: int, lr: float = 1e-2, log=None):
    """The canonical PyTorch training loop.

    for each epoch:
        model.train()
        for xb, yb in loader:
            opt.zero_grad()          # gradients ACCUMULATE by default - reset them
            loss = loss_fn(model(xb), yb)
            loss.backward()          # autograd fills p.grad for every parameter
            opt.step()               # the optimiser reads p.grad and updates p
        model.eval(); with torch.no_grad(): evaluate
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"loss": [], "val_loss": [], "val_acc": []}
    Xv, yv = val_data
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        model.eval()
        with torch.no_grad():  # no graph needed for evaluation: faster, less memory
            out = model(Xv)
            hist["loss"].append(tot / n)
            hist["val_loss"].append(loss_fn(out, yv).item())
            hist["val_acc"].append((out.argmax(1) == yv).float().mean().item())
        if log and (ep % log == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  train loss {hist['loss'][-1]:.4f}  val acc {hist['val_acc'][-1]:.3f}")
    return hist


def to_tensors(X, y, dtype=torch.float32):
    return torch.as_tensor(X, dtype=dtype), torch.as_tensor(y, dtype=torch.long)


def numpy_vs_torch_gd(np_model, X, y, steps: int, lr: float):
    """Run full-batch gradient descent with *identical* initial weights in the
    NumPy MLP and in PyTorch (float64); return both loss histories and the
    maximum gradient discrepancy at step 0."""
    from .mlp_numpy import SGD

    tm = TorchMLP([l.params["W"].shape[0] for l in np_model.layers if "W" in l.params]
                  + [np_model.layers[-1].params["W"].shape[1]], act="tanh").double()
    copy_numpy_weights(np_model, tm)
    Xt, yt = to_tensors(X, y, torch.float64)
    opt_t = torch.optim.SGD(tm.parameters(), lr=lr)
    opt_n = SGD(np_model.param_list(), lr)
    ln, lt, gdiff = [], [], None
    for k in range(steps):
        ln.append(np_model.loss_and_grads(X, y))
        opt_t.zero_grad()
        loss = nn.functional.cross_entropy(tm(Xt), yt)
        loss.backward()
        lt.append(loss.item())
        if k == 0:
            t_lin = [m for m in tm.net if isinstance(m, nn.Linear)]
            n_lin = [l for l in np_model.layers if "W" in l.params]
            gdiff = max(max(np.abs(a.grads["W"] - b.weight.grad.numpy().T).max(),
                            np.abs(a.grads["b"] - b.bias.grad.numpy()).max()) for a, b in zip(n_lin, t_lin))
        opt_n.step()
        opt_t.step()
    return np.array(ln), np.array(lt), gdiff
