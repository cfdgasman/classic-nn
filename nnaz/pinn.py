"""Lesson 09: physics-informed neural networks (PINNs) and neural surrogates.

PINN idea (Raissi, Perdikaris & Karniadakis 2019): represent the solution by a
network u_theta(x, t) and minimise the PDE residual

    r(x, t) = u_t + u u_x - nu u_xx

at "collocation" points scattered over the space-time domain. The derivatives
are *exact* derivatives of the network, obtained by automatic differentiation
(``torch.autograd.grad`` with ``create_graph=True`` so we can differentiate
again for u_xx and then back-propagate the loss to theta). No solution data is used.

Boundary/initial conditions can be imposed softly (extra loss terms) or, as
here, *hard*: we build them into the ansatz

    u(x, t) = -A sin(pi x) + t (1 - x^2) N_theta(x, t),

which satisfies u(x, 0) = -A sin(pi x) and u(+-1, t) = 0 for any theta. The
optimiser then only has to drive the residual to zero.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


class MLPNet(nn.Module):
    def __init__(self, n_in=2, n_out=1, width=32, depth=6, act=nn.Tanh):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)   # Xavier suits tanh (lesson 03)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class BurgersPINN(nn.Module):
    def __init__(self, A=1.0, **kw):
        super().__init__()
        self.A, self.net = A, MLPNet(2, 1, **kw)

    def forward(self, x, t):
        # inputs rescaled to [-1, 1] (x already is; t in [0,1] -> 2t - 1)
        n = self.net(torch.cat([x, 2 * t - 1], 1))
        return -self.A * torch.sin(np.pi * x) + t * (1 - x ** 2) * n


def burgers_residual(model, x, t, nu):
    x.requires_grad_(True); t.requires_grad_(True)
    u = model(x, t)
    ones = torch.ones_like(u)
    u_x, u_t = torch.autograd.grad(u, (x, t), ones, create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def train_pinn(model, nu, n_colloc=8000, adam_steps=3000, lbfgs_steps=2000, lr=1e-3, seed=0,
               eval_fn=None, eval_every=250, log=True):
    """Adam first (robust far from the solution), then L-BFGS (fast, accurate
    near it) - the standard PINN recipe."""
    g = torch.Generator().manual_seed(seed)
    # collocation points: uniform in x, t, plus extra points early in time and near
    # x = 0 where the steep front forms (a simple, fixed refinement)
    xc = torch.rand(n_colloc, 1, generator=g, dtype=torch.float64) * 2 - 1
    tc = torch.rand(n_colloc, 1, generator=g, dtype=torch.float64)
    xs = torch.randn(n_colloc // 4, 1, generator=g, dtype=torch.float64) * 0.1
    ts = torch.rand(n_colloc // 4, 1, generator=g, dtype=torch.float64)
    xc, tc = torch.cat([xc, xs.clamp(-1, 1)]), torch.cat([tc, ts])
    hist = {"it": [], "loss": [], "err": []}
    t0 = time.time()

    def loss_fn():
        return (burgers_residual(model, xc, tc, nu) ** 2).mean()

    def record(it, loss):
        hist["it"].append(it); hist["loss"].append(float(loss))
        if eval_fn is not None:
            hist["err"].append(eval_fn(model))
        if log:
            e = f"  rel. L2 error {hist['err'][-1]:.2e}" if eval_fn else ""
            print(f"    iter {it:5d}  residual loss {loss:.3e}{e}  ({time.time() - t0:.0f}s)")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(adam_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if it % eval_every == 0:
            record(it, loss.item())
    lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=eval_every, history_size=50,
                           tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        l = loss_fn()
        l.backward()
        return l

    done = 0
    while done < lbfgs_steps:
        lb.step(closure)
        done += eval_every
        record(adam_steps + done, loss_fn().item())
    hist["time"] = time.time() - t0
    return hist


def pinn_eval_grid(model, x, t):
    """Evaluate the PINN on the tensor product grid x (Nx) x t (Nt) -> [Nt, Nx]."""
    X, T = np.meshgrid(x, t)
    with torch.no_grad():
        u = model(torch.as_tensor(X.reshape(-1, 1)), torch.as_tensor(T.reshape(-1, 1)))
    return u.numpy().reshape(T.shape)


# ---------------------------------------------------------------- surrogate
class Surrogate(nn.Module):
    """Parameters (nu, A) -> the whole solution u(x, T) on a fixed grid."""

    def __init__(self, n_out, width=128, depth=3):
        super().__init__()
        self.net = MLPNet(2, n_out, width=width, depth=depth, act=nn.GELU)

    def forward(self, p):
        return self.net(p)


def normalise_params(nu, A, nu_rng, A_rng):
    """Map (log nu, A) to [-1, 1]^2 - viscosities span a decade, so use log nu."""
    lnu = (np.log(nu) - np.log(nu_rng[0])) / (np.log(nu_rng[1]) - np.log(nu_rng[0]))
    a = (A - A_rng[0]) / (A_rng[1] - A_rng[0])
    return np.c_[2 * lnu - 1, 2 * a - 1]


def train_surrogate(model, P, U, epochs=3000, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    P, U = torch.as_tensor(P, dtype=torch.float32), torch.as_tensor(U, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(P))
        for i in range(0, len(P), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = ((model(P[idx]) - U[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        sched.step()
        hist.append(loss.item())
    return hist
