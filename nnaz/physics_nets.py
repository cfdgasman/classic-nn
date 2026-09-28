"""Lesson 15: physics-structured networks - Hamiltonian neural networks and neural ODEs.

Hamiltonian neural network (HNN, Greydanus, Dzamba & Yosinski 2019)
--------------------------------------------------------------------
A conservative mechanical system is fully described by a scalar energy H(q, p):

    dq/dt = dH/dp,     dp/dt = -dH/dq.

A plain network that maps (q, p) -> (dq/dt, dp/dt) can learn any vector field,
including slightly dissipative or energy-pumping ones, and small errors make
long simulations drift. An HNN instead learns the SCALAR H_theta(q, p) and gets the
vector field from its gradient with autograd - the flow is then exactly
Hamiltonian (for H_theta), so the learnt energy is conserved by construction.

Neural ODE (Chen et al. 2018)
-----------------------------
Model dx/dt = f_theta(x) and fit it to *trajectory samples* (no derivatives
needed) by integrating the ODE with a differentiable solver and back-propagating
through the solver steps ("discretise-then-optimise"; the adjoint method is the
memory-cheap alternative). A continuous-time model handles irregular sampling
and can be queried at any time step.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def mlp(i, o, h=128, depth=2, act=nn.Tanh):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), act()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


# ================================================================ pendulum
def pendulum_H(q, p):
    return 0.5 * p ** 2 + (1 - np.cos(q))


def pendulum_field(x):
    q, p = x[..., 0], x[..., 1]
    return np.stack([p, -np.sin(q)], -1)


def pendulum_data(n_traj, rng, T=10.0, dt=0.1, noise=0.05, e_range=(0.2, 1.6)):
    """Trajectories at random energies below the separatrix (H < 2), sampled every dt;
    targets are the true time derivatives plus Gaussian noise (as in Greydanus et al.)."""
    from scipy.integrate import solve_ivp

    X, dX = [], []
    t_eval = np.arange(0, T, dt)
    for _ in range(n_traj):
        E = rng.uniform(*e_range)
        q0 = np.arccos(1 - E) * rng.choice([-1, 1]) * rng.uniform(0.2, 1.0)
        p0 = np.sign(rng.normal()) * np.sqrt(max(2 * (E - (1 - np.cos(q0))), 0))
        sol = solve_ivp(lambda t, x: pendulum_field(x), (0, T), [q0, p0], t_eval=t_eval, rtol=1e-10, atol=1e-10)
        x = sol.y.T
        X.append(x)
        dX.append(pendulum_field(x) + noise * rng.normal(size=x.shape))
    return np.concatenate(X), np.concatenate(dX)


class Baseline(nn.Module):
    """(q, p) -> (dq/dt, dp/dt) directly."""

    def __init__(self):
        super().__init__()
        self.net = mlp(2, 2)

    def forward(self, x):
        return self.net(x)


class HNN(nn.Module):
    """(q, p) -> H; the field is the symplectic gradient J grad H."""

    def __init__(self):
        super().__init__()
        self.H = mlp(2, 1)

    def forward(self, x):
        with torch.enable_grad():
            x = x if x.requires_grad else x.requires_grad_(True)
            H = self.H(x).sum()
            dH = torch.autograd.grad(H, x, create_graph=True)[0]
        return torch.stack([dH[:, 1], -dH[:, 0]], 1)     # (dH/dp, -dH/dq)


def train_field(model, X, dX, steps=3000, lr=1e-3, batch=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    X, dX = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(dX, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for s in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        loss = ((model(X[idx]) - dX[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
    return hist


def rk4(f, x0, dt, n):
    """Classical RK4 with a (numpy or torch) vector field f; returns [n+1, ...]."""
    xs = [x0]
    x = x0
    for _ in range(n):
        k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return xs


def model_field_np(model):
    def f(x):
        xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
        out = model(xt).detach().numpy()
        return out.reshape(np.shape(x))
    return f


# ================================================================ neural ODE: Lotka-Volterra
LV = dict(a=1.0, b=0.4, c=0.4, d=0.1)   # prey growth, predation, predator death, conversion


def lotka_volterra(x):
    u, v = x[..., 0], x[..., 1]
    return np.stack([LV["a"] * u - LV["b"] * u * v, -LV["c"] * v + LV["d"] * u * v], -1)


def lv_data(rng, T=30.0, n_obs=120, x0=(10.0, 5.0), noise=0.02):
    """One trajectory observed at IRREGULAR times with 2 % multiplicative noise."""
    from scipy.integrate import solve_ivp

    t = np.sort(rng.uniform(0, T, n_obs)); t[0] = 0.0
    sol = solve_ivp(lambda _, x: lotka_volterra(x), (0, T), x0, t_eval=t, rtol=1e-10, atol=1e-10)
    y = sol.y.T * (1 + noise * rng.normal(size=sol.y.T.shape))
    return t, y, sol.y.T


class ODEFunc(nn.Module):
    """dx/dt = f_theta(x), working in log-populations for positivity and scale:
    z = log x,  dz/dt = f(z)."""

    def __init__(self):
        super().__init__()
        self.net = mlp(2, 2, h=64, act=nn.Tanh)

    def forward(self, z):
        return self.net(z)


def odeint_rk4(f, z0, t_grid, substeps=4):
    """Integrate from t_grid[0] through all t_grid points (irregular spacing allowed),
    with `substeps` RK4 steps between consecutive points; differentiable in torch."""
    out = [z0]
    z = z0
    for k in range(len(t_grid) - 1):
        h = (t_grid[k + 1] - t_grid[k]) / substeps
        for _ in range(substeps):
            k1 = f(z); k2 = f(z + 0.5 * h * k1); k3 = f(z + 0.5 * h * k2); k4 = f(z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(z)
    return torch.stack(out)


def train_neural_ode(func, t, y, steps=1500, window=12, batch=16, lr=5e-3, seed=0, log=500):
    """Multiple shooting: random windows of `window` consecutive observations,
    each integrated from its own (observed) start - short horizons keep gradients sane."""
    g = torch.Generator().manual_seed(seed)
    Z = torch.as_tensor(np.log(y), dtype=torch.float32)
    tt = torch.as_tensor(t, dtype=torch.float32)
    opt = torch.optim.Adam(func.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist, t0 = [], time.time()
    offs = torch.arange(window)
    for s in range(steps):
        starts = torch.randint(0, len(t) - window, (batch,), generator=g)
        idx = starts[:, None] + offs[None]                  # [batch, window]
        Zw, Tw = Z[idx], tt[idx]                            # all windows integrated TOGETHER,
        z, preds = Zw[:, 0], [Zw[:, 0]]                     # each with its own (irregular) steps
        for k in range(window - 1):
            h = ((Tw[:, k + 1] - Tw[:, k]) / 2)[:, None]    # 2 RK4 substeps per interval
            for _ in range(2):
                k1 = func(z); k2 = func(z + 0.5 * h * k1); k3 = func(z + 0.5 * h * k2); k4 = func(z + h * k3)
                z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            preds.append(z)
        loss = ((torch.stack(preds, 1) - Zw) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
        if log and (s + 1) % log == 0:
            print(f"    step {s + 1:5d}  loss {np.mean(hist[-log:]):.2e}  ({time.time() - t0:.0f}s)")
    return hist, time.time() - t0
