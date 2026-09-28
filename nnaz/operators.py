"""Lesson 14: neural operators - DeepONet and the Fourier Neural Operator (FNO).

A neural *operator* learns a map between FUNCTIONS, here the Burgers solution
operator  G: u0(x) -> u(x, T),  from many (u0, u(T)) pairs produced by a solver.
Once trained, one forward pass replaces a whole PDE solve for any new u0.

DeepONet (Lu et al. 2021). Universal approximation theorem for operators
(Chen & Chen 1995):  G(u0)(x) ~ sum_k b_k(u0) t_k(x) + b0,
  branch net b: the input function sampled at m fixed "sensors" -> p coefficients,
  trunk net  t: a query location x -> p basis functions.
It is a learnt, nonlinear version of a reduced basis expansion.

FNO (Li et al. 2021). Layers of the form
  v <- act( W v + F^-1[ R_k . F[v]_k ]  for the lowest k_max Fourier modes ),
i.e. a pointwise linear map plus a GLOBAL convolution whose kernel is learnt
directly in Fourier space (truncated to low modes). Because the parameters live
on Fourier modes, not grid points, the same network can be evaluated on a finer
grid than it was trained on ("zero-shot super-resolution").
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


# ================================================================ data: pseudo-spectral Burgers
def grf_1d(n_samples, N, rng, alpha=2.0, tau=5.0, sigma=25.0):
    """Periodic Gaussian random fields on [0,1) with covariance
    sigma^2 (-Laplace + tau^2 I)^-alpha  (the setting of the FNO Burgers benchmark)."""
    k = np.fft.rfftfreq(N, d=1.0 / N)                  # 0, 1, ..., N/2
    amp = sigma * (4 * np.pi ** 2 * k ** 2 + tau ** 2) ** (-alpha / 2)
    amp[0] = 0.0                                         # zero mean
    coef = (rng.normal(size=(n_samples, len(k))) + 1j * rng.normal(size=(n_samples, len(k)))) * amp
    return np.fft.irfft(coef, n=N) * N / np.sqrt(2)


def burgers_spectral(u0, nu, T, Lx=1.0, dt=None, cfl=0.4):
    """Pseudo-spectral solver for u_t + (u^2/2)_x = nu u_xx on a periodic domain.

    Fourier in space (exact derivatives for smooth periodic functions), 2/3-rule
    de-aliasing of the quadratic term, and an *integrating factor* for diffusion:
    with v = e^{nu k^2 t} u_hat the stiff linear term disappears and classical RK4
    is applied to the nonlinear part only, so dt is limited by advection, not by
    dx^2/nu. u0: [B, N] (batched over initial conditions).
    """
    u0 = np.atleast_2d(u0)
    B, N = u0.shape
    k = 2 * np.pi / Lx * np.fft.rfftfreq(N, d=1.0 / N)
    dealias = k <= (2 / 3) * k.max()
    if dt is None:
        dt = cfl * (Lx / N) / max(np.abs(u0).max(), 1e-8)
    n = int(np.ceil(T / dt)); dt = T / n
    E, E2 = np.exp(-nu * k ** 2 * dt), np.exp(-nu * k ** 2 * dt / 2)

    def N_hat(uh):
        u = np.fft.irfft(uh, n=N)
        return -0.5j * k * np.fft.rfft(u * u) * dealias

    uh = np.fft.rfft(u0)
    for _ in range(n):                                   # Lawson / integrating-factor RK4
        a = N_hat(uh)
        b = N_hat(E2 * (uh + 0.5 * dt * a))
        c = N_hat(E2 * uh + 0.5 * dt * b)
        d = N_hat(E * uh + dt * E2 * c)
        uh = E * uh + dt / 6 * (E * a + 2 * E2 * (b + c) + d)
    return np.fft.irfft(uh, n=N), dict(n_steps=n, dt=dt)


# ================================================================ DeepONet
def mlp(i, o, h=128, depth=3, act=nn.GELU):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), act()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class DeepONet(nn.Module):
    def __init__(self, m_sensors, p=128, h=128):
        super().__init__()
        self.branch = mlp(m_sensors, p, h)
        self.trunk = mlp(2, p, h)       # x encoded as (cos 2 pi x, sin 2 pi x): periodicity built in
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u0, x):
        """u0: [B, m] at the sensors, x: [Q] query points -> [B, Q]."""
        t = self.trunk(torch.stack([torch.cos(2 * np.pi * x), torch.sin(2 * np.pi * x)], 1))
        return self.branch(u0) @ t.T + self.b0


# ================================================================ FNO
class SpectralConv1d(nn.Module):
    def __init__(self, c_in, c_out, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (c_in * c_out)
        self.w = nn.Parameter(scale * torch.randn(c_in, c_out, modes, dtype=torch.cfloat))

    def forward(self, v):                                 # v: [B, C, N]
        vh = torch.fft.rfft(v)
        out = torch.zeros(v.shape[0], self.w.shape[1], vh.shape[-1], dtype=torch.cfloat)
        m = min(self.modes, vh.shape[-1])
        out[..., :m] = torch.einsum("bik,iok->bok", vh[..., :m], self.w[..., :m])
        return torch.fft.irfft(out, n=v.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64, layers=4):
        super().__init__()
        self.lift = nn.Linear(2, width)                   # input channels: u0(x), x
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, u0):                                # u0: [B, N] on a uniform grid of [0,1)
        B, N = u0.shape
        x = torch.linspace(0, 1, N + 1)[:-1].expand(B, N)
        v = self.lift(torch.stack([u0, x], -1)).transpose(1, 2)   # [B, width, N]
        for i, (s, w) in enumerate(zip(self.spec, self.pw)):
            v = s(v) + w(v)
            if i < len(self.spec) - 1:
                v = nn.functional.gelu(v)
        return self.proj(v.transpose(1, 2)).squeeze(-1)


def rel_l2(pred, true):
    return (torch.linalg.norm(pred - true, dim=-1) / torch.linalg.norm(true, dim=-1))


def train_operator(model, U0, UT, epochs=300, batch=20, lr=1e-3, deeponet_x=None, seed=0, log=100):
    """Relative-L2 loss (the metric of the FNO paper), Adam + cosine schedule."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(U0) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(U0))
        tot = 0.0
        for i in range(0, len(U0), batch):
            idx = perm[i:i + batch]
            pred = model(U0[idx], deeponet_x) if deeponet_x is not None else model(U0[idx])
            loss = rel_l2(pred, UT[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        hist.append(tot / len(U0))
        if log and (ep + 1) % log == 0:
            print(f"    epoch {ep + 1:4d}  train rel. L2 {hist[-1]:.4f}  ({time.time() - t0:.0f}s)")
    return hist, time.time() - t0
