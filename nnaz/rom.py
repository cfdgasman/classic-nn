"""Lesson 16: reduced-order modelling (ROM) of a CFD flow field.

Snapshot matrix S = [q_1 ... q_m] (each column a flattened vorticity field).

POD (proper orthogonal decomposition) = PCA of the snapshots: with the mean
removed, S' = U Sigma V^T; the columns of U are the POD modes (orthonormal
spatial structures ranked by energy sigma_k^2), and a_k(t) = U_k^T q'(t) are
the temporal coefficients. Truncating to r modes gives the optimal LINEAR
r-dimensional representation (Eckart-Young; see lesson 07).

A periodic wake is a limit cycle: its state is essentially one phase angle, so
a NONLINEAR 2-D code (a circle) can represent it, whereas POD needs pairs of
modes for every harmonic of the shedding frequency. A convolutional
autoencoder can exploit that.

Dynamics in the reduced space:
  * DMD (dynamic mode decomposition, Schmid 2010): the best LINEAR map
    a_{n+1} = A a_n fitted by least squares; its eigenvalues give frequencies
    and growth rates.
  * a small neural map z_{n+1} = z_n + g(z_n) in the autoencoder's latent space.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def crop(snaps, cx, D):
    """Wake window: 256 x 96 cells from just upstream of the cylinder, 2x downsampled -> 128 x 48."""
    x0 = cx - D // 2
    s = snaps[:, x0:x0 + 256, 2:98]
    return s[:, ::2, ::2].transpose(0, 2, 1).copy()          # [m, 48, 128] (rows = y)


def pod(S):
    """S: [m, n] snapshots as rows. Returns mean, modes [n, m'], singular values, coefficients."""
    mu = S.mean(0)
    U, s, Vt = np.linalg.svd((S - mu).T, full_matrices=False)
    return mu, U, s, (S - mu) @ U


def dmd_fit(A_coef):
    """Least-squares linear propagator for coefficient rows a_n: a_{n+1} ~ a_n M."""
    X, Y = A_coef[:-1], A_coef[1:]
    return np.linalg.lstsq(X, Y, rcond=None)[0]


class ConvAE(nn.Module):
    def __init__(self, r=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.GELU(),        # 24 x 64
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),       # 12 x 32
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),       # 6 x 16
            nn.Flatten(), nn.Linear(64 * 6 * 16, r))
        self.dec = nn.Sequential(
            nn.Linear(r, 64 * 6 * 16), nn.GELU(), nn.Unflatten(1, (64, 6, 16)),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(model, F, epochs=600, batch=16, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(F) / batch)))
    X = torch.as_tensor(F[:, None], dtype=torch.float32)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(xb)
        hist.append(tot / len(X))
    return hist, time.time() - t0


class LatentStep(nn.Module):
    """z_{n+1} = z_n + g(z_n)."""

    def __init__(self, r, h=64):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(r, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, r))

    def forward(self, z):
        return z + self.g(z)


def train_latent(step, Z, epochs=3000, lr=3e-3, horizon=5, seed=0):
    """Fit multi-step rollouts of length `horizon` (more robust than one-step fitting)."""
    torch.manual_seed(seed)
    Z = torch.as_tensor(Z, dtype=torch.float32)
    opt = torch.optim.Adam(step.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Z) - horizon
    for ep in range(epochs):
        z = Z[:n]
        loss = 0.0
        for h in range(1, horizon + 1):
            z = step(z)
            loss = loss + ((z - Z[h:n + h]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return loss.item()
