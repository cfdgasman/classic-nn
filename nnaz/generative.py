"""Lesson 10: generative models - a variational autoencoder and a diffusion model.

VAE (Kingma & Welling 2014). A latent-variable model p(x) = Int p(x|z) p(z) dz with
p(z) = N(0, I) and a decoder p(x|z). The encoder q(z|x) = N(mu(x), diag sigma^2(x))
approximates the posterior, and we maximise the evidence lower bound

    log p(x) >= ELBO = E_q[ log p(x|z) ] - KL( q(z|x) || p(z) ).

The KL between two Gaussians is analytic:  KL = 1/2 sum( mu^2 + sigma^2 - log sigma^2 - 1 ).
The expectation is estimated with ONE sample z = mu + sigma * eps, eps ~ N(0, I):
the *reparameterisation trick* makes z a differentiable function of (mu, sigma).

DDPM (Ho, Jain & Abbeel 2020). A fixed forward process adds Gaussian noise,

    x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,     abar_t = prod_{s<=t} (1 - beta_s),

and a network eps_theta(x_t, t) is trained to predict the noise (simple MSE loss).
Sampling runs the process backwards:

    x_{t-1} = ( x_t - beta_t / sqrt(1 - abar_t) * eps_theta ) / sqrt(1 - beta_t) + sqrt(beta_t) z.

The optimal noise predictor is  eps*(x, t) = -sqrt(1 - abar_t) grad_x log q_t(x),
i.e. the network learns the *score* of the noised data distribution. For a
Gaussian-mixture data set q_t is again a Gaussian mixture, so eps* is known in
closed form and we can measure how well the network learnt it.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ================================================================ VAE
class VAE(nn.Module):
    def __init__(self, k=2, h=512, D=784):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h, k), nn.Linear(h, k)
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, D))
        self.k = k

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterisation trick
        return self.dec(z), mu, logvar                             # decoder returns Bernoulli logits


def neg_elbo(model, x):
    """Per-sample negative ELBO (nats) for binary x: reconstruction NLL + KL."""
    logits, mu, logvar = model(x)
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)
    kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1)
    return rec + kl, rec, kl


@torch.no_grad()
def iwae_nll(model, x, K=100):
    """Importance-weighted estimate of -log p(x) (Burda et al. 2016):
        log p(x) ~ log 1/K sum_k p(x|z_k) p(z_k) / q(z_k|x),   z_k ~ q(z|x),
    a lower bound on log p(x) that becomes tight as K -> infinity."""
    mu, logvar = model.encode(x)
    std = torch.exp(0.5 * logvar)
    out = []
    for _ in range(K):
        z = mu + std * torch.randn_like(std)
        log_px_z = -F.binary_cross_entropy_with_logits(model.dec(z), x, reduction="none").sum(1)
        log_pz = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).sum(1)
        log_qz = (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(1)
        out.append(log_px_z + log_pz - log_qz)
    lw = torch.stack(out)
    return -(torch.logsumexp(lw, 0) - math.log(K))


def train_vae(model, X, epochs=20, batch=128, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]]
            loss = neg_elbo(model, xb)[0].mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        hist.append(tot / len(X))
    return hist, time.time() - t0


# ================================================================ diffusion
def ring_of_gaussians(n, rng, k=8, radius=2.0, std=0.1):
    """Mixture of k isotropic Gaussians evenly spaced on a circle."""
    means = radius * np.c_[np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)]
    comp = rng.integers(0, k, n)
    return means[comp] + std * rng.normal(size=(n, 2)), means, std


class Schedule:
    def __init__(self, T=100, beta1=1e-4, betaT=0.1):
        self.T = T
        self.beta = torch.linspace(beta1, betaT, T, dtype=torch.float64)
        self.alpha = 1 - self.beta
        self.abar = torch.cumprod(self.alpha, 0)


def time_embedding(t, T, dim=32):
    """Sinusoidal embedding of the (integer) time step, as in transformers."""
    freqs = torch.exp(-math.log(1000.0) * torch.arange(dim // 2) / (dim // 2))
    a = (t.float()[:, None] / T) * 1000 * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class EpsNet(nn.Module):
    def __init__(self, T, h=128, temb=32):
        super().__init__()
        self.T, self.temb = T, temb
        self.net = nn.Sequential(nn.Linear(2 + temb, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, time_embedding(t, self.T, self.temb)], 1))


def train_ddpm(model, sched, X0, steps=6000, batch=512, lr=2e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    X0 = torch.as_tensor(X0, dtype=torch.float32)
    abar = sched.abar.float()
    hist = []
    for s in range(steps):
        x0 = X0[torch.randint(0, len(X0), (batch,), generator=g)]
        t = torch.randint(0, sched.T, (batch,), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        a = abar[t][:, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps          # forward process in closed form
        loss = ((model(xt, t) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); lr_s.step()
        if s % 100 == 0:
            hist.append(loss.item())
    return hist


def exact_eps(x, t, sched, means, std):
    """Optimal noise prediction for Gaussian-mixture data (equal weights).

    q_t = sum_k 1/K N( sqrt(abar) m_k, s2 I ),  s2 = abar std^2 + 1 - abar.
    grad log q_t(x) = sum_k r_k(x) (sqrt(abar) m_k - x) / s2  with responsibilities r_k.
    eps* = -sqrt(1 - abar) grad log q_t."""
    x = np.asarray(x, float)
    ab = sched.abar[t].numpy()[:, None] if np.ndim(t) else float(sched.abar[t])
    ab = np.broadcast_to(ab, (len(x), 1)) if np.ndim(ab) == 0 else ab
    s2 = ab * std ** 2 + 1 - ab
    mu = np.sqrt(ab)[:, :, None] * means.T[None]            # [N, 2, K]
    d2 = ((x[:, :, None] - mu) ** 2).sum(1)                 # [N, K]
    logr = -d2 / (2 * s2)
    r = np.exp(logr - logr.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    score = ((mu - x[:, :, None]) * r[:, None]).sum(2) / s2
    return -np.sqrt(1 - ab) * score


@torch.no_grad()
def ddpm_sample(eps_fn, sched, n, seed=0, keep=()):
    """Ancestral sampling; eps_fn(x [n,2] float64 tensor, t int) -> eps. Returns final
    samples and the intermediate states at the steps listed in ``keep``."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g, dtype=torch.float64)
    traj = {}
    for t in range(sched.T - 1, -1, -1):
        if t + 1 in keep or (t == sched.T - 1 and sched.T in keep):
            traj[t + 1] = x.clone().numpy()
        e = eps_fn(x, t)
        b, a, ab = sched.beta[t], sched.alpha[t], sched.abar[t]
        x = (x - b / torch.sqrt(1 - ab) * e) / torch.sqrt(a)
        if t > 0:
            x = x + torch.sqrt(b) * torch.randn(x.shape, generator=g, dtype=torch.float64)
    traj[0] = x.numpy()
    return x.numpy(), traj


def mmd_rbf(A, B, bandwidths=(0.05, 0.2, 1.0)):
    """Squared maximum mean discrepancy with a sum of RBF kernels (unbiased form).
    ~0 when the two samples come from the same distribution."""
    A, B = torch.as_tensor(A, dtype=torch.float64), torch.as_tensor(B, dtype=torch.float64)

    def k(X, Y):
        d2 = torch.cdist(X, Y) ** 2
        return sum(torch.exp(-d2 / (2 * h * h)) for h in bandwidths)

    m, n = len(A), len(B)
    Kaa, Kbb, Kab = k(A, A), k(B, B), k(A, B)
    return float((Kaa.sum() - Kaa.diag().sum()) / (m * (m - 1)) + (Kbb.sum() - Kbb.diag().sum()) / (n * (n - 1))
                 - 2 * Kab.mean())


def mode_stats(S, means, std, n_sigma=3):
    """Fraction of samples within n_sigma * std of some mode, and how many modes are hit."""
    d = np.sqrt(((S[:, None] - means[None]) ** 2).sum(-1))
    near = d.min(1) < n_sigma * std
    hit = np.unique(d.argmin(1)[near])
    return float(near.mean()), int(len(hit))
