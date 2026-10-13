"""Lesson 07: autoencoders - compression, denoising, and the link with PCA/SVD.

An autoencoder is a pair encoder e: R^D -> R^k, decoder d: R^k -> R^D trained to
reproduce its input,  min E |x - d(e(x))|^2,  with a bottleneck k << D.

Linear autoencoder = PCA. For a centred data matrix X = U S V^T (SVD), the best
rank-k reconstruction in the least-squares sense is X V_k V_k^T (Eckart-Young),
with error  sum_{i>k} s_i^2 / N.  A *linear* autoencoder x -> W_d W_e x can do
no better, and (Baldi & Hornik 1989) all its minima span the same subspace as
V_k - but W_d need not be orthonormal, so the individual latent coordinates are
an arbitrary invertible mix of the principal components. We check both facts
with principal angles between the decoder's column space and span(V_k).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------- PCA (exact)
def pca_fit(X):
    """X: [N, D] -> (mean, V (D x D right singular vectors), s singular values)."""
    mu = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt.T, s


def pca_reconstruct(X, mu, V, k):
    Z = (X - mu) @ V[:, :k]        # coordinates in the principal subspace
    return Z @ V[:, :k].T + mu


def principal_angles(A, B):
    """Angles (radians) between the column spaces of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.arccos(np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1, 1))


# ---------------------------------------------------------------- models
class LinearAE(nn.Module):
    def __init__(self, D, k):
        super().__init__()
        self.enc, self.dec = nn.Linear(D, k), nn.Linear(k, D)

    def forward(self, x):
        return self.dec(self.enc(x))


class MLPAE(nn.Module):
    def __init__(self, D, k, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, h), nn.ReLU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, D), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


class ConvAE(nn.Module):
    """Convolutional autoencoder for 28x28 images.

    encoder: 1x28x28 -conv s2-> 16x14x14 -conv s2-> 32x7x7 -linear-> k
    decoder: k -linear-> 32x7x7 -convT s2-> 16x14x14 -convT s2-> 1x28x28
    Strided convolutions downsample; transposed convolutions ("learnable
    upsampling") invert the shape change.
    """

    def __init__(self, k=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 7 * 7, k))
        self.dec = nn.Sequential(
            nn.Linear(k, 32 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


class DenoisingConvNet(nn.Module):
    """A small fully convolutional denoiser with *residual learning* (as in DnCNN,
    Zhang et al. 2017): the network predicts the correction to add to the noisy
    image,  x_clean ~ x_noisy + f(x_noisy),  which is much easier to learn than
    the clean image itself (f starts near 0, i.e. near the identity map)."""

    def __init__(self, c=32, depth=5):
        super().__init__()
        layers = [nn.Conv2d(1, c, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU()]
        last = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last)

    def forward(self, x):
        return (x + self.net(x)).clamp(0, 1)


def train_ae(model, X, target=None, epochs=5, batch=256, lr=1e-3, noise=0.0, seed=0, shape=None):
    """Minimise mean squared reconstruction error. If ``noise`` > 0 the INPUT is
    corrupted with fresh Gaussian noise every batch (denoising autoencoder)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    target = X if target is None else target
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            xb = X[idx]
            if noise:
                xb = (xb + noise * torch.randn(xb.shape, generator=g)).clamp(0, 1)
            if shape:
                xb = xb.view(-1, *shape)
            opt.zero_grad()
            out = model(xb).reshape(len(idx), -1)
            loss = ((out - target[idx].reshape(len(idx), -1)) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / len(X))
    return hist, time.time() - t0


@torch.no_grad()
def reconstruct(model, X, shape=None, batch=5000):
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        if shape:
            xb = xb.view(-1, *shape)
        outs.append(model(xb).reshape(len(xb), -1))
    model.train()
    return torch.cat(outs)


def psnr(a, b):
    """Peak signal-to-noise ratio in dB for images in [0, 1]."""
    mse = float(((np.asarray(a) - np.asarray(b)) ** 2).mean())
    return 10 * np.log10(1.0 / mse)
