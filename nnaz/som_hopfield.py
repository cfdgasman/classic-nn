"""Lesson 11: two classic "non-backprop" networks.

Self-organising map (Kohonen 1982)
----------------------------------
A 2-D grid of units, unit j having a prototype (weight) vector w_j in data space.
For every input x:
    1. best-matching unit   b = argmin_j |x - w_j|                 (competition)
    2. neighbourhood        h_j = exp(-|r_j - r_b|^2 / 2 sigma^2)  (r_j = grid position)
    3. update               w_j <- w_j + eta h_j (x - w_j)          (cooperation)
with eta and sigma shrinking over time. Units that are neighbours ON THE GRID end
up with similar prototypes: the map is a topology-preserving, discretised
projection of the data (a nonlinear cousin of PCA / a "k-means with a map").

We also implement the *batch* SOM (every unit moves to the neighbourhood-
weighted mean of the data), which is deterministic and vectorises well.

Hopfield network (Hopfield 1982) - a network with feedback loops
----------------------------------------------------------------
N binary neurons s_i = +-1, all-to-all symmetric weights (no self-loops),
stored patterns xi^mu by the Hebbian rule

    W = (1/N) sum_mu xi^mu xi^mu^T,   W_ii = 0,

and asynchronous dynamics  s_i <- sign( sum_j W_ij s_j ).  (The Hebbian rule assumes nearly
uncorrelated patterns; for correlated ones, e.g. letters that share a white
background, the projection rule below stores them exactly.)  Each update cannot
increase the energy  E(s) = -1/2 s^T W s, so the state slides downhill into a
local minimum - ideally a stored pattern: a content-addressable memory that
completes corrupted inputs. Capacity: about 0.138 N random patterns.

The *modern* (dense) Hopfield network (Ramsauer et al. 2020) uses the update
    s <- X softmax(beta X^T s)       (X = matrix of stored patterns),
which is exactly the attention of lesson 08, with exponentially large capacity.
"""
from __future__ import annotations

import numpy as np


# ================================================================ SOM
class SOM:
    def __init__(self, rows, cols, dim, rng, init_data=None):
        self.rows, self.cols = rows, cols
        self.grid = np.array([(r, c) for r in range(rows) for c in range(cols)], float)
        if init_data is not None:  # linear init along the first two principal axes
            mu = init_data.mean(0)
            _, s, Vt = np.linalg.svd(init_data - mu, full_matrices=False)
            sd = s[:2] / np.sqrt(len(init_data))
            a = (self.grid[:, 0] / max(rows - 1, 1) - 0.5) * 2
            b = (self.grid[:, 1] / max(cols - 1, 1) - 0.5) * 2
            self.W = mu + a[:, None] * sd[0] * Vt[0] + b[:, None] * sd[1] * Vt[1]
        else:
            self.W = rng.uniform(-0.05, 0.05, size=(rows * cols, dim))

    def bmu(self, X):
        # |x - w|^2 = |x|^2 - 2 x.w + |w|^2 ; |x|^2 does not change the argmin
        d = -2 * X @ self.W.T + (self.W ** 2).sum(1)[None]
        return d.argmin(1)

    def neighbourhood(self, b, sigma):
        d2 = ((self.grid[None] - self.grid[b][:, None]) ** 2).sum(-1)   # [N, units]
        return np.exp(-d2 / (2 * sigma ** 2))

    def train_online(self, X, epochs, rng, eta0=0.5, sigma0=None, snapshots=0):
        sigma0 = sigma0 or max(self.rows, self.cols) / 2
        T = epochs * len(X)
        snaps, step = [], 0
        for ep in range(epochs):
            for i in rng.permutation(len(X)):
                frac = step / T
                eta = eta0 * (0.01 / eta0) ** frac          # exponential decay to 0.01
                sigma = sigma0 * (0.5 / sigma0) ** frac     # shrink to half a grid cell
                b = self.bmu(X[i:i + 1])
                h = self.neighbourhood(b, sigma)[0]
                self.W += eta * h[:, None] * (X[i] - self.W)
                if snapshots and step % snapshots == 0:
                    snaps.append(self.W.copy())
                step += 1
        return snaps

    def train_batch(self, X, epochs, sigma0=None, sigma_end=0.7):
        """Batch SOM: w_j = sum_i h_{j,b(i)} x_i / sum_i h_{j,b(i)}."""
        sigma0 = sigma0 or max(self.rows, self.cols) / 2
        for ep in range(epochs):
            sigma = sigma0 * (sigma_end / sigma0) ** (ep / max(epochs - 1, 1))
            b = self.bmu(X)
            # neighbourhood between units (units x units), then aggregate the data per BMU
            H = self.neighbourhood(np.arange(len(self.W)), sigma)
            counts = np.bincount(b, minlength=len(self.W)).astype(float)
            sums = np.zeros_like(self.W)
            np.add.at(sums, b, X)
            num, den = H @ sums, H @ counts
            ok = den > 1e-12
            self.W[ok] = num[ok] / den[ok, None]

    def quantisation_error(self, X):
        return float(np.sqrt(((X - self.W[self.bmu(X)]) ** 2).sum(1)).mean())

    def topographic_error(self, X):
        """Fraction of samples whose best and second-best units are NOT grid neighbours."""
        d = -2 * X @ self.W.T + (self.W ** 2).sum(1)[None]
        two = np.argsort(d, 1)[:, :2]
        gd = np.abs(self.grid[two[:, 0]] - self.grid[two[:, 1]]).max(1)
        return float((gd > 1).mean())

    def u_matrix(self):
        """Mean distance of each unit's prototype to its 4 grid neighbours."""
        U = np.zeros((self.rows, self.cols))
        Wg = self.W.reshape(self.rows, self.cols, -1)
        for r in range(self.rows):
            for c in range(self.cols):
                nb = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]
                U[r, c] = np.mean([np.linalg.norm(Wg[r, c] - Wg[a, b]) for a, b in nb])
        return U


def kmeans(X, k, rng, iters=50):
    """Lloyd's algorithm with k-means++ seeding: the topology-free baseline."""
    C = [X[rng.integers(len(X))]]
    d = ((X - C[0]) ** 2).sum(1)             # squared distance to the nearest centre so far
    for _ in range(k - 1):
        C.append(X[rng.choice(len(X), p=d / d.sum())])
        d = np.minimum(d, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        lab = (-2 * X @ C.T + (C ** 2).sum(1)[None]).argmin(1)
        for j in range(k):
            if np.any(lab == j):
                C[j] = X[lab == j].mean(0)
    return C


# ================================================================ Hopfield
def hebbian_weights(P):
    """P: [n_patterns, N] of +-1."""
    N = P.shape[1]
    W = P.T @ P / N
    np.fill_diagonal(W, 0.0)
    return W


def projection_weights(P):
    """Pseudo-inverse ("projection") rule, Personnaz et al. 1985:
        W = P^T (P P^T)^-1 P,  W_ii = 0.
    W projects onto the span of the stored patterns, so every stored pattern is an
    exact fixed point even when the patterns are strongly correlated - where the
    Hebbian rule (which assumes nearly orthogonal patterns) produces mixtures."""
    W = P.T @ np.linalg.pinv(P @ P.T) @ P
    np.fill_diagonal(W, 0.0)
    return W


def energy(W, s):
    return float(-0.5 * s @ W @ s)


def hopfield_recall(W, s, rng, max_sweeps=50, record=False):
    """Asynchronous updates in random order until a full sweep changes nothing."""
    s = s.copy()
    hist = [s.copy()] if record else None
    energies = [energy(W, s)]
    for _ in range(max_sweeps):
        changed = 0
        for i in rng.permutation(len(s)):
            new = 1.0 if W[i] @ s >= 0 else -1.0
            if new != s[i]:
                s[i] = new
                changed += 1
                if record:
                    hist.append(s.copy())
        energies.append(energy(W, s))
        if changed == 0:
            break
    return s, energies, hist


def modern_hopfield_recall(P, s, beta=8.0, steps=3):
    """Dense associative memory: s <- P^T softmax(beta P s / sqrt N)... = attention over stored patterns."""
    N = P.shape[1]
    for _ in range(steps):
        a = beta * P @ s / np.sqrt(N)
        a = np.exp(a - a.max()); a /= a.sum()
        s = P.T @ a
    return np.sign(s + 1e-12)
