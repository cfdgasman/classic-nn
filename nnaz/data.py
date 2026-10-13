"""Datasets used throughout the course.

Toy 2-D data are generated on the fly. MNIST / Fashion-MNIST are downloaded
once by torchvision into ``data/`` and cached as uint8 NumPy arrays, so later
lessons (and the tests) load them in a fraction of a second.
"""
from __future__ import annotations

import numpy as np

from .common import DATA


# ---------------------------------------------------------------- toy 2-D data
def linearly_separable(n: int, rng: np.random.Generator, margin: float = 0.1,
                       w=(1.0, 2.0), b: float = -0.5):
    """Gaussian points labelled by the line w.x + b = 0, with a band of half-width
    ``margin`` around the line removed so the classes are strictly separable."""
    w = np.asarray(w, float) / np.linalg.norm(w)
    X = rng.normal(size=(4 * n, 2))
    s = X @ w + b  # signed distance to the line
    keep = np.abs(s) > margin
    X, s = X[keep][:n], s[keep][:n]
    y = (s > 0).astype(np.int64)
    return X, y


def blobs(n: int, rng: np.random.Generator, sep: float = 1.5):
    """Two overlapping Gaussian blobs (not separable: logistic-regression data)."""
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 2)) + sep * (2 * y[:, None] - 1) * np.array([0.8, 0.6])
    return X, y


def xor(n: int, rng: np.random.Generator, noise: float = 0.15):
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    X += noise * rng.normal(size=X.shape) * 0.3
    return X, y


def spirals(n: int, rng: np.random.Generator, n_classes: int = 3, noise: float = 0.2,
            turns: float = 1.0):
    """The classic CS231n spiral data set: ``n_classes`` interleaved arms."""
    per = n // n_classes
    X = np.zeros((per * n_classes, 2))
    y = np.zeros(per * n_classes, dtype=np.int64)
    for k in range(n_classes):
        r = np.linspace(0.05, 1.0, per)
        t = np.linspace(k * 2 * np.pi / n_classes, k * 2 * np.pi / n_classes
                        + turns * 2 * np.pi, per) + noise * rng.normal(size=per)
        X[k * per:(k + 1) * per] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[k * per:(k + 1) * per] = k
    return X, y


# ---------------------------------------------------------------- image data
def load_mnist(name: str = "MNIST", train: bool = True):
    """Return (images uint8 [N,28,28], labels int64 [N]) for MNIST/FashionMNIST."""
    cache = DATA / f"{name}_{'train' if train else 'test'}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]
    import torchvision

    cls = {"MNIST": torchvision.datasets.MNIST,
           "FashionMNIST": torchvision.datasets.FashionMNIST}[name]
    ds = cls(str(DATA), train=train, download=True)
    x = ds.data.numpy().astype(np.uint8)
    y = ds.targets.numpy().astype(np.int64)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
                   "Shirt", "Sneaker", "Bag", "Boot"]


# ---------------------------------------------------------------- time series
def van_der_pol(n_steps: int, dt: float = 0.05, mu: float = 2.0, x0=(2.0, 0.0),
                noise: float = 0.0, rng: np.random.Generator | None = None):
    """Integrate the Van der Pol oscillator  x'' - mu (1 - x^2) x' + x = 0.

    Uses SciPy's adaptive RK45 with tight tolerances and samples the solution
    every ``dt``; optional Gaussian *observation* noise is added to x.
    Returns (t, x_clean, x_noisy).
    """
    from scipy.integrate import solve_ivp

    t = np.arange(n_steps) * dt
    sol = solve_ivp(lambda _, s: [s[1], mu * (1 - s[0] ** 2) * s[1] - s[0]],
                    (0, t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-10)
    x = sol.y[0]
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        return t, x, x + noise * rng.normal(size=x.shape)
    return t, x, x.copy()


# ---------------------------------------------------------------- text
def load_text(max_chars: int | None = None) -> str:
    """Tiny Shakespeare (public domain; from karpathy/char-rnn)."""
    text = (DATA / "text" / "tiny_shakespeare.txt").read_text()
    return text[:max_chars] if max_chars else text
