"""Lesson 01: the perceptron and the single sigmoid neuron, gradients by hand.

A neuron computes  z = w . x + b  and an output  a = phi(z).

* Perceptron (Rosenblatt 1958): phi = sign, labels y in {-1, +1}.
  On every mistake (y z <= 0) update  w <- w + y x,  b <- b + y.
* Logistic neuron: phi = sigma(z) = 1 / (1 + e^-z), labels y in {0, 1},
  trained by gradient descent on the mean binary cross-entropy

      L(w, b) = -1/N sum_i [ y_i log a_i + (1 - y_i) log(1 - a_i) ] + lam/2 |w|^2.

  Chain rule for one sample, using sigma'(z) = sigma(z)(1 - sigma(z)):

      dL/da = -(y/a) + (1-y)/(1-a) = (a - y) / (a (1 - a))
      da/dz = a (1 - a)
      =>  dL/dz = a - y                  (the famous "prediction minus target")
      =>  dL/dw = (a - y) x,   dL/db = a - y.

  Averaged over the batch:  grad_w = X^T (a - y) / N + lam w.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ perceptron
def perceptron_train(X, y01, epochs: int = 100, rng=None, record: bool = False):
    """Classic mistake-driven perceptron. Returns (w, b, n_mistakes, history)."""
    y = 2 * y01 - 1  # {0,1} -> {-1,+1}
    w, b = np.zeros(X.shape[1]), 0.0
    mistakes, hist = 0, []
    rng = rng or np.random.default_rng(0)
    for _ in range(epochs):
        clean = True
        for i in rng.permutation(len(X)):
            if y[i] * (X[i] @ w + b) <= 0:  # misclassified (or on the boundary)
                w, b = w + y[i] * X[i], b + y[i]
                mistakes += 1
                clean = False
                if record:
                    hist.append((w.copy(), b, i))
        if clean:  # a full pass without mistakes: converged
            break
    return w, b, mistakes, hist


def novikoff_bound(X, y01, w_star, b_star) -> tuple[float, float, float]:
    """Novikoff (1962): the perceptron makes at most (R / gamma)^2 mistakes.

    Inputs are augmented to [x, 1] so the bias is just another weight;
    R = max |[x, 1]|, and gamma is the margin of *any* unit-norm separating
    vector u = [w*, b*] / |[w*, b*]|, i.e. gamma = min_i y_i u . [x_i, 1] > 0.
    Returns (R, gamma, bound).
    """
    y = 2 * y01 - 1
    Xa = np.c_[X, np.ones(len(X))]
    u = np.r_[w_star, b_star]
    u = u / np.linalg.norm(u)
    R = np.linalg.norm(Xa, axis=1).max()
    gamma = float(np.min(y * (Xa @ u)))
    return float(R), gamma, float((R / gamma) ** 2)


# ------------------------------------------------------------------ logistic neuron
def sigmoid(z):
    # numerically stable: never exponentiate a large positive number
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_loss(params, X, y, lam: float = 0.0) -> float:
    w, b = params[:-1], params[-1]
    z = X @ w + b
    # log(1 + e^z) - y z  is BCE written in terms of the logit (stable form)
    return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * w @ w)


def bce_grad(params, X, y, lam: float = 0.0):
    """Hand-derived gradient: [X^T (a - y) / N + lam w,  mean(a - y)]."""
    w, b = params[:-1], params[-1]
    a = sigmoid(X @ w + b)
    r = a - y
    return np.r_[X.T @ r / len(y) + lam * w, r.mean()]


def numerical_grad(f, p, eps: float = 1e-6):
    """Central differences: (f(p + eps e_k) - f(p - eps e_k)) / (2 eps), error O(eps^2)."""
    g = np.zeros_like(p)
    for k in range(len(p)):
        e = np.zeros_like(p)
        e[k] = eps
        g[k] = (f(p + e) - f(p - e)) / (2 * eps)
    return g


def rel_error(a, b) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b))))


def logistic_gd(X, y, lr: float = 0.5, steps: int = 500, lam: float = 0.0, record_every=0):
    """Full-batch gradient descent on the logistic neuron."""
    p = np.zeros(X.shape[1] + 1)
    losses, snaps = [], []
    for k in range(steps):
        losses.append(bce_loss(p, X, y, lam))
        if record_every and k % record_every == 0:
            snaps.append(p.copy())
        p -= lr * bce_grad(p, X, y, lam)
    losses.append(bce_loss(p, X, y, lam))
    return p, np.array(losses), snaps


def logistic_reference(X, y, lam: float):
    """The exact minimiser (the problem is strictly convex for lam > 0) via L-BFGS."""
    from scipy.optimize import minimize

    res = minimize(bce_loss, np.zeros(X.shape[1] + 1), args=(X, y, lam), jac=bce_grad,
                   method="L-BFGS-B", options=dict(gtol=1e-12, ftol=1e-15, maxiter=10_000))
    return res.x, res.fun


def predict(p, X):
    return (X @ p[:-1] + p[-1] > 0).astype(np.int64)
