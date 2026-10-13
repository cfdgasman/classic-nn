"""Lessons 02-03: a multilayer perceptron written from scratch in NumPy.

Everything is a *layer* with two methods:

    forward(x)      -> y          (caches what backward will need)
    backward(dy)    -> dx         (given dL/dy, returns dL/dx and stores dL/dparams)

Backpropagation is then nothing more than calling ``backward`` on the layers in
reverse order: each layer applies the chain rule locally,

    dL/dx = (dy/dx)^T dL/dy,

so the gradient of the loss with respect to *every* parameter is obtained in a
single backward sweep, at roughly the cost of one forward pass.

Shapes: a mini-batch is a matrix X with one sample per row, shape [N, D].
"""
from __future__ import annotations

import numpy as np


class Layer:
    params: dict
    grads: dict
    training = True

    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dy):
        raise NotImplementedError


# ---------------------------------------------------------------- linear layer
class Linear(Layer):
    """y = x W + b.

    Backward (with dy = dL/dy of shape [N, out]):
        dW = x^T dy          [in, out]     (sums over the batch automatically)
        db = sum_n dy        [out]
        dx = dy W^T          [N, in]
    """

    def __init__(self, n_in, n_out, rng, init: str = "he", bias: bool = True):
        super().__init__()
        scale = {"he": np.sqrt(2.0 / n_in),            # keeps ReLU variance ~ constant
                 "xavier": np.sqrt(1.0 / n_in),        # LeCun/Xavier for tanh
                 "small": 0.01}.get(init, init)        # too small: signal vanishes
        self.params["W"] = rng.normal(0.0, scale, size=(n_in, n_out))
        if bias:
            self.params["b"] = np.zeros(n_out)

    def forward(self, x):
        self.x = x
        y = x @ self.params["W"]
        return y + self.params["b"] if "b" in self.params else y

    def backward(self, dy):
        self.grads["W"] = self.x.T @ dy
        if "b" in self.params:
            self.grads["b"] = dy.sum(0)
        return dy @ self.params["W"].T


# ---------------------------------------------------------------- activations
class ReLU(Layer):
    """y = max(0, x);  dy/dx = 1[x > 0]  (element-wise, so backward is a mask)."""

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask


class Tanh(Layer):
    """y = tanh(x);  dy/dx = 1 - y^2."""

    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dy):
        return dy * (1.0 - self.y ** 2)


class Sigmoid(Layer):
    """y = 1 / (1 + e^-x);  dy/dx = y (1 - y) <= 1/4, hence vanishing gradients."""

    def forward(self, x):
        self.y = 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free identity
        return self.y

    def backward(self, dy):
        return dy * self.y * (1.0 - self.y)


ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}


# ---------------------------------------------------------------- batch norm
class BatchNorm(Layer):
    """Batch normalisation (Ioffe & Szegedy 2015) for [N, D] activations.

    Training:  mu = mean_n x,  var = mean_n (x - mu)^2,
               xh = (x - mu) / sqrt(var + eps),   y = gamma xh + beta.
    Inference: uses running averages of mu and var instead of batch statistics.

    Backward. Because mu and var depend on *every* sample, dL/dx_n gets three
    contributions; after simplification (per feature, N samples):

        dxh = dy * gamma
        dx  = (1 / (N sigma)) * ( N dxh - sum(dxh) - xh * sum(dxh * xh) )
    """

    def __init__(self, dim, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.params["gamma"] = np.ones(dim)
        self.params["beta"] = np.zeros(dim)
        self.running_mean, self.running_var = np.zeros(dim), np.ones(dim)
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        if self.training:
            mu, var = x.mean(0), x.var(0)
            m = self.momentum
            self.running_mean = m * self.running_mean + (1 - m) * mu
            self.running_var = m * self.running_var + (1 - m) * var
        else:
            mu, var = self.running_mean, self.running_var
        self.sigma = np.sqrt(var + self.eps)
        self.xh = (x - mu) / self.sigma
        return self.params["gamma"] * self.xh + self.params["beta"]

    def backward(self, dy):
        N = dy.shape[0]
        self.grads["gamma"] = (dy * self.xh).sum(0)
        self.grads["beta"] = dy.sum(0)
        dxh = dy * self.params["gamma"]
        if not self.training:  # statistics are constants at inference time
            return dxh / self.sigma
        return (N * dxh - dxh.sum(0) - self.xh * (dxh * self.xh).sum(0)) / (N * self.sigma)


# ---------------------------------------------------------------- dropout
class Dropout(Layer):
    """Inverted dropout: zero each unit with probability p during training and
    scale the survivors by 1/(1-p) so the expected activation is unchanged;
    at test time it is the identity. Backward applies the same mask."""

    def __init__(self, p: float, rng):
        super().__init__()
        self.p, self.rng = p, rng

    def forward(self, x):
        if not self.training or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dy):
        return dy if self.mask is None else dy * self.mask


# ---------------------------------------------------------------- loss
def softmax(z):
    z = z - z.max(1, keepdims=True)  # shift for stability (softmax is shift-invariant)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """Mean cross-entropy  L = -1/N sum_n log softmax(z_n)[y_n]  and dL/dlogits.

    The gradient is again "prediction minus target":  (softmax(z) - onehot(y)) / N.
    """
    N = len(y)
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = -logp[np.arange(N), y].mean()
    d = np.exp(logp)
    d[np.arange(N), y] -= 1.0
    return loss, d / N


def mse_loss(pred, target):
    diff = pred - target
    return 0.5 * np.mean(np.sum(diff ** 2, axis=1)), diff / len(pred)


# ---------------------------------------------------------------- the network
class MLP:
    """A stack of layers: [Linear -> (BatchNorm) -> act -> (Dropout)] x depth -> Linear."""

    def __init__(self, sizes, rng, act: str = "relu", init: str = "he",
                 batchnorm: bool = False, dropout: float = 0.0):
        self.layers: list[Layer] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            hidden = i < len(sizes) - 2
            # a bias right before BatchNorm is redundant (BN subtracts the mean
            # and adds its own shift beta), and its gradient is exactly zero
            self.layers.append(Linear(a, b, rng, init, bias=not (hidden and batchnorm)))
            if hidden:
                if batchnorm:
                    self.layers.append(BatchNorm(b))
                self.layers.append(ACTIVATIONS[act]())
                if dropout > 0:
                    self.layers.append(Dropout(dropout, rng))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy):
        for layer in reversed(self.layers):  # the chain rule, applied right to left
            dy = layer.backward(dy)
        return dy

    def train(self, mode: bool = True):
        for layer in self.layers:
            layer.training = mode

    def param_list(self):
        """(layer, name) pairs: optimisers update layer.params[name] in place."""
        return [(layer, k) for layer in self.layers for k in layer.params]

    def loss_and_grads(self, X, y, weight_decay: float = 0.0):
        logits = self.forward(X)
        loss, dlogits = softmax_cross_entropy(logits, y)
        self.backward(dlogits)
        if weight_decay:  # L2 penalty on weight matrices only (not biases)
            for layer, k in self.param_list():
                if k == "W":
                    loss += 0.5 * weight_decay * np.sum(layer.params[k] ** 2)
                    layer.grads[k] = layer.grads[k] + weight_decay * layer.params[k]
        return loss

    def predict(self, X):
        was = self.layers[0].training
        self.train(False)
        out = self.forward(X).argmax(1)
        self.train(was)
        return out


# ---------------------------------------------------------------- gradient check
def gradient_check(model: MLP, X, y, eps: float = 1e-6, weight_decay: float = 0.0,
                   max_per_param: int | None = None, rng=None):
    """Compare backprop with central finite differences for every parameter.

    Relative error  |g_bp - g_fd| / max(|g_bp| + |g_fd|, tiny).  In float64 with
    eps = 1e-6 the truncation error is O(eps^2) ~ 1e-12 and round-off ~ 1e-16/eps
    ~ 1e-10, so a correct implementation gives relative errors << 1e-6.

    Dropout masks must be frozen during the check, so we reseed the dropout RNG
    identically before each forward pass.
    """
    drop = [layer for layer in model.layers if isinstance(layer, Dropout)]
    states = [d.rng.bit_generator.state for d in drop]

    def restore():
        for d, s in zip(drop, states):
            d.rng.bit_generator.state = s

    restore()
    model.loss_and_grads(X, y, weight_decay)
    # Snapshot ALL backprop gradients now: every finite-difference evaluation
    # below re-runs backward and overwrites layer.grads (at a perturbed point!).
    analytic = [layer.grads[k].copy() for layer, k in model.param_list()]
    worst = 0.0
    report = {}
    rng = rng or np.random.default_rng(0)
    for (layer, k), G in zip(model.param_list(), analytic):
        P = layer.params[k]
        idx = list(np.ndindex(P.shape))
        if max_per_param and len(idx) > max_per_param:
            idx = [idx[i] for i in rng.choice(len(idx), max_per_param, replace=False)]
        g_bp, g_fd = [], []
        for ix in idx:
            old = P[ix]
            P[ix] = old + eps
            restore()
            lp = model.loss_and_grads(X, y, weight_decay)
            P[ix] = old - eps
            restore()
            lm = model.loss_and_grads(X, y, weight_decay)
            P[ix] = old
            g_fd.append((lp - lm) / (2 * eps))
            g_bp.append(G[ix])
        err = _rel(np.array(g_bp), np.array(g_fd))
        li = model.layers.index(layer)
        report[f"{li}:{type(layer).__name__}.{k}"] = err
        worst = max(worst, err)
    restore()
    model.loss_and_grads(X, y, weight_decay)  # leave consistent grads behind
    return worst, report


def _rel(a, b) -> float:
    """Norm-wise relative error ||a - b|| / (||a|| + ||b||).

    Element-wise relative errors are ill-conditioned for entries that are
    themselves ~1e-7 (round-off in the loss, ~1e-16 / eps, dominates there), so
    we compare whole gradient tensors. Tensors whose true gradient is exactly
    zero fall back to the absolute error."""
    den = np.linalg.norm(a) + np.linalg.norm(b)
    num = np.linalg.norm(a - b)
    return float(num / den) if den > 1e-8 else float(num)


def input_gradient_check(model: MLP, X, y, eps=1e-6):
    """Also check dL/dX (the gradient that flows *out* of the network)."""
    drop = [layer for layer in model.layers if isinstance(layer, Dropout)]
    states = [d.rng.bit_generator.state for d in drop]

    def loss_at(Z):
        for d, s in zip(drop, states):  # identical dropout masks every time
            d.rng.bit_generator.state = s
        return softmax_cross_entropy(model.forward(Z), y)

    _, d = loss_at(X)
    dX = model.backward(d)
    fd = np.zeros_like(X)
    for ix in np.ndindex(X.shape):
        old = X[ix]
        X[ix] = old + eps
        lp = loss_at(X)[0]
        X[ix] = old - eps
        lm = loss_at(X)[0]
        X[ix] = old
        fd[ix] = (lp - lm) / (2 * eps)
    return _rel(dX, fd)


# ---------------------------------------------------------------- optimisers
class SGD:
    """Plain SGD, optionally with (heavy-ball) momentum:
        v <- mu v - lr g;   theta <- theta + v."""

    def __init__(self, params, lr, momentum: float = 0.0):
        self.params, self.lr, self.mu = params, lr, momentum
        self.v = [np.zeros_like(layer.params[k]) for layer, k in params]

    def step(self):
        for (layer, k), v in zip(self.params, self.v):
            v *= self.mu
            v -= self.lr * layer.grads[k]
            layer.params[k] += v


class Adam:
    """Adam (Kingma & Ba 2015): per-parameter step sizes from running moments.
        m <- b1 m + (1-b1) g,   s <- b2 s + (1-b2) g^2
        m_hat = m / (1-b1^t),   s_hat = s / (1-b2^t)     (bias correction)
        theta <- theta - lr m_hat / (sqrt(s_hat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, (self.b1, self.b2), self.eps = params, lr, betas, eps
        self.m = [np.zeros_like(layer.params[k]) for layer, k in params]
        self.s = [np.zeros_like(layer.params[k]) for layer, k in params]
        self.t = 0

    def step(self):
        self.t += 1
        for (layer, k), m, s in zip(self.params, self.m, self.s):
            g = layer.grads[k]
            m *= self.b1; m += (1 - self.b1) * g
            s *= self.b2; s += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            sh = s / (1 - self.b2 ** self.t)
            layer.params[k] -= self.lr * mh / (np.sqrt(sh) + self.eps)


# ---------------------------------------------------------------- LR schedules
def lr_schedule(kind: str, base: float, step: int, total: int, warmup: int = 0) -> float:
    """Learning rate at ``step`` (0-based) of ``total``."""
    if warmup and step < warmup:  # linear warm-up avoids early instability
        return base * (step + 1) / warmup
    s, T = step - warmup, max(1, total - warmup)
    if kind == "constant":
        return base
    if kind == "step":            # divide by 10 at 50% and 75% of training
        return base * (0.1 ** ((s >= T // 2) + (s >= 3 * T // 4)))
    if kind == "cosine":          # smooth decay to zero (Loshchilov & Hutter 2017)
        return base * 0.5 * (1 + np.cos(np.pi * s / T))
    if kind == "exp":
        return base * 0.01 ** (s / T)
    raise ValueError(kind)


# ---------------------------------------------------------------- training loop
def train(model: MLP, X, y, Xval=None, yval=None, epochs=100, batch=64, opt="adam", lr=1e-2,
          momentum=0.9, schedule="constant", warmup=0, weight_decay=0.0, rng=None,
          record_every: int = 0):
    """Mini-batch training; returns a history dict (loss/acc per epoch)."""
    rng = rng or np.random.default_rng(0)
    params = model.param_list()
    optim = Adam(params, lr) if opt == "adam" else SGD(params, lr, momentum if opt == "momentum" else 0.0)
    n_batches = int(np.ceil(len(X) / batch))
    total = epochs * n_batches
    hist = {"loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "snapshots": []}
    step = 0
    for ep in range(epochs):
        model.train(True)
        perm = rng.permutation(len(X))
        ep_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch:(b + 1) * batch]
            optim.lr = lr_schedule(schedule, lr, step, total, warmup)
            loss = model.loss_and_grads(X[idx], y[idx], weight_decay)
            if not np.isfinite(loss):
                raise FloatingPointError(f"loss diverged at epoch {ep}")
            optim.step()
            ep_loss += loss * len(idx)
            step += 1
        model.train(False)
        hist["loss"].append(ep_loss / len(X))
        hist["lr"].append(optim.lr)
        hist["train_acc"].append((model.predict(X) == y).mean())
        if Xval is not None:
            hist["val_loss"].append(softmax_cross_entropy(model.forward(Xval), yval)[0])
            hist["val_acc"].append((model.predict(Xval) == yval).mean())
        if record_every and ep % record_every == 0:
            hist["snapshots"].append([layer.params[k].copy() for layer, k in params])
    model.train(False)
    return hist
