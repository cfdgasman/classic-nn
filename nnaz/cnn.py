"""Lesson 05: convolutional networks for images.

A 2-D convolution layer (really a cross-correlation) with kernel K of size
k x k, C_in input and C_out output channels computes

    Y[o, i, j] = b[o] + sum_c sum_{u,v} K[o, c, u, v] * X[c, i + u, j + v].

Compared with a fully connected layer it has two built-in assumptions
("inductive biases") that suit images:
  * locality        - each output looks only at a small k x k neighbourhood,
  * weight sharing  - the same kernel is used at every position, so a feature
                      detector learnt in one place works everywhere
                      (translation equivariance) and the parameter count does
                      not depend on the image size.
Pooling (max over 2 x 2 windows) then gives a little translation invariance
and halves the resolution, so deeper layers see larger regions.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def conv2d_numpy(x, k):
    """Naive 'valid' cross-correlation of one channel, written as loops over the
    output: the definition above, with no tricks. x: [H, W], k: [kh, kw]."""
    H, W = x.shape
    kh, kw = k.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * k)
    return out


def conv2d_im2col(x, K, b):
    """Multi-channel convolution as ONE matrix product (how libraries do it).

    x: [C, H, W], K: [O, C, k, k]. im2col unrolls every k x k x C patch into a
    column, so the convolution becomes  Y = K_flat @ patches  of shapes
    [O, C k k] @ [C k k, H_out W_out].
    """
    C, H, W = x.shape
    O, _, k, _ = K.shape
    Ho, Wo = H - k + 1, W - k + 1
    s = x.strides
    patches = np.lib.stride_tricks.as_strided(x, shape=(C, k, k, Ho, Wo),
                                              strides=(s[0], s[1], s[2], s[1], s[2]))
    cols = patches.reshape(C * k * k, Ho * Wo)
    return (K.reshape(O, -1) @ cols + b[:, None]).reshape(O, Ho, Wo)


class SmallCNN(nn.Module):
    """conv(1->16, 5x5) -> ReLU -> maxpool 2 -> conv(16->32, 5x5) -> ReLU -> maxpool 2
    -> flatten (32*4*4) -> Linear 128 -> ReLU -> dropout -> Linear 10.   (~ 70 k params)

    Shapes: 28x28 -> conv5 -> 24x24 -> pool -> 12x12 -> conv5 -> 8x8 -> pool -> 4x4.
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(0.3)

    def features(self, x):
        """Return the intermediate feature maps (for visualisation)."""
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(F.max_pool2d(a1, 2)))
        return a1, a2

    def forward(self, x):
        a1, a2 = self.features(x)
        h = F.max_pool2d(a2, 2).flatten(1)
        return self.fc2(self.drop(F.relu(self.fc1(h))))


class MLPBaseline(nn.Module):
    """A fully connected net with a similar parameter count, for comparison."""

    def __init__(self, n_classes: int = 10, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class LogReg(nn.Module):
    """Multinomial logistic regression on raw pixels: the classic linear baseline."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.lin = nn.Linear(784, n_classes)

    def forward(self, x):
        return self.lin(x.flatten(1))


def to_tensor_images(x_uint8, mean=None, std=None):
    """uint8 [N,28,28] -> float [N,1,28,28], standardised with the TRAIN mean/std."""
    x = torch.as_tensor(x_uint8, dtype=torch.float32).div_(255.0).unsqueeze(1)
    if mean is None:
        mean, std = x.mean().item(), x.std().item()
    return (x - mean) / std, mean, std


def fit(model, X, y, epochs=2, batch=128, lr=1e-3, Xte=None, yte=None, log_every=100, seed=0):
    """Mini-batch Adam training. Returns history with per-log-step train loss and
    per-epoch test accuracy."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"step": [], "loss": [], "test_acc": [], "time": 0.0}
    t0, step = time.time(), 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        run = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
            if step % log_every == 0:
                hist["step"].append(step); hist["loss"].append(run / log_every); run = 0.0
        if Xte is not None:
            hist["test_acc"].append(accuracy(model, Xte, yte))
    hist["time"] = time.time() - t0
    return hist


@torch.no_grad()
def predict(model, X, batch=2000):
    model.eval()
    return torch.cat([model(X[i:i + batch]).argmax(1) for i in range(0, len(X), batch)])


def accuracy(model, X, y) -> float:
    return (predict(model, X) == y).float().mean().item()


def confusion(y_true, y_pred, n=10):
    M = np.zeros((n, n), dtype=np.int64)
    np.add.at(M, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return M
