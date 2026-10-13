"""Lesson 06: recurrent networks (RNN, LSTM) for time-series forecasting.

A recurrent net reads a sequence one step at a time and carries a *state*:

    vanilla RNN:  h_t = tanh(W_x x_t + W_h h_{t-1} + b)

Back-propagating through T steps multiplies T Jacobians dh_t/dh_{t-1}; their
product tends to vanish or explode, so vanilla RNNs struggle with long memory.
The LSTM (Hochreiter & Schmidhuber 1997) adds a cell state c_t updated
*additively* through gates (sigma = logistic sigmoid, * = element-wise):

    f_t = sigma(W_f [h_{t-1}, x_t] + b_f)        forget gate
    i_t = sigma(W_i [h_{t-1}, x_t] + b_i)        input gate
    g_t = tanh (W_g [h_{t-1}, x_t] + b_g)        candidate
    o_t = sigma(W_o [h_{t-1}, x_t] + b_o)        output gate
    c_t = f_t * c_{t-1} + i_t * g_t              (gradient highway when f ~ 1)
    h_t = o_t * tanh(c_t)
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def make_windows(series, p):
    """All (window of p values, next value) pairs of a 1-D series."""
    idx = np.arange(len(series) - p)[:, None] + np.arange(p)[None]
    return series[idx], series[p:]


# ---------------------------------------------------------------- baselines
def fit_linear_ar(series, p, ridge=1e-6):
    """Linear autoregressive model x_{t} = a . x_{t-p:t} + c, solved *exactly*
    by (ridge-regularised) least squares - the optimal linear one-step predictor."""
    Xw, y = make_windows(series, p)
    A = np.c_[Xw, np.ones(len(Xw))]
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)
    return coef


def ar_predict(coef, windows):
    return windows @ coef[:-1] + coef[-1]


# ---------------------------------------------------------------- networks
class SeqRegressor(nn.Module):
    """Read a window with an RNN/LSTM and map the last hidden state to x_{t+1}."""

    def __init__(self, cell: str = "lstm", hidden: int = 64, layers: int = 1):
        super().__init__()
        Cell = {"lstm": nn.LSTM, "rnn": nn.RNN, "gru": nn.GRU}[cell]
        self.rnn = Cell(1, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: [B, T]
        out, _ = self.rnn(x.unsqueeze(-1))  # out: [B, T, H]
        # residual form: predict the *increment* over the last observed value
        return x[:, -1] + self.head(out[:, -1]).squeeze(-1)


def train_seq(model, Xw, y, Xv, yv, steps=3000, batch=128, lr=3e-3, seed=0, log=500):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xw, y = torch.as_tensor(Xw, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    Xv, yv = torch.as_tensor(Xv, dtype=torch.float32), torch.as_tensor(yv, dtype=torch.float32)
    hist = {"step": [], "train": [], "val": []}
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, len(Xw), (batch,), generator=g)
        opt.zero_grad()
        loss = ((model(Xw[idx]) - y[idx]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guards against exploding gradients
        opt.step(); sched.step()
        if s % 100 == 0 or s == steps - 1:
            with torch.no_grad():
                v = ((model(Xv) - yv) ** 2).mean().item()
            hist["step"].append(s); hist["train"].append(loss.item()); hist["val"].append(v)
            if log and s % log == 0:
                print(f"    step {s:5d}  train mse {loss.item():.4f}  val mse {v:.4f}")
    hist["time"] = time.time() - t0
    return hist


@torch.no_grad()
def net_predict(model, windows):
    model.eval()
    return model(torch.as_tensor(windows, dtype=torch.float32)).numpy()


def rollout(predict_fn, start_windows, horizon):
    """Closed-loop (autoregressive) forecast: feed each prediction back in.
    start_windows: [M, p] -> returns [M, horizon]."""
    w = start_windows.copy()
    out = np.zeros((len(w), horizon))
    for h in range(horizon):
        nxt = predict_fn(w)
        out[:, h] = nxt
        w = np.c_[w[:, 1:], nxt]
    return out
