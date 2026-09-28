"""Lesson 12: the Differentiable Neural Computer (Graves et al., Nature 2016).

A DNC is a neural network (the *controller*, here an LSTM) coupled to an external
memory matrix M (N slots x W words) that it reads and writes through
*differentiable attention*, so the whole system trains end-to-end with backprop.
At every time step the controller emits an "interface vector" that parametrises:

content addressing  c = softmax( beta * cosine(M[i], key) )        (find by similarity)
dynamic allocation  u <- (u + w_w - u*w_w) * prod_r (1 - f_r w_r)   usage of each slot
                    a[phi_j] = (1 - u[phi_j]) prod_{i<j} u[phi_i]   (phi = slots sorted by usage)
write weighting     w_w = g_w ( g_a a + (1 - g_a) c_w )
write               M <- M * (1 - w_w e^T) + w_w v^T                (erase, then add)
temporal links      L[i,j] <- (1 - w_w[i] - w_w[j]) L[i,j] + w_w[i] p[j]   ("i written after j")
                    p <- (1 - sum w_w) p + w_w                      (precedence)
read weighting      w_r = pi_back (L^T w_r) + pi_content c_r + pi_forward (L w_r)
read                r = w_r^T M

Allocation lets it write to fresh slots, the link matrix lets it replay what was
written *in order* - exactly what the copy task below needs, and what an LSTM,
whose memory is a fixed-size state vector, lacks when sequences get longer.
"""
from __future__ import annotations

import time

import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-6


def oneplus(x):
    return 1 + F.softplus(x)


def cosine_weights(M, keys, beta):
    """M: [B,N,W], keys: [B,H,W], beta: [B,H] -> [B,H,N]."""
    sim = torch.einsum("bhw,bnw->bhn", F.normalize(keys, dim=-1, eps=EPS), F.normalize(M, dim=-1, eps=EPS))
    return torch.softmax(beta.unsqueeze(-1) * sim, -1)


def allocation(u):
    """Allocation weighting from usage u [B,N] (differentiable except through the sort)."""
    u = EPS + (1 - EPS) * u
    su, phi = torch.sort(u, dim=1)
    prod_excl = torch.cumprod(torch.cat([torch.ones_like(su[:, :1]), su[:, :-1]], 1), 1)
    a_sorted = (1 - su) * prod_excl
    return torch.zeros_like(u).scatter(1, phi, a_sorted)


class DNC(nn.Module):
    def __init__(self, n_in, n_out, hidden=64, N=32, W=16, R=1):
        super().__init__()
        self.N, self.W, self.R = N, W, R
        self.ctrl = nn.LSTMCell(n_in + R * W, hidden)
        # interface: R read keys, R strengths, write key, strength, erase, write vec,
        #            R free gates, alloc gate, write gate, R x 3 read modes
        self.n_if = R * W + R + W + 1 + W + W + R + 1 + 1 + 3 * R
        self.iface = nn.Linear(hidden, self.n_if)
        self.out = nn.Linear(hidden + R * W, n_out)

    def init_state(self, B):
        z = lambda *s: torch.zeros(B, *s)
        return dict(h=z(self.ctrl.hidden_size), c=z(self.ctrl.hidden_size), M=z(self.N, self.W) + EPS,
                    u=z(self.N), L=z(self.N, self.N), p=z(self.N), ww=z(self.N),
                    wr=z(self.R, self.N), r=z(self.R * self.W))

    def step(self, x, s):
        B, N, W, R = x.shape[0], self.N, self.W, self.R
        h, c = self.ctrl(torch.cat([x, s["r"]], 1), (s["h"], s["c"]))
        xi = self.iface(h)
        i = 0

        def take(n):
            nonlocal i
            v = xi[:, i:i + n]; i += n
            return v

        k_r = take(R * W).view(B, R, W); b_r = oneplus(take(R))
        k_w = take(W).view(B, 1, W); b_w = oneplus(take(1))
        e = torch.sigmoid(take(W)); v = take(W)
        f = torch.sigmoid(take(R)); g_a = torch.sigmoid(take(1)); g_w = torch.sigmoid(take(1))
        pi = torch.softmax(take(3 * R).view(B, R, 3), -1)

        # 1. usage and allocation (memory freed by the previous reads if free gates are open)
        psi = torch.prod(1 - f.unsqueeze(-1) * s["wr"], 1)
        u = (s["u"] + s["ww"] - s["u"] * s["ww"]) * psi
        a = allocation(u)
        # 2. write
        c_w = cosine_weights(s["M"], k_w, b_w)[:, 0]
        ww = g_w * (g_a * a + (1 - g_a) * c_w)
        M = s["M"] * (1 - ww.unsqueeze(2) * e.unsqueeze(1)) + ww.unsqueeze(2) * v.unsqueeze(1)
        # 3. temporal link matrix and precedence
        L = (1 - ww.unsqueeze(2) - ww.unsqueeze(1)) * s["L"] + ww.unsqueeze(2) * s["p"].unsqueeze(1)
        L = L * (1 - torch.eye(N)).unsqueeze(0)
        p = (1 - ww.sum(1, keepdim=True)) * s["p"] + ww
        # 4. read: backward, content, forward modes
        fwd = torch.einsum("bij,brj->bri", L, s["wr"])
        bwd = torch.einsum("bji,brj->bri", L, s["wr"])
        c_r = cosine_weights(M, k_r, b_r)
        wr = pi[..., 0:1] * bwd + pi[..., 1:2] * c_r + pi[..., 2:3] * fwd
        r = torch.einsum("brn,bnw->brw", wr, M).reshape(B, R * W)
        y = self.out(torch.cat([h, r], 1))
        return y, dict(h=h, c=c, M=M, u=u, L=L, p=p, ww=ww, wr=wr, r=r)

    def forward(self, X, record=False):
        """X: [B, T, n_in] -> logits [B, T, n_out] (and optionally the write/read weightings)."""
        s = self.init_state(X.shape[0])
        ys, rec = [], {"ww": [], "wr": []}
        for t in range(X.shape[1]):
            y, s = self.step(X[:, t], s)
            ys.append(y)
            if record:
                rec["ww"].append(s["ww"][0].detach()); rec["wr"].append(s["wr"][0, 0].detach())
        out = torch.stack(ys, 1)
        return (out, rec) if record else out


class LSTMBaseline(nn.Module):
    def __init__(self, n_in, n_out, hidden=128, layers=2):
        super().__init__()
        self.rnn = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, X):
        return self.out(self.rnn(X)[0])


# ---------------------------------------------------------------- the copy task
def copy_batch(B, L, bits, g):
    """Input: L random bit vectors, then a delimiter flag, then L blank steps.
    Target: the same L vectors during the blank steps.  Returns X, Y, mask."""
    T = 2 * L + 1
    seq = (torch.rand(B, L, bits, generator=g) < 0.5).float()
    X = torch.zeros(B, T, bits + 1)
    X[:, :L, :bits] = seq
    X[:, L, bits] = 1.0                       # "now repeat" delimiter
    Y = torch.zeros(B, T, bits)
    Y[:, L + 1:] = seq
    mask = torch.zeros(B, T, 1)
    mask[:, L + 1:] = 1.0
    return X, Y, mask


def train_copy(model, bits=6, steps=4000, B=32, Lmax=10, lr=1e-3, seed=0, log=500):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"step": [], "loss": []}
    t0, run = time.time(), 0.0
    for s in range(steps):
        L = int(torch.randint(1, Lmax + 1, (1,), generator=g))
        X, Y, m = copy_batch(B, L, bits, g)
        logits = model(X)
        loss = (F.binary_cross_entropy_with_logits(logits, Y, reduction="none") * m).sum() / (m.sum() * bits)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        run += loss.item()
        if (s + 1) % 100 == 0:
            hist["step"].append(s + 1); hist["loss"].append(run / 100); run = 0.0
            if log and (s + 1) % log == 0:
                print(f"    step {s + 1:5d}  loss {hist['loss'][-1]:.4f}  ({time.time() - t0:.0f}s)")
    hist["time"] = time.time() - t0
    return hist


@torch.no_grad()
def bit_error_rate(model, L, bits=6, B=200, seed=123):
    g = torch.Generator().manual_seed(seed + L)
    X, Y, m = copy_batch(B, L, bits, g)
    pred = (model(X) > 0).float()
    return float(((pred != Y).float() * m).sum() / (m.sum() * bits))
