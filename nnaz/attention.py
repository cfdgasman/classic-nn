"""Lesson 08: attention from scratch and a tiny character-level GPT.

Scaled dot-product attention. Each position emits a query q, a key k and a value v
(linear projections of its embedding). Position i attends to position j with weight

    A_ij = softmax_j( q_i . k_j / sqrt(d_k) + M_ij ),      out_i = sum_j A_ij v_j,

where the mask M_ij = -inf for j > i makes it *causal* (no peeking at the future).
The 1/sqrt(d_k) keeps the logits' variance ~1 so the softmax does not saturate.
Multi-head attention runs h such maps in parallel on d_model/h-dim slices and
concatenates the results, letting different heads specialise.

A GPT block (pre-LayerNorm variant):
    x = x + MultiHeadAttention(LayerNorm(x))      # communicate between positions
    x = x + MLP(LayerNorm(x))                     # compute per position
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------- from scratch
def attention_numpy(Q, K, V, causal=False):
    """Q, K: [T, d_k], V: [T, d_v] -> (out [T, d_v], weights [T, T])."""
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    if causal:
        S = S + np.triu(np.full(S.shape, -np.inf), k=1)
    S = S - S.max(-1, keepdims=True)
    A = np.exp(S)
    A /= A.sum(-1, keepdims=True)
    return A @ V, A


class ScratchMHA(nn.Module):
    """Multi-head self-attention written out with plain tensor ops."""

    def __init__(self, d_model, n_head, causal=True):
        super().__init__()
        assert d_model % n_head == 0
        self.h, self.dk, self.causal = n_head, d_model // n_head, causal
        self.qkv = nn.Linear(d_model, 3 * d_model)   # the three projections fused
        self.proj = nn.Linear(d_model, d_model)
        self.last_attn = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, h, T, dk]: each head works on its own slice
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        att = q @ k.transpose(-2, -1) / math.sqrt(self.dk)          # [B, h, T, T]
        if self.causal:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(-1)
        self.last_attn = att.detach()
        y = (att @ v).transpose(1, 2).reshape(B, T, C)               # re-join the heads
        return self.proj(y)


# ---------------------------------------------------------------- GPT
class Block(nn.Module):
    def __init__(self, d, h, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = ScratchMHA(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, block_size=64, d=128, n_layer=4, n_head=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)      # learnt absolute positions
        self.blocks = nn.ModuleList([Block(d, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight           # weight tying: in- and output embeddings shared
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, generator=None):
        """Autoregressive sampling: predict, sample, append, repeat."""
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -self.block_size:])
            p = F.softmax(logits[:, -1] / temperature, -1)
            idx = torch.cat([idx, torch.multinomial(p, 1, generator=generator)], 1)
        self.train()
        return idx


class CharData:
    def __init__(self, text, split=0.9):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(split * len(data))
        self.train, self.val = data[:n], data[n:]

    @property
    def vocab(self):
        return len(self.chars)

    def encode(self, s):
        return torch.tensor([[self.stoi[c] for c in s]], dtype=torch.long)

    def decode(self, t):
        return "".join(self.chars[i] for i in t)

    def batch(self, split, B, T, g):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - T - 1, (B,), generator=g)
        x = torch.stack([d[i:i + T] for i in ix])
        return x, torch.stack([d[i + 1:i + T + 1] for i in ix])


def ngram_baselines(data: CharData):
    """Cross-entropy (nats/char) on the validation text of a unigram and an
    add-one-smoothed bigram model estimated on the training text."""
    V = data.vocab
    tr, va = data.train.numpy(), data.val.numpy()
    uni = np.bincount(tr, minlength=V) + 1.0
    uni /= uni.sum()
    big = np.ones((V, V))
    np.add.at(big, (tr[:-1], tr[1:]), 1)
    big /= big.sum(1, keepdims=True)
    return dict(uniform=float(np.log(V)), unigram=float(-np.log(uni[va]).mean()),
                bigram=float(-np.log(big[va[:-1], va[1:]]).mean()))


@torch.no_grad()
def estimate_loss(model, data, B, T, iters=20, seed=123):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = float(np.mean([model(*data.batch(split, B, T, g))[1].item() for _ in range(iters)]))
    model.train()
    return out


def train_gpt(model, data, steps=2500, B=32, lr=2e-3, eval_every=250, seed=0, sample_every=0, prompt="\n"):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    T = model.block_size
    hist = {"step": [], "train": [], "val": [], "samples": []}
    t0 = time.time()
    for s in range(steps + 1):
        if s % eval_every == 0 or s == steps:
            est = estimate_loss(model, data, B, T)
            hist["step"].append(s); hist["train"].append(est["train"]); hist["val"].append(est["val"])
            print(f"    step {s:5d}  train {est['train']:.3f}  val {est['val']:.3f}  ({time.time() - t0:.0f}s)")
        if sample_every and s % sample_every == 0:
            gs = torch.Generator().manual_seed(7)
            hist["samples"].append((s, data.decode(model.generate(data.encode(prompt), 160, generator=gs)[0].tolist())))
        if s == steps:
            break
        x, y = data.batch("train", B, T, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    hist["time"] = time.time() - t0
    return hist
