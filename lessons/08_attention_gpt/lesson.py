"""Lesson 08 - Attention from scratch, then a tiny character-level GPT.

Questions this lesson answers
-----------------------------
* What is attention, mathematically? (A soft, content-based lookup table.)
* Why the 1/sqrt(d) scaling, the causal mask, multiple heads?
* Is my from-scratch multi-head attention identical to PyTorch's? (Checked to 1e-6.)
* How does a GPT turn "predict the next character" into text generation, and how much
  better than n-gram statistics is it?

Data: Tiny Shakespeare (1.1 M characters, public domain), 90 % train / 10 % validation.

Run:  python lessons/08_attention_gpt/lesson.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import nn

from nnaz.attention import CharData, ScratchMHA, TinyGPT, attention_numpy, ngram_baselines, train_gpt
from nnaz.common import (banner, lesson_dir, save_gif, save_results, savefig, seed_everything,
                         set_torch_threads, setup_matplotlib)
from nnaz.data import load_text

LESSON = 8
plt = setup_matplotlib()


def scratch_part():
    banner("1. Attention from scratch, checked against PyTorch")
    rng = np.random.default_rng(0)
    T, d = 6, 4
    Q, K, V = rng.normal(size=(3, T, d))
    out, A = attention_numpy(Q, K, V, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(torch.from_numpy(a)[None] for a in (Q, K, V)), is_causal=True)[0].numpy()
    e_np = float(np.abs(out - ref).max())
    print(f"  NumPy attention vs F.scaled_dot_product_attention: max diff {e_np:.1e}")
    print(f"  rows of A sum to 1: {np.allclose(A.sum(1), 1)};  upper triangle zero (causal): {np.allclose(np.triu(A, 1), 0)}")

    # our multi-head module vs nn.MultiheadAttention with the same weights
    torch.manual_seed(0)
    D, H = 32, 4
    mine = ScratchMHA(D, H, causal=True)
    ref_mha = nn.MultiheadAttention(D, H, batch_first=True)
    with torch.no_grad():
        ref_mha.in_proj_weight.copy_(mine.qkv.weight); ref_mha.in_proj_bias.copy_(mine.qkv.bias)
        ref_mha.out_proj.weight.copy_(mine.proj.weight); ref_mha.out_proj.bias.copy_(mine.proj.bias)
    x = torch.randn(2, 10, D)
    mask = torch.ones(10, 10, dtype=torch.bool).triu(1)
    with torch.no_grad():
        e_mha = (mine(x) - ref_mha(x, x, x, attn_mask=mask, need_weights=False)[0]).abs().max().item()
    print(f"  ScratchMHA vs nn.MultiheadAttention (4 heads, causal): max diff {e_mha:.1e}")

    # why 1/sqrt(d): the softmax saturates without it
    ds = [4, 16, 64, 256, 1024]
    ent_scaled, ent_raw = [], []
    for dd in ds:
        q, k = rng.normal(size=(2, 64, dd))
        for scale, store in ((np.sqrt(dd), ent_scaled), (1.0, ent_raw)):
            S = q @ k.T / scale
            P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            store.append(float(np.mean(np.max(P, 1))))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].imshow(A, cmap="viridis"); axs[0].set_title("causal attention weights A (T=6)")
    axs[0].set_xlabel("key position j"); axs[0].set_ylabel("query position i"); axs[0].grid(False)
    axs[1].semilogx(ds, ent_raw, "o-", label=r"$q\cdot k$ (no scaling)")
    axs[1].semilogx(ds, ent_scaled, "s-", label=r"$q\cdot k/\sqrt{d}$")
    axs[1].set_xlabel("head dimension d"); axs[1].set_ylabel("mean of max attention weight")
    axs[1].set_title("without scaling, softmax becomes one-hot"); axs[1].legend()
    savefig(fig, LESSON, "attention_basics.png")
    return dict(attn_numpy_vs_torch=e_np, mha_scratch_vs_torch=e_mha)


def gpt_part(quick):
    banner("2. A tiny GPT on Tiny Shakespeare")
    data = CharData(load_text(200_000 if quick else None))
    base = ngram_baselines(data)
    print(f"  vocab {data.vocab};  baselines (val nats/char): uniform {base['uniform']:.3f}, "
          f"unigram {base['unigram']:.3f}, bigram {base['bigram']:.3f}")
    torch.manual_seed(0)
    model = TinyGPT(data.vocab, block_size=64, d=128, n_layer=4, n_head=4)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  TinyGPT: 4 layers, 4 heads, d=128, context 64  ->  {n_par / 1e6:.2f} M parameters")
    steps = 200 if quick else 3000
    hist = train_gpt(model, data, steps=steps, eval_every=250 if not quick else 50,
                     sample_every=steps // 6, prompt="ROMEO:\n")
    gs = torch.Generator().manual_seed(42)
    sample = data.decode(model.generate(data.encode("ROMEO:\n"), 600, generator=gs)[0].tolist())
    (lesson_dir(LESSON) / "sample.txt").write_text(sample)
    print("  ---- sample (temperature 0.8) ----\n" + "\n".join("  | " + l for l in sample[:400].splitlines()))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(hist["step"], hist["train"], label="GPT train")
    ax.plot(hist["step"], hist["val"], label="GPT validation")
    for k, ls in (("unigram", ":"), ("bigram", "--")):
        ax.axhline(base[k], color="k", ls=ls, lw=0.8, label=f"{k} baseline")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy (nats / char)"); ax.legend(fontsize=8)
    ax.set_title(f"TinyGPT ({n_par / 1e6:.2f} M params), {hist['time'] / 60:.1f} min on CPU")
    savefig(fig, LESSON, "gpt_training.png")

    # attention maps of the trained model on a line of text
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    idx = data.encode(text[:64])
    model.eval()
    with torch.no_grad():
        model(idx)
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.5))
    for li, layer in enumerate((0, 3)):
        att = model.blocks[layer].attn.last_attn[0]
        for h in range(4):
            ax = axs[li, h]
            ax.imshow(att[h, :40, :40], cmap="magma"); ax.grid(False)
            ax.set_title(f"layer {layer + 1}, head {h + 1}", fontsize=9)
            ax.set_xticks(range(40), list(text[:40].replace("\n", "⏎")), fontsize=5)
            ax.set_yticks(range(40), list(text[:40].replace("\n", "⏎")), fontsize=5)
    savefig(fig, LESSON, "gpt_attention_maps.png")

    # GIF: the same prompt sampled at different stages of training
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axis("off")
    txt = ax.text(0.0, 1.0, "", va="top", family="monospace", fontsize=8, transform=ax.transAxes)
    ttl = ax.set_title("")

    def draw(k):
        s, smp = hist["samples"][k]
        txt.set_text("\n".join(smp.splitlines()[:9]))
        ttl.set_text(f"sample after {s} training steps")
        return txt, ttl

    save_gif(FuncAnimation(fig, draw, frames=len(hist["samples"])), LESSON, "gpt_samples.gif", fps=1, dpi=80)
    return dict(baselines=base, n_params=n_par, final_train=hist["train"][-1], final_val=hist["val"][-1],
                train_time_s=hist["time"], steps=steps)


def main(quick: bool = False) -> dict:
    seed_everything(8)
    set_torch_threads()
    res = scratch_part()
    res.update(gpt_part(quick))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
