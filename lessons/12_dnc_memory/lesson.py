"""Lesson 12 - Memory-augmented networks: the Differentiable Neural Computer (DNC).

Questions this lesson answers
-----------------------------
* How can a network use an external, addressable memory and still be trained by backprop?
  (Every read and write is a soft, differentiable attention over memory slots.)
* What are content-based addressing, dynamic allocation and temporal links for?
* Does a DNC generalise to longer sequences than it was trained on, where an LSTM fails?

Task: the copy task - read L random 6-bit vectors, see a delimiter, reproduce the L vectors.
Training uses L in 1..10; we test on L up to 24 (beyond the training range).

Run:  python lessons/12_dnc_memory/lesson.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from nnaz.common import banner, save_results, savefig, seed_everything, set_torch_threads, setup_matplotlib
from nnaz.dnc import DNC, LSTMBaseline, allocation, bit_error_rate, copy_batch, train_copy

LESSON = 12
plt = setup_matplotlib()
BITS = 6


def mechanics_part():
    banner("1. The memory mechanics in isolation")
    u = torch.tensor([[0.9, 0.1, 0.5, 0.0]])
    a = allocation(u)
    print(f"  usage {u.tolist()[0]} -> allocation weighting {[round(v, 3) for v in a.tolist()[0]]}"
          "  (the free slot gets the write)")
    return dict(allocation_example=a.tolist()[0])


def train_part(quick):
    banner("2. Training a DNC and an LSTM on the copy task (L = 1..10)")
    steps = 400 if quick else 6000
    torch.manual_seed(0)
    dnc = DNC(BITS + 1, BITS, hidden=64, N=32, W=16, R=1)
    torch.manual_seed(0)
    lstm = LSTMBaseline(BITS + 1, BITS, hidden=128, layers=2)
    n_d = sum(p.numel() for p in dnc.parameters())
    n_l = sum(p.numel() for p in lstm.parameters())
    print(f"  DNC: LSTM controller 64 + memory 32 x 16, {n_d} params | LSTM baseline 2 x 128: {n_l} params")
    print("  DNC:")
    h_d = train_copy(dnc, BITS, steps=steps, log=1000)
    print("  LSTM:")
    h_l = train_copy(lstm, BITS, steps=steps, log=1000)
    Ls = [2, 5, 10, 12, 16, 20, 24]
    ber = {"DNC": [bit_error_rate(dnc, L) for L in Ls], "LSTM": [bit_error_rate(lstm, L) for L in Ls]}
    for name, v in ber.items():
        print(f"  {name:5s} bit error rate: " + ", ".join(f"L={L}: {e:.3f}" for L, e in zip(Ls, v)))
    print("  (chance level: 0.5)")

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    axs[0].semilogy(h_d["step"], h_d["loss"], label=f"DNC ({n_d} params)")
    axs[0].semilogy(h_l["step"], h_l["loss"], label=f"LSTM ({n_l} params)")
    axs[0].set_xlabel("training step"); axs[0].set_ylabel("BCE per bit"); axs[0].legend()
    axs[0].set_title("copy task, training lengths 1..10")
    for name, v in ber.items():
        axs[1].plot(Ls, v, "o-", label=name)
    axs[1].axvspan(0, 10, color="0.9", zorder=0); axs[1].text(1, 0.45, "training range", fontsize=8)
    axs[1].axhline(0.5, color="k", ls=":", lw=0.8)
    axs[1].set_xlabel("sequence length L"); axs[1].set_ylabel("bit error rate"); axs[1].legend()
    axs[1].set_title("generalisation to longer sequences")
    savefig(fig, LESSON, "dnc_copy.png")

    # what the memory does: write and read weightings over time for one sequence
    g = torch.Generator().manual_seed(5)
    L = 12
    X, Y, m = copy_batch(1, L, BITS, g)
    with torch.no_grad():
        out, rec = dnc(X, record=True)
    ww = torch.stack(rec["ww"]).numpy().T
    wr = torch.stack(rec["wr"]).numpy().T
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.4, 1.4]})
    axs[0].imshow(X[0, :, :BITS].T, cmap="gray_r", aspect="auto"); axs[0].set_title("input (then delimiter)")
    axs[1].imshow(torch.sigmoid(out[0]).T, cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    axs[1].set_title("DNC output")
    axs[2].imshow(ww, cmap="magma", aspect="auto"); axs[2].set_title("write weighting (slot x time)")
    axs[3].imshow(wr, cmap="magma", aspect="auto"); axs[3].set_title("read weighting (slot x time)")
    for a in axs:
        a.set_xlabel("time step"); a.grid(False)
    savefig(fig, LESSON, "dnc_memory_access.png")
    return dict(dnc_params=n_d, lstm_params=n_l, lengths=Ls, ber=ber, dnc_time_s=h_d["time"],
                lstm_time_s=h_l["time"], dnc_final_loss=h_d["loss"][-1], lstm_final_loss=h_l["loss"][-1])


def main(quick: bool = False) -> dict:
    seed_everything(12)
    set_torch_threads()
    res = mechanics_part()
    res.update(train_part(quick))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
