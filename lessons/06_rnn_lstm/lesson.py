"""Lesson 06 - RNN / LSTM forecasting of a noisy nonlinear oscillator (Van der Pol).

Questions this lesson answers
-----------------------------
* How does a recurrent network process a sequence, and why was the LSTM invented?
* How good is a *linear* autoregressive model (the classical baseline) on nonlinear data?
* One-step prediction vs closed-loop (autoregressive) forecasting: why are they different?
* What is the irreducible error when the observations are noisy?

Data: x'' - mu (1 - x^2) x' + x = 0 with mu = 2 (relaxation oscillations), integrated
with SciPy RK45 (rtol 1e-10), sampled every dt = 0.1, with Gaussian observation noise
of standard deviation 0.1 added. First 60 % train, next 20 % validation, last 20 % test.

Run:  python lessons/06_rnn_lstm/lesson.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from nnaz.common import banner, save_gif, save_results, savefig, seed_everything, set_torch_threads, setup_matplotlib
from nnaz.data import van_der_pol
from nnaz.rnn import SeqRegressor, ar_predict, fit_linear_ar, make_windows, net_predict, rollout, train_seq

LESSON = 6
plt = setup_matplotlib()
P = 40          # window length (4 time units, about half a period)
NOISE = 0.1
HORIZON = 150   # closed-loop forecast length (~2 periods of 7.6 time units)


def data_part(rng):
    banner("1. Data: the Van der Pol oscillator with observation noise")
    t, clean, noisy = van_der_pol(8000, dt=0.1, mu=2.0, noise=NOISE, rng=rng)
    n = len(t)
    s = noisy[: int(0.6 * n)].std()  # scale with TRAIN statistics only
    d = dict(t=t, clean=clean / s, noisy=noisy / s, scale=s,
             tr=slice(0, int(0.6 * n)), va=slice(int(0.6 * n), int(0.8 * n)), te=slice(int(0.8 * n), n))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    axs[0].plot(t[:600], noisy[:600], ".", ms=2, color="0.5", label="observed (noisy)")
    axs[0].plot(t[:600], clean[:600], "k", lw=1, label="true state x(t)")
    axs[0].set_xlabel("t"); axs[0].legend(); axs[0].set_title(r"Van der Pol, $\mu=2$, noise $\sigma=0.1$")
    v = np.gradient(clean, t)
    axs[1].plot(clean[:3000], v[:3000], lw=0.8); axs[1].set_xlabel("x"); axs[1].set_ylabel("dx/dt")
    axs[1].set_title("phase portrait: a limit cycle")
    savefig(fig, LESSON, "vdp_data.png")
    return d


def models_part(d, quick):
    banner("2. Baselines (persistence, linear AR) and recurrent nets (RNN, LSTM)")
    x, c = d["noisy"], d["clean"]
    Xtr, ytr = make_windows(x[d["tr"]], P)
    Xva, yva = make_windows(x[d["va"]], P)
    Xte, yte = make_windows(x[d["te"]], P)
    _, yte_clean = make_windows(c[d["te"]], P)

    preds, hists = {}, {}
    preds["persistence"] = lambda w: w[:, -1]
    coef = fit_linear_ar(x[d["tr"]], P)
    preds[f"linear AR({P})"] = lambda w: ar_predict(coef, w)
    for cell in ("rnn", "lstm"):
        torch.manual_seed(0)
        model = SeqRegressor(cell, hidden=64)
        print(f"  training {cell.upper()} ({sum(p.numel() for p in model.parameters())} params)")
        hists[cell] = train_seq(model, Xtr, ytr, Xva, yva, steps=300 if quick else 3000, log=1000)
        preds[cell.upper()] = (lambda m: (lambda w: net_predict(m, w)))(model)

    sc = d["scale"]
    one = {}
    for name, f in preds.items():
        p = f(Xte)
        one[name] = dict(rmse_vs_noisy=float(np.sqrt(np.mean((p - yte) ** 2)) * sc),
                         rmse_vs_clean=float(np.sqrt(np.mean((p - yte_clean) ** 2)) * sc))
        print(f"  one-step test RMSE {name:15s} vs noisy obs {one[name]['rmse_vs_noisy']:.4f}"
              f"   vs true state {one[name]['rmse_vs_clean']:.4f}")
    print(f"  (noise floor: an oracle that knows the true state scores {NOISE:.3f} against the noisy obs)")
    return preds, hists, one, (Xte, yte_clean)


def rollout_part(d, preds):
    banner("3. Closed-loop forecasts: feed predictions back in for 150 steps")
    x, c = d["noisy"][d["te"]], d["clean"][d["te"]]
    starts = np.arange(0, len(x) - P - HORIZON, 25)
    W0 = np.stack([x[s:s + P] for s in starts])
    truth = np.stack([c[s + P:s + P + HORIZON] for s in starts])
    roll, rmse_h = {}, {}
    for name, f in preds.items():
        r = rollout(f, W0, HORIZON)
        roll[name] = r
        rmse_h[name] = np.sqrt(np.mean((r - truth) ** 2, axis=0)) * d["scale"]
    out = {name: {"rmse_h10": float(v[9]), "rmse_h50": float(v[49]), "rmse_h150": float(v[-1]),
                  "rmse_mean": float(v.mean())} for name, v in rmse_h.items()}
    for name, v in out.items():
        print(f"  {name:15s} RMSE after 10 steps {v['rmse_h10']:.3f} | 50 steps {v['rmse_h50']:.3f}"
              f" | 150 steps {v['rmse_h150']:.3f}   (starts: {len(starts)})")
    std = np.std(c) * d["scale"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for name, v in rmse_h.items():
        axs[0].semilogy(np.arange(1, HORIZON + 1) * 0.1, v, label=name)
    axs[0].axhline(std, color="k", ls=":", lw=0.8)
    axs[0].text(0.2, std * 1.03, "std of the signal", fontsize=8)
    axs[0].set_xlabel("forecast horizon (time units)"); axs[0].set_ylabel("RMSE vs true state")
    axs[0].set_title(f"closed-loop error vs horizon (mean over {len(starts)} starts)"); axs[0].legend(fontsize=8)
    k = 3
    tt = np.arange(HORIZON) * 0.1
    axs[1].plot(np.arange(-P, 0) * 0.1, W0[k] * d["scale"], ".", color="0.5", ms=3, label="observed window")
    axs[1].plot(tt, truth[k] * d["scale"], "k", lw=2, label="truth")
    for name in (f"linear AR({P})", "RNN", "LSTM"):
        axs[1].plot(tt, roll[name][k] * d["scale"], lw=1.2, label=name)
    axs[1].set_xlabel("t (relative)"); axs[1].legend(fontsize=7); axs[1].set_title("one example forecast")
    savefig(fig, LESSON, "rollout.png")

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-P * 0.1, HORIZON * 0.1); ax.set_ylim(-2.8, 2.8)
    ax.plot(np.arange(-P, 0) * 0.1, W0[k] * d["scale"], ".", color="0.5", ms=3)
    ax.axvline(0, color="k", lw=0.5)
    lines = {n: ax.plot([], [], lw=1.5 if n != "truth" else 2.5, label=n,
                        color="k" if n == "truth" else None)[0]
             for n in ("truth", f"linear AR({P})", "LSTM")}
    ax.legend(fontsize=7, loc="lower left"); ttl = ax.set_title("")

    def draw(f):
        h = 3 * f + 1
        lines["truth"].set_data(tt[:h], truth[k, :h] * d["scale"])
        for n in (f"linear AR({P})", "LSTM"):
            lines[n].set_data(tt[:h], roll[n][k, :h] * d["scale"])
        ttl.set_text(f"closed-loop forecast, step {h}")
        return list(lines.values())

    save_gif(FuncAnimation(fig, draw, frames=HORIZON // 3), LESSON, "forecast.gif", fps=12)
    return out


def main(quick: bool = False) -> dict:
    rng = seed_everything(6)
    set_torch_threads()
    d = data_part(rng)
    preds, hists, one, _ = models_part(d, quick)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for cell, h in hists.items():
        ax.semilogy(h["step"], h["val"], label=f"{cell.upper()} validation MSE")
    ax.set_xlabel("step"); ax.legend(); ax.set_title("training (scaled units)")
    savefig(fig, LESSON, "training.png")
    res = {"one_step": one, "rollout": rollout_part(d, preds),
           "train_time_s": {k: h["time"] for k, h in hists.items()}}
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
