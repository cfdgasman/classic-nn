"""Lesson 14 - Neural operators: DeepONet and the Fourier Neural Operator on Burgers.

Questions this lesson answers
-----------------------------
* What is operator learning, and how is it different from a PINN or a parameter surrogate?
* How do DeepONet (branch x trunk) and FNO (learnt spectral convolutions) represent an operator?
* How do they compare with the best *linear* operator and with the solver itself?
* Can an FNO trained on a coarse grid be evaluated on a finer one?

Data: u0 ~ Gaussian random field (periodic, zero mean), u_t + u u_x = nu u_xx on [0,1),
nu = 0.01, target u(x, T=1). The pseudo-spectral solver is validated against Cole-Hopf.

Run:  python lessons/14_neural_operators/lesson.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from nnaz.burgers import cole_hopf
from nnaz.common import DATA, banner, save_results, savefig, seed_everything, set_torch_threads, setup_matplotlib
from nnaz.operators import DeepONet, FNO1d, burgers_spectral, grf_1d, rel_l2, train_operator

LESSON = 14
plt = setup_matplotlib()
NU, T, N_SOLVE, N_TRAIN_GRID = 0.01, 1.0, 512, 128


def solver_part():
    banner("1. The data generator: a pseudo-spectral Burgers solver, checked against Cole-Hopf")
    out = {}
    for N, nu in ((256, 0.05), (512, 0.05), (1024, 0.01 / np.pi), (2048, 0.01 / np.pi)):
        x = -1 + 2 * np.arange(N) / N
        u, _ = burgers_spectral(-np.sin(np.pi * x), nu, 1.0, Lx=2.0)
        e = float(np.abs(u[0] - cole_hopf(x, 1.0, nu)).max())
        out[f"N{N}_nu{nu:.4f}"] = e
        print(f"  N = {N:4d}, nu = {nu:.4f}: max error vs exact {e:.1e}")
    return {"spectral_solver_check": out}


def make_data(n, seed):
    """Generate (or load cached) pairs; also check the resolution of the solver."""
    path = DATA / f"burgers_operator_{n}_{seed}.npz"
    if path.exists():
        d = np.load(path)
        return d["u0"], d["uT"], float(d["t"])
    rng = np.random.default_rng(seed)
    u0 = grf_1d(n, N_SOLVE, rng, sigma=25.0)
    t0 = time.time()
    uT, _ = burgers_spectral(u0, NU, T)
    t = (time.time() - t0) / n
    np.savez_compressed(path, u0=u0, uT=uT, t=t)
    return u0, uT, t


def operator_part(quick):
    banner("2. Learning G: u0 -> u(., T=1) from solver data")
    n_tr, n_te = (200, 50) if quick else (1000, 200)
    u0, uT, t_solve = make_data(n_tr + n_te, 14)
    # resolution check: re-solve a few samples on a 2x finer grid
    rng = np.random.default_rng(1)
    idx = rng.choice(len(u0), 4, replace=False)
    fine0 = np.fft.irfft(np.fft.rfft(u0[idx]), n=2 * N_SOLVE) * 2       # spectral interpolation
    fineT, _ = burgers_spectral(fine0, NU, T)
    res_err = float(np.max(np.abs(fineT[:, ::2] - uT[idx])) / np.abs(uT[idx]).max())
    print(f"  {n_tr} train / {n_te} test pairs; solver N={N_SOLVE}: {1000 * t_solve:.1f} ms per sample (batched);"
          f" change on a 2x finer grid {res_err:.1e} (relative)")
    s = N_SOLVE // N_TRAIN_GRID
    U0 = torch.as_tensor(u0[:, ::s], dtype=torch.float32)
    UT = torch.as_tensor(uT[:, ::s], dtype=torch.float32)
    U0tr, UTtr, U0te, UTte = U0[:n_tr], UT[:n_tr], U0[n_tr:], UT[n_tr:]
    res = {"solver_ms_per_sample": 1000 * t_solve, "solver_resolution_check": res_err}

    # baselines: identity (u(T) = u0), and the best LINEAR operator (least squares, ridge)
    res["identity"] = float(rel_l2(U0te, UTte).mean())
    A = np.linalg.solve(U0tr.T.numpy() @ U0tr.numpy() + 1e-3 * np.eye(N_TRAIN_GRID), U0tr.T.numpy() @ UTtr.numpy())
    res["best linear operator"] = float(rel_l2(U0te @ torch.as_tensor(A, dtype=torch.float32), UTte).mean())
    print(f"  baselines (test rel. L2): identity {res['identity']:.3f} | best linear operator {res['best linear operator']:.3f}")

    epochs = 30 if quick else 300
    x = torch.linspace(0, 1, N_TRAIN_GRID + 1)[:-1]
    torch.manual_seed(0)
    don = DeepONet(N_TRAIN_GRID)
    print(f"  DeepONet ({sum(p.numel() for p in don.parameters())} params)")
    _, t_don = train_operator(don, U0tr, UTtr, epochs=epochs, deeponet_x=x, log=100 if not quick else 0)
    torch.manual_seed(0)
    fno = FNO1d(modes=16, width=64)
    print(f"  FNO ({sum(p.numel() for p in fno.parameters())} params)")
    _, t_fno = train_operator(fno, U0tr, UTtr, epochs=epochs, log=100 if not quick else 0)
    with torch.no_grad():
        p_don = don(U0te, x)
        p_fno = fno(U0te)
        e_don, e_fno = rel_l2(p_don, UTte), rel_l2(p_fno, UTte)
        # zero-shot super-resolution: evaluate the same FNO on the full 512-point grid
        U0f = torch.as_tensor(u0[n_tr:], dtype=torch.float32)
        UTf = torch.as_tensor(uT[n_tr:], dtype=torch.float32)
        t0 = time.time()
        p_fno_f = fno(U0f)
        t_inf = (time.time() - t0) / n_te
        e_fno_f = rel_l2(p_fno_f, UTf)
        xf = torch.linspace(0, 1, N_SOLVE + 1)[:-1]
        e_don_f = rel_l2(don(U0te, xf), UTf)   # DeepONet: same sensors, finer query points
    res.update({"DeepONet": float(e_don.mean()), "FNO": float(e_fno.mean()),
                "FNO on 512 grid (trained on 128)": float(e_fno_f.mean()),
                "DeepONet queried on 512 grid": float(e_don_f.mean()),
                "train_s": {"DeepONet": t_don, "FNO": t_fno}, "fno_ms_per_sample": 1000 * t_inf})
    print(f"  test rel. L2: DeepONet {res['DeepONet']:.4f} | FNO {res['FNO']:.4f} | FNO evaluated on the 4x finer"
          f" grid {res['FNO on 512 grid (trained on 128)']:.4f} | DeepONet queried at 512 points {res['DeepONet queried on 512 grid']:.4f}")
    print(f"  cost per sample: solver {res['solver_ms_per_sample']:.2f} ms vs FNO {res['fno_ms_per_sample']:.3f} ms")

    xs = np.linspace(0, 1, N_TRAIN_GRID, endpoint=False)
    xf_np = np.linspace(0, 1, N_SOLVE, endpoint=False)
    worst = int(torch.argmax(e_fno)); med = int(torch.argsort(e_fno)[len(e_fno) // 2])
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    for a, k, lab in [(axs[0], med, "median"), (axs[1], worst, "worst")]:
        a.plot(xs, U0te[k], color="0.7", label="u0")
        a.plot(xf_np, UTf[k], "k", lw=3, alpha=0.4, label="solver u(T)")
        a.plot(xs, p_don[k], "--", label="DeepONet")
        a.plot(xf_np, p_fno_f[k], ":", lw=2, label="FNO (512 grid)")
        a.set_title(f"{lab} FNO test case"); a.legend(fontsize=7); a.set_xlabel("x")
    names = ["identity", "best linear operator", "DeepONet", "FNO", "FNO on 512 grid (trained on 128)"]
    axs[2].bar(range(len(names)), [res[n] for n in names], color=["0.6", "0.6", "C0", "C1", "C1"])
    axs[2].set_yscale("log"); axs[2].set_xticks(range(len(names)), ["identity", "linear", "DeepONet", "FNO", "FNO\n(4x grid)"])
    axs[2].set_ylabel("mean test rel. L2 error"); axs[2].set_title("operator learning on Burgers")
    savefig(fig, LESSON, "operators.png")

    # what the FNO learnt: the spectral weights (magnitude per mode, first layer)
    w = fno.spec[0].w.detach().abs().mean((0, 1)).numpy()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(w)), w); ax.set_xlabel("Fourier mode k"); ax.set_ylabel("mean |R_k|")
    ax.set_title("FNO layer 1: learnt spectral weights")
    savefig(fig, LESSON, "fno_modes.png")
    return res


def main(quick: bool = False) -> dict:
    seed_everything(14)
    set_torch_threads()
    res = solver_part()
    res.update(operator_part(quick))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
