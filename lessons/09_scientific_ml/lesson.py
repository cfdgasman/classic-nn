"""Lesson 09 - Scientific ML: a PINN for viscous Burgers, and a neural surrogate of a PDE solver.

Questions this lesson answers
-----------------------------
* How can a network solve a PDE without any solution data? (Minimise the residual.)
* How accurate is it against the exact (Cole-Hopf) solution, and what does it cost
  compared with a classical finite-difference solver?
* Where do PINNs struggle? (Steep fronts: the classic nu = 0.01/pi shock-like layer.)
* What is a *surrogate*, and when does learning from solver data pay off?

Problem:  u_t + u u_x = nu u_xx on [-1,1] x [0,1],  u(x,0) = -sin(pi x),  u(+-1,t) = 0,
          nu = 0.01/pi (the benchmark of Raissi et al. 2019).

Run:  python lessons/09_scientific_ml/lesson.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from nnaz.burgers import NU_CLASSIC, cole_hopf, fd_solve
from nnaz.common import banner, save_gif, save_results, savefig, seed_everything, set_torch_threads, setup_matplotlib
from nnaz.pinn import BurgersPINN, Surrogate, normalise_params, pinn_eval_grid, train_pinn, train_surrogate

LESSON = 9
plt = setup_matplotlib()


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def reference_part():
    banner("1. The exact Cole-Hopf solution, and a finite-difference solver checked against it")
    x = np.linspace(-1, 1, 801)
    q = max(np.abs(cole_hopf(x, t, n_quad=120) - cole_hopf(x, t, n_quad=200)).max() for t in (0.05, 0.5, 1.0))
    print(f"  Gauss-Hermite quadrature converged: |u(120 nodes) - u(200 nodes)| <= {q:.1e}")
    Ns, errs, times = [128, 256, 512, 1024, 2048], [], []
    for N in Ns:
        t0 = time.time()
        xg, u, info = fd_solve(NU_CLASSIC, 1.0, 1.0, N=N)
        times.append(time.time() - t0)
        errs.append(rel_l2(u[0], cole_hopf(xg, 1.0)))
        print(f"  FD N={N:5d}: rel. L2 error at t=1 {errs[-1]:.2e}  cell Reynolds {info['cell_reynolds']:.2f}"
              f"  steps {info['n_steps']:5d}  {times[-1]:.2f}s")
    order = np.polyfit(np.log(Ns[1:]), np.log(errs[1:]), 1)[0]
    print(f"  observed order of accuracy: {-order:.2f} (second-order scheme)")
    return dict(quadrature_check=q, fd_N=Ns, fd_err=errs, fd_time=times, fd_order=float(-order))


def pinn_part(quick):
    banner("2. A PINN for Burgers, nu = 0.01/pi (no solution data used for training)")
    torch.set_default_dtype(torch.float64)   # second derivatives: float64 avoids round-off trouble
    x = np.linspace(-1, 1, 256)
    t = np.linspace(0, 1, 101)
    U_exact = cole_hopf(x[None], t[:, None])
    torch.manual_seed(0)
    model = BurgersPINN(width=32, depth=6)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  network: 6 x 32 tanh, {n_par} parameters; hard-constrained IC/BC")
    ev = lambda m: rel_l2(pinn_eval_grid(m, x, t), U_exact)
    hist = train_pinn(model, NU_CLASSIC, n_colloc=6000 if not quick else 1000,
                      adam_steps=3000 if not quick else 300, lbfgs_steps=3000 if not quick else 250,
                      eval_fn=ev, eval_every=250 if not quick else 50)
    U = pinn_eval_grid(model, x, t)
    err = rel_l2(U, U_exact)
    err_t = [rel_l2(U[k], U_exact[k]) for k in (25, 50, 75, 100)]
    max_err = float(np.abs(U - U_exact).max())
    print(f"  PINN rel. L2 error over the space-time grid: {err:.2e} (max abs error {max_err:.2e})"
          f"  training time {hist['time']:.0f}s")
    print(f"  per time: t=0.25 {err_t[0]:.1e}, t=0.5 {err_t[1]:.1e}, t=0.75 {err_t[2]:.1e}, t=1 {err_t[3]:.1e}")

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    ext = [0, 1, -1, 1]
    for ax, F, ttl, cm in [(axs[0], U_exact, "exact (Cole-Hopf)", "RdBu_r"), (axs[1], U, "PINN", "RdBu_r"),
                           (axs[2], np.abs(U - U_exact), "|PINN - exact|", "magma")]:
        im = ax.imshow(F.T, origin="lower", aspect="auto", extent=ext, cmap=cm)
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(ttl); ax.grid(False)
        fig.colorbar(im, ax=ax)
    savefig(fig, LESSON, "pinn_fields.png")

    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, k in zip(axs, (0, 25, 50, 100)):
        ax.plot(x, U_exact[k], "k", lw=3, alpha=0.4, label="exact")
        ax.plot(x, U[k], "r--", lw=1.2, label="PINN")
        ax.set_title(f"t = {t[k]:.2f}"); ax.set_xlabel("x")
    axs[0].legend()
    savefig(fig, LESSON, "pinn_slices.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogy(hist["it"], hist["loss"], label="mean squared PDE residual")
    ax.semilogy(hist["it"], hist["err"], label="rel. L2 error vs exact")
    ax.axvline(3000 if not quick else 300, color="k", lw=0.8, ls=":")
    ax.text(3000 if not quick else 300, ax.get_ylim()[1] * 0.3, " Adam | L-BFGS", fontsize=8)
    ax.set_xlabel("iteration"); ax.legend(); ax.set_title("PINN training")
    savefig(fig, LESSON, "pinn_training.png")

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(5, 3))
    le, = ax.plot(x, U_exact[0], "k", lw=3, alpha=0.4, label="exact")
    lp, = ax.plot(x, U[0], "r--", label="PINN")
    ax.set_ylim(-1.1, 1.1); ax.legend(loc="upper right"); ttl = ax.set_title("")

    def draw(k):
        le.set_ydata(U_exact[k]); lp.set_ydata(U[k]); ttl.set_text(f"Burgers, t = {t[k]:.2f}")
        return le, lp

    save_gif(FuncAnimation(fig, draw, frames=range(0, 101, 2)), LESSON, "pinn_burgers.gif", fps=12)
    torch.set_default_dtype(torch.float32)
    return dict(pinn_rel_l2=err, pinn_max_abs=max_err, pinn_rel_l2_t=err_t, pinn_params=n_par,
                pinn_time_s=hist["time"])


def surrogate_part(quick):
    banner("3. A neural surrogate: (nu, A) -> u(x, T=0.5), trained on finite-difference solutions")
    nu_rng, A_rng, T, N = (0.005, 0.05), (0.5, 1.5), 0.5, 512
    rng = np.random.default_rng(9)
    n_train, n_test = (400, 200) if not quick else (80, 40)

    def sample(n):
        return np.exp(rng.uniform(*np.log(nu_rng), n)), rng.uniform(*A_rng, n)

    nu_tr, A_tr = sample(n_train)
    nu_te, A_te = sample(n_test)
    t0 = time.time()
    x, U_tr, info = fd_solve(nu_tr, A_tr, T, N=N)   # all training runs at once (vectorised)
    solve_time = (time.time() - t0) / n_train
    _, U_te, _ = fd_solve(nu_te, A_te, T, N=N)
    # sanity: the solver against the exact solution at the extreme parameters
    chk = max(rel_l2(fd_solve(nu, A, T, N=N)[1][0], cole_hopf(x, T, nu, A))
              for nu, A in [(nu_rng[0], A_rng[1]), (nu_rng[1], A_rng[0])])
    print(f"  solver: N={N}, max cell Reynolds {info['cell_reynolds']:.2f}; worst rel. L2 error vs Cole-Hopf"
          f" at the parameter-space corners {chk:.1e}; {1000 * solve_time:.1f} ms per solution (vectorised)")
    sub = slice(None, None, 4)                      # the surrogate predicts 128 grid values
    xs, Ytr, Yte = x[sub], U_tr[:, sub], U_te[:, sub]
    Ptr, Pte = normalise_params(nu_tr, A_tr, nu_rng, A_rng), normalise_params(nu_te, A_te, nu_rng, A_rng)

    torch.manual_seed(0)
    model = Surrogate(Ytr.shape[1])
    t0 = time.time()
    train_surrogate(model, Ptr, Ytr, epochs=1500 if not quick else 100)
    train_time = time.time() - t0
    with torch.no_grad():
        t0 = time.time()
        pred = model(torch.as_tensor(Pte, dtype=torch.float32)).numpy()
        nn_time = (time.time() - t0) / n_test
    rel = np.linalg.norm(pred - Yte, axis=1) / np.linalg.norm(Yte, axis=1)
    # baselines: the mean solution, and nearest neighbour in parameter space
    mean_rel = np.linalg.norm(Ytr.mean(0) - Yte, axis=1) / np.linalg.norm(Yte, axis=1)
    nn_idx = np.argmin(((Pte[:, None] - Ptr[None]) ** 2).sum(-1), 1)
    nn_rel = np.linalg.norm(Ytr[nn_idx] - Yte, axis=1) / np.linalg.norm(Yte, axis=1)
    from scipy.interpolate import LinearNDInterpolator
    lin = LinearNDInterpolator(Ptr, Ytr)(Pte)
    ok = ~np.isnan(lin).any(1)
    lin_rel = np.linalg.norm(lin[ok] - Yte[ok], axis=1) / np.linalg.norm(Yte[ok], axis=1)
    res = dict(n_train=n_train, n_test=n_test, solver_check=chk, solver_ms=1000 * solve_time,
               surrogate_ms=1000 * nn_time, surrogate_train_s=train_time,
               rel_err_surrogate_median=float(np.median(rel)), rel_err_surrogate_p95=float(np.percentile(rel, 95)),
               rel_err_mean_median=float(np.median(mean_rel)), rel_err_nn_median=float(np.median(nn_rel)),
               rel_err_nn_p95=float(np.percentile(nn_rel, 95)),
               rel_err_lin_median=float(np.median(lin_rel)), rel_err_lin_p95=float(np.percentile(lin_rel, 95)))
    print(f"  test rel. L2 error (median / 95th pct): surrogate {res['rel_err_surrogate_median']:.2e} /"
          f" {res['rel_err_surrogate_p95']:.2e} | nearest neighbour {res['rel_err_nn_median']:.2e} /"
          f" {res['rel_err_nn_p95']:.2e} | linear interpolation {res['rel_err_lin_median']:.2e} /"
          f" {res['rel_err_lin_p95']:.2e} | mean field {res['rel_err_mean_median']:.2e}")
    print(f"  cost: solver {res['solver_ms']:.1f} ms/solution, surrogate {res['surrogate_ms']:.4f} ms/solution"
          f" (+ {train_time:.0f}s training)")

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    worst = np.argsort(rel)
    for ax, k, lab in [(axs[0], worst[len(worst) // 2], "median"), (axs[1], worst[-1], "worst")]:
        ax.plot(xs, Yte[k], "k", lw=3, alpha=0.4, label="FD solver")
        ax.plot(xs, pred[k], "r--", label="surrogate")
        ax.plot(xs, Ytr[nn_idx[k]], ":", color="tab:blue", label="nearest neighbour")
        ax.set_title(f"{lab} test case: nu={nu_te[k]:.3f}, A={A_te[k]:.2f}"); ax.legend(fontsize=8)
    sc = axs[2].scatter(nu_te, A_te, c=rel, cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
    axs[2].scatter(nu_tr, A_tr, s=3, c="0.6", label="training runs")
    axs[2].set_xscale("log"); axs[2].set_xlabel(r"$\nu$"); axs[2].set_ylabel("A")
    fig.colorbar(sc, ax=axs[2], label="surrogate rel. error"); axs[2].legend(fontsize=7)
    axs[2].set_title("error over parameter space")
    savefig(fig, LESSON, "surrogate.png")
    return res


def main(quick: bool = False) -> dict:
    seed_everything(9)
    set_torch_threads()
    res = reference_part()
    res.update(pinn_part(quick))
    res.update(surrogate_part(quick))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
