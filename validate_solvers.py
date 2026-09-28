"""Verify every CFD / PDE solver used in the course against exact solutions.

    python validate_solvers.py          (~1-2 min on a laptop CPU)

The neural networks of lessons 09, 13, 14 and 16 are trained on, or compared with,
data from classical solvers written for this course (``nnaz/burgers.py``,
``nnaz/operators.py``, ``nnaz/gnn.py``, ``nnaz/lbm.py``). A learnt model can only
be as trustworthy as its reference, so each solver is verified here by a grid-
convergence study against an exact solution:

1. Cole-Hopf solution of viscous Burgers (Gauss-Hermite quadrature): node convergence.
2. Finite-difference Burgers solver (central FD + RK4) vs Cole-Hopf: order 2.
3. Pseudo-spectral Burgers solver (Fourier + integrating-factor RK4) vs Cole-Hopf:
   spectral (exponential) convergence.
4. P1 finite elements for -Laplace(u) = f vs the manufactured u = sin(pi x) sin(pi y): order 2.
5. Lattice Boltzmann (D2Q9 BGK) vs the exact Poiseuille profile: order 2.
   (The cylinder-wake Strouhal number is checked in lesson 16.)

Outputs: docs/solvers/solver_convergence.png and docs/results/solvers.json.
"""
from __future__ import annotations

import json
import time

import numpy as np

from nnaz.burgers import NU_CLASSIC, cole_hopf, fd_solve
from nnaz.common import DOCS, setup_matplotlib
from nnaz.gnn import fem_poisson, square_mesh
from nnaz.lbm import poiseuille
from nnaz.operators import burgers_spectral


def order(h, e):
    """Least-squares slope of log(error) vs log(h)."""
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def main():
    plt = setup_matplotlib()
    res, t0 = {}, time.time()

    # 1. quadrature
    x = np.linspace(-1, 1, 801)
    ref = cole_hopf(x, 0.5, n_quad=250)
    nq = [20, 40, 60, 80, 120, 160]
    eq = [float(np.abs(cole_hopf(x, 0.5, n_quad=n) - ref).max()) for n in nq]
    res["cole_hopf_quadrature"] = dict(nodes=nq, max_err_vs_250=eq)

    # 2. finite differences
    Ns = [512, 1024, 2048, 4096]
    efd = []
    for N in Ns:
        xg, u, info = fd_solve(NU_CLASSIC, 1.0, 1.0, N=N)
        ex = cole_hopf(xg, 1.0)
        efd.append(float(np.linalg.norm(u[0] - ex) / np.linalg.norm(ex)))
    res["fd_burgers"] = dict(N=Ns, rel_l2=efd, order=-order(Ns, efd))

    # 3. spectral
    Nsp = [128, 192, 256, 384, 512]
    esp = []
    for N in Nsp:
        xs = -1 + 2 * np.arange(N) / N
        u, _ = burgers_spectral(-np.sin(np.pi * xs), 0.02, 1.0, Lx=2.0, cfl=0.2)
        esp.append(float(np.abs(u[0] - cole_hopf(xs, 1.0, 0.02)).max()))
    res["spectral_burgers_nu0.02"] = dict(N=Nsp, max_err=esp)

    # 4. FEM
    rng = np.random.default_rng(0)
    nf = [9, 17, 33, 65, 129]
    efem = []
    for n in nf:
        P, tri, bnd = square_mesh(n, rng)
        ue = np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])
        efem.append(float(np.linalg.norm(fem_poisson(P, tri, bnd, 2 * np.pi ** 2 * ue) - ue) / np.linalg.norm(ue)))
    hf = [1 / (n - 1) for n in nf]
    res["p1_fem_poisson"] = dict(n=nf, rel_l2=efem, order=order(hf, efem))

    # 5. lattice Boltzmann Poiseuille
    nys = [8, 16, 32]
    elbm = []
    for ny in nys:
        y, u, ue = poiseuille(ny=ny, n_steps=int(40 * ny ** 2))
        elbm.append(float(np.abs(u - ue).max() / ue.max()))
    res["lbm_poiseuille"] = dict(ny=nys, max_rel_err=elbm, order=-order(nys, elbm))
    res["time_s"] = time.time() - t0

    for k, v in res.items():
        print(k, v)
    fig, axs = plt.subplots(1, 5, figsize=(20, 3.6))
    axs[0].semilogy(nq, np.maximum(eq, 1e-17), "o-"); axs[0].set_xlabel("Gauss-Hermite nodes")
    axs[0].set_title("Cole-Hopf quadrature (t = 0.5)"); axs[0].set_ylabel("max error")
    axs[1].loglog(Ns, efd, "o-", label="FD + RK4")
    axs[1].loglog(Ns, efd[0] * (np.array(Ns) / Ns[0]) ** -2.0, "k:", label="slope -2")
    axs[1].set_xlabel("grid points N"); axs[1].set_title(r"FD Burgers vs exact, $\nu = 0.01/\pi$"); axs[1].legend()
    axs[2].semilogy(Nsp, esp, "o-"); axs[2].set_xlabel("Fourier modes N")
    axs[2].set_title(r"spectral Burgers vs exact, $\nu=0.02$ (exponential)")
    axs[3].loglog(hf, efem, "o-", label="P1 FEM"); axs[3].loglog(hf, efem[0] * (np.array(hf) / hf[0]) ** 2, "k:", label="slope 2")
    axs[3].set_xlabel("mesh size h"); axs[3].set_title("FEM Poisson, manufactured solution"); axs[3].legend()
    axs[4].loglog(nys, elbm, "o-", label="D2Q9 BGK"); axs[4].loglog(nys, elbm[0] * (np.array(nys) / nys[0]) ** -2.0, "k:", label="slope -2")
    axs[4].set_xlabel("cells across the channel"); axs[4].set_title("lattice Boltzmann, Poiseuille flow"); axs[4].legend()
    out = DOCS / "solvers"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "solver_convergence.png", dpi=100, bbox_inches="tight")
    (DOCS / "results").mkdir(exist_ok=True)
    (DOCS / "results" / "solvers.json").write_text(json.dumps(res, indent=2))
    print(f"saved docs/solvers/solver_convergence.png ({res['time_s']:.0f}s)")


if __name__ == "__main__":
    main()
