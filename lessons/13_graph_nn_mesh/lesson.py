"""Lesson 13 - Spatial graph neural networks on unstructured meshes (a learnt Poisson solver).

Questions this lesson answers
-----------------------------
* How do neural networks work on meshes that are not regular grids? (Graphs + message passing.)
* What is the difference between spectral (GCN) and spatial (message-passing) GNNs,
  and why do edge features (relative positions) matter for physics?
* Can a GNN trained on one family of meshes predict the FEM solution of -Laplace(u) = f
  on *new* meshes, including finer ones it never saw? Where does it break?

Data: P1 finite-element solutions (validated against a manufactured solution) on
jittered Delaunay meshes of the unit square, with random Gaussian-bump sources.

Run:  python lessons/13_graph_nn_mesh/lesson.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from nnaz.common import banner, save_results, savefig, seed_everything, set_torch_threads, setup_matplotlib
from nnaz.gnn import GCN, MPNN, PointMLP, collate, fem_poisson, make_graph, rel_errors, square_mesh, train_gnn

LESSON = 13
plt = setup_matplotlib()


def fem_part(rng):
    banner("1. The reference solver: P1 finite elements, checked with a manufactured solution")
    ns, errs = [9, 17, 33, 65], []
    for n in ns:
        P, tri, bnd = square_mesh(n, rng)
        ue = np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])  # exact; -Laplace(ue) = 2 pi^2 ue
        u = fem_poisson(P, tri, bnd, 2 * np.pi ** 2 * ue)
        errs.append(float(np.linalg.norm(u - ue) / np.linalg.norm(ue)))
        print(f"  {n:3d} x {n:3d} nodes: rel. L2 nodal error {errs[-1]:.2e}")
    order = -np.polyfit(np.log(ns), np.log(errs), 1)[0]
    print(f"  observed order {order:.2f} (P1 FEM: 2)")
    return dict(fem_n=ns, fem_err=errs, fem_order=float(order))


def gnn_part(rng, quick):
    banner("2. Learning the solution operator f -> u on random meshes")
    n_train = 60 if quick else 400
    t0 = time.time()
    train = [make_graph(int(rng.integers(14, 21)), rng) for _ in range(n_train)]
    test = [make_graph(int(rng.integers(14, 21)), rng) for _ in range(50)]
    fine = [make_graph(30, rng) for _ in range(20)]          # finer than any training mesh
    print(f"  generated {n_train} training meshes (196-400 nodes) + test sets with FEM in {time.time() - t0:.0f}s")
    f_scale = np.std(np.concatenate([g["f"] for g in train]))
    u_scale = np.std(np.concatenate([g["u"] for g in train]))
    steps = 150 if quick else 2500
    models = {"MPNN (spatial, edge vectors), 12 layers": MPNN(K=12, h=32),
              "GCN (spectral-derived, isotropic), 12 layers": GCN(K=12, h=32),
              "per-node MLP (no neighbours)": PointMLP()}
    res = {}
    for name, m in models.items():
        torch.manual_seed(0)
        m.apply(lambda mod: mod.reset_parameters() if hasattr(mod, "reset_parameters") else None)
        n_par = sum(p.numel() for p in m.parameters())
        print(f"  training {name} ({n_par} params)")
        hist, t = train_gnn(m, train, f_scale, u_scale, steps=steps, B=4, lr=3e-3, log=500 if not quick else 0)
        e_test, e_fine = rel_errors(m, test, f_scale, u_scale), rel_errors(m, fine, f_scale, u_scale)
        res[name] = dict(params=n_par, train_s=t, test_median=float(np.median(e_test)),
                         test_p90=float(np.percentile(e_test, 90)), fine_median=float(np.median(e_fine)))
        print(f"    rel. L2 error: test meshes median {res[name]['test_median']:.3f} (90th pct"
              f" {res[name]['test_p90']:.3f}) | finer 30x30 meshes median {res[name]['fine_median']:.3f}")
    res["zero prediction"] = dict(test_median=1.0, fine_median=1.0)

    # figure: one test mesh
    g = test[3]
    m = models["MPNN (spatial, edge vectors), 12 layers"]
    with torch.no_grad():
        pred = m(collate([g], f_scale, u_scale)).numpy() * u_scale
    predg = models["GCN (spectral-derived, isotropic), 12 layers"]
    with torch.no_grad():
        pg = predg(collate([g], f_scale, u_scale)).numpy() * u_scale
    fig, axs = plt.subplots(1, 5, figsize=(18, 3.6))
    tri = g["tri"]
    axs[0].triplot(g["P"][:, 0], g["P"][:, 1], tri, lw=0.4, color="0.4"); axs[0].set_title(f"mesh ({len(g['P'])} nodes)")
    for a, v, ttl in [(axs[1], g["f"], "source f"), (axs[2], g["u"], "FEM solution u"),
                      (axs[3], pred, "MPNN prediction"), (axs[4], pg, "GCN prediction")]:
        vmax = np.abs(g["u"]).max() if a is not axs[1] else np.abs(v).max()
        tc = a.tripcolor(g["P"][:, 0], g["P"][:, 1], tri, v, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(ttl); fig.colorbar(tc, ax=a)
    for a in axs:
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    savefig(fig, LESSON, "gnn_poisson.png")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = [k for k in res if k != "zero prediction"]
    xx = np.arange(len(names))
    ax.bar(xx - 0.2, [res[k]["test_median"] for k in names], 0.4, label="test meshes (same size range)")
    ax.bar(xx + 0.2, [res[k]["fine_median"] for k in names], 0.4, label="finer 30x30 meshes")
    ax.axhline(1.0, color="k", ls=":", lw=0.8); ax.text(-0.4, 1.02, "zero prediction", fontsize=8)
    ax.set_xticks(xx, [k.split(",")[0].split(" (")[0] for k in names]); ax.set_ylabel("median rel. L2 error")
    ax.legend(fontsize=8); ax.set_title("learnt Poisson solvers on unseen meshes")
    savefig(fig, LESSON, "gnn_errors.png")
    return res


def main(quick: bool = False) -> dict:
    rng = seed_everything(13)
    set_torch_threads()
    res = fem_part(rng)
    res["gnn"] = gnn_part(rng, quick)
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
