"""Lesson 16 - Reduced-order modelling for CFD: POD vs a convolutional autoencoder on a cylinder wake.

Questions this lesson answers
-----------------------------
* How do engineers compress a flow simulation into a handful of numbers? (POD / SVD.)
* Why does a *nonlinear* autoencoder need far fewer latent variables than POD for a
  periodic wake? (A limit cycle is a curve; POD needs a pair of modes per harmonic.)
* How can we forecast the flow cheaply in the reduced space (DMD vs a neural latent map)?

Data: our own lattice-Boltzmann simulation (D2Q9) of flow past a cylinder at Re = 100,
validated through its Strouhal number against Williamson's correlation.

Run:  python lessons/16_rom_cfd/lesson.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from nnaz.common import DATA, banner, save_gif, save_results, savefig, seed_everything, set_torch_threads, setup_matplotlib
from nnaz.lbm import cylinder_flow, strouhal
from nnaz.rom import ConvAE, LatentStep, crop, dmd_fit, pod, train_ae, train_latent

LESSON = 16
plt = setup_matplotlib()


def simulate(quick):
    banner("1. The CFD data: lattice-Boltzmann flow past a cylinder, Re = 100")
    path = DATA / f"lbm_cylinder{'_quick' if quick else ''}.npz"
    if path.exists():
        d = dict(np.load(path, allow_pickle=True))
        d = {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}
    else:
        kw = dict(n_steps=6000, save_from=3000) if quick else {}
        d = cylinder_flow(**kw)
        np.savez_compressed(path, **d)
    St = strouhal(d["probe_v"], d["D"], d["U"])
    St_w = 0.2175 - 5.1064 / d["Re"]          # Williamson (1988) fit, unconfined, 49 < Re < 178
    blockage = d["D"] / d["ny"]
    print(f"  grid {d['nx']} x {d['ny']}, D = {d['D']}, U = {d['U']}, tau = {d['tau']:.3f}, blockage {blockage:.2f};"
          f" {len(d['snaps'])} snapshots; {d['time_s']:.0f}s")
    print(f"  Strouhal number St = f D / U = {St:.4f}  (Williamson, unconfined: {St_w:.4f};"
          f" difference {100 * (St / St_w - 1):+.1f} %)")

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 2.2))
    v = np.percentile(np.abs(d["snaps"][-1]), 99)
    im = ax.imshow(d["snaps"][0].T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.contourf(d["solid"].T, levels=[0.5, 1], colors="k")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    def draw(k):
        im.set_data(d["snaps"][k].T); ax.set_title(f"vorticity, Re = 100 (lattice Boltzmann), snapshot {k}")
        return im,

    save_gif(FuncAnimation(fig, draw, frames=range(0, min(160, len(d["snaps"])), 2)), LESSON, "cylinder_wake.gif", fps=15, dpi=60)
    return d, dict(strouhal=St, strouhal_williamson=St_w, blockage=blockage, lbm_time_s=d["time_s"], tau=d["tau"])


def rom_part(d, quick):
    banner("2. Compression: POD (linear) vs a convolutional autoencoder (nonlinear)")
    F = crop(d["snaps"], int(d["cx"]), int(d["D"]))
    F = F / F.std()
    m = len(F)
    n_tr = int(0.75 * m)
    Ftr, Fte = F[:n_tr], F[n_tr:]
    Str, Ste = Ftr.reshape(n_tr, -1), Fte.reshape(len(Fte), -1)
    mu, U, s, A = pod(Str)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    rel = lambda P, T: float(np.linalg.norm(P - T) / np.linalg.norm(T - mu))
    res = {"n_snapshots": m, "pod_energy": energy[:20].tolist()}
    ranks = [1, 2, 4, 8, 16]
    res["pod_test_err"] = {r: rel((Ste - mu) @ U[:, :r] @ U[:, :r].T + mu, Ste) for r in ranks}
    print("  POD test error (relative to the fluctuation): "
          + ", ".join(f"r={r}: {e:.3f}" for r, e in res["pod_test_err"].items()))
    res["ae_test_err"], aes = {}, {}
    for r in (1, 2, 4):
        torch.manual_seed(0)
        ae = ConvAE(r)
        train_ae(ae, Ftr, epochs=60 if quick else 600)
        with torch.no_grad():
            rec = ae(torch.as_tensor(Fte[:, None], dtype=torch.float32)).numpy()[:, 0]
        res["ae_test_err"][r] = rel(rec.reshape(len(Fte), -1), Ste)
        aes[r] = ae
        print(f"  conv autoencoder, latent {r}: test error {res['ae_test_err'][r]:.3f}")

    fig, axs = plt.subplots(2, 3, figsize=(14, 4.6))
    for k in range(6):
        a = axs.flat[k]
        mode = U[:, k].reshape(F.shape[1:])
        v = np.abs(mode).max()
        a.imshow(mode, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(f"POD mode {k + 1} ({100 * s[k] ** 2 / np.sum(s ** 2):.1f} % energy)", fontsize=9)
    savefig(fig, LESSON, "pod_modes.png")

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    axs[0].semilogy(np.arange(1, 31), 1 - energy[:30] + 1e-12, "o-", ms=3)
    axs[0].set_xlabel("number of POD modes r"); axs[0].set_ylabel("unexplained energy"); axs[0].set_title("POD spectrum (modes come in pairs)")
    axs[1].semilogy(ranks, [res["pod_test_err"][r] for r in ranks], "o-", label="POD")
    axs[1].semilogy(list(res["ae_test_err"]), list(res["ae_test_err"].values()), "s-", label="conv autoencoder")
    axs[1].set_xscale("log", base=2); axs[1].set_xlabel("latent dimension"); axs[1].set_ylabel("test relative error")
    axs[1].legend(); axs[1].set_title("compression of unseen snapshots")
    with torch.no_grad():
        Z = aes[2].enc(torch.as_tensor(F[:, None], dtype=torch.float32)).numpy()
    axs[2].plot(A[:, 0], A[:, 1], ".", ms=3, label="POD a1, a2 (scaled)", alpha=0.6)
    sc = axs[2].scatter(Z[:, 0] / Z[:, 0].std() * A[:, 0].std(), Z[:, 1] / Z[:, 1].std() * A[:, 1].std(),
                        c=np.arange(m), s=4, cmap="viridis", label="autoencoder z1, z2")
    axs[2].set_title("latent space: the limit cycle is a closed curve"); axs[2].legend(fontsize=7)
    savefig(fig, LESSON, "rom_compression.png")

    k = n_tr + 7
    with torch.no_grad():
        rec2 = aes[2](torch.as_tensor(F[k][None, None], dtype=torch.float32)).numpy()[0, 0]
    pod2 = ((F[k].ravel() - mu) @ U[:, :2] @ U[:, :2].T + mu).reshape(F.shape[1:])
    pod8 = ((F[k].ravel() - mu) @ U[:, :8] @ U[:, :8].T + mu).reshape(F.shape[1:])
    fig, axs = plt.subplots(1, 4, figsize=(16, 2.4))
    v = np.abs(F[k]).max()
    for a, img, ttl in [(axs[0], F[k], "LBM snapshot (test)"), (axs[1], pod2, "POD, 2 modes"),
                        (axs[2], pod8, "POD, 8 modes"), (axs[3], rec2, "conv autoencoder, latent 2")]:
        a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v); a.set_title(ttl); a.axis("off")
    savefig(fig, LESSON, "rom_reconstruction.png")
    return res, (F, n_tr, mu, U, aes)


def dynamics_part(F, n_tr, mu, U, aes, quick):
    banner("3. Forecasting in the reduced space: DMD on POD coefficients vs a neural latent map")
    S = F.reshape(len(F), -1)
    out = {}
    horizon = len(F) - n_tr
    truth = S[n_tr:]
    rel = lambda P: float(np.linalg.norm(P - truth) / np.linalg.norm(truth - mu))
    for r in (2, 8):
        Acoef = (S - mu) @ U[:, :r]
        M = dmd_fit(Acoef[:n_tr])
        a = Acoef[n_tr - 1]
        pred = []
        for _ in range(horizon):
            a = a @ M
            pred.append(a)
        P = np.array(pred) @ U[:, :r].T + mu
        eig = np.linalg.eigvals(M)
        out[f"DMD on {r} POD modes"] = rel(P)
        out[f"DMD r={r} max |eigenvalue|"] = float(np.abs(eig).max())
    ae = aes[2]
    with torch.no_grad():
        Z = ae.enc(torch.as_tensor(F[:, None], dtype=torch.float32)).numpy()
    zs, zm = Z[:n_tr].std(0), Z[:n_tr].mean(0)
    step = LatentStep(2)
    train_latent(step, (Z[:n_tr] - zm) / zs, epochs=300 if quick else 3000)
    z = torch.as_tensor((Z[n_tr - 1] - zm) / zs, dtype=torch.float32)[None]
    preds = []
    with torch.no_grad():
        for _ in range(horizon):
            z = step(z)
            preds.append(ae.dec(z * torch.as_tensor(zs) + torch.as_tensor(zm)).numpy()[0, 0].ravel())
    out["autoencoder (latent 2) + neural latent map"] = rel(np.array(preds))
    # reference: the best the latent-2 autoencoder can do (reconstruction of the true test snapshots)
    with torch.no_grad():
        rec = ae(torch.as_tensor(F[n_tr:, None], dtype=torch.float32)).numpy().reshape(horizon, -1)
    out["autoencoder reconstruction (no forecasting)"] = rel(rec)
    out["mean flow (trivial baseline)"] = rel(np.broadcast_to(mu, truth.shape))
    for k, v in out.items():
        if "eigen" not in k:
            print(f"  {k:48s} relative error over the {horizon}-snapshot forecast: {v:.3f}")
    return out


def main(quick: bool = False) -> dict:
    seed_everything(16)
    set_torch_threads()
    d, res = simulate(quick)
    r2, (F, n_tr, mu, U, aes) = rom_part(d, quick)
    res.update(r2)
    res["forecast"] = dynamics_part(F, n_tr, mu, U, aes, quick)
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
