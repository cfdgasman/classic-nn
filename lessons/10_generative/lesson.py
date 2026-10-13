"""Lesson 10 - Generative models: a VAE on MNIST and a minimal diffusion model (DDPM) in 2-D.

Questions this lesson answers
-----------------------------
* How can a network *sample* new data instead of classifying it?
* What is the ELBO, the reparameterisation trick, and what does a VAE latent space look like?
* How does a diffusion model turn noise into data, and what does its network actually learn?
  (The score of the noised data distribution - checked against the exact formula.)
* How do we measure sample quality honestly? (Exact likelihood baselines, MMD, mode coverage.)

Run:  python lessons/10_generative/lesson.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from nnaz.common import banner, save_gif, save_results, savefig, seed_everything, set_torch_threads, setup_matplotlib
from nnaz.data import load_mnist
from nnaz.generative import (VAE, EpsNet, Schedule, ddpm_sample, exact_eps, iwae_nll, mmd_rbf, mode_stats,
                             neg_elbo, ring_of_gaussians, train_ddpm, train_vae)

LESSON = 10
plt = setup_matplotlib()


def vae_part(quick):
    banner("1. A variational autoencoder on binarised MNIST")
    xtr, _ = load_mnist("MNIST", True)
    xte, yte = load_mnist("MNIST", False)
    n = 10000 if quick else 60000
    # binarise (pixel > 127) so p(x|z) is a proper Bernoulli likelihood and the
    # numbers below are true log-likelihoods in nats
    Xtr = torch.as_tensor((xtr[:n] > 127).reshape(n, -1), dtype=torch.float32)
    Xte = torch.as_tensor((xte[:5000] > 127).reshape(5000, -1), dtype=torch.float32)
    # trivial baseline: independent Bernoulli pixels with the training-set means (exact NLL)
    p = Xtr.mean(0).clamp(1e-4, 1 - 1e-4)
    base_nll = float(-(Xte * p.log() + (1 - Xte) * (1 - p).log()).sum(1).mean())
    res = {"baseline_independent_pixels_nll": base_nll}
    print(f"  baseline (independent pixels): NLL = {base_nll:.1f} nats / image")
    models = {}
    for k in (2, 16):
        torch.manual_seed(0)
        vae = VAE(k=k)
        hist, t = train_vae(vae, Xtr, epochs=3 if quick else 20)
        vae.eval()
        with torch.no_grad():
            torch.manual_seed(1)
            ne, rec, kl = neg_elbo(vae, Xte)
            torch.manual_seed(2)
            iw = iwae_nll(vae, Xte[:1000], K=100 if not quick else 10)
        res[f"k{k}"] = dict(neg_elbo=float(ne.mean()), rec=float(rec.mean()), kl=float(kl.mean()),
                            iwae_nll=float(iw.mean()), train_time_s=t)
        print(f"  VAE latent {k:2d}: -ELBO {ne.mean():.1f} = reconstruction {rec.mean():.1f} + KL {kl.mean():.1f};"
              f"  IWAE(K=100) NLL {iw.mean():.1f} nats  ({t:.0f}s)")
        models[k] = (vae, hist)

    vae2 = models[2][0]
    with torch.no_grad():
        mu, _ = vae2.encode(Xte)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    sc = axs[0].scatter(mu[:, 0], mu[:, 1], c=yte[:5000], cmap="tab10", s=3)
    fig.colorbar(sc, ax=axs[0], ticks=range(10)); axs[0].set_title("2-D latent means of test digits")
    # decode a grid of latent points spread by the prior's quantiles
    from scipy.stats import norm
    g = norm.ppf(np.linspace(0.03, 0.97, 15))
    Z = torch.tensor(np.array([[a, b] for b in g[::-1] for a in g]), dtype=torch.float32)
    with torch.no_grad():
        imgs = torch.sigmoid(vae2.dec(Z)).numpy().reshape(15, 15, 28, 28)
    axs[1].imshow(imgs.transpose(0, 2, 1, 3).reshape(15 * 28, 15 * 28), cmap="gray_r")
    axs[1].set_title("decoder output over the latent plane (the learnt manifold)")
    axs[1].axis("off")
    savefig(fig, LESSON, "vae_latent.png")

    vae16 = models[16][0]
    torch.manual_seed(3)
    with torch.no_grad():
        samples = torch.sigmoid(vae16.dec(torch.randn(40, 16))).numpy()
        recon = torch.sigmoid(vae16(Xte[:10])[0]).numpy()
    fig, axs = plt.subplots(6, 10, figsize=(9, 5.8))
    for i, a in enumerate(axs.flat):
        img = Xte[i % 10].numpy() if i < 10 else (recon[i - 10] if i < 20 else samples[i - 20])
        a.imshow(img.reshape(28, 28), cmap="gray_r"); a.axis("off")
    axs[0, 0].set_title("test images", loc="left", fontsize=9)
    axs[1, 0].set_title("VAE reconstructions (k=16)", loc="left", fontsize=9)
    axs[2, 0].set_title("new samples: decode z ~ N(0, I)", loc="left", fontsize=9)
    savefig(fig, LESSON, "vae_samples.png")
    return {"vae": res}


def diffusion_part(quick):
    banner("2. A denoising diffusion model on a ring of 8 Gaussians")
    rng = np.random.default_rng(10)
    X0, means, std = ring_of_gaussians(20000, rng)
    sched = Schedule(T=100)
    torch.manual_seed(0)
    model = EpsNet(sched.T)
    hist = train_ddpm(model, sched, X0, steps=6000 if not quick else 600)

    # (a) how close is the learnt eps to the exact optimum eps*?
    errs = []
    ts = list(range(0, sched.T, 10)) + [sched.T - 1]
    for t in ts:
        ab = float(sched.abar[t])
        x0, _, _ = ring_of_gaussians(4000, rng)
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * rng.normal(size=x0.shape)
        with torch.no_grad():
            e = model(torch.as_tensor(xt, dtype=torch.float32), torch.full((len(xt),), t)).numpy()
        e_star = exact_eps(xt, np.full(len(xt), t), sched, means, std)
        errs.append(float(np.sqrt(((e - e_star) ** 2).sum(1).mean() / (e_star ** 2).sum(1).mean())))
    print("  learnt vs exact noise predictor, relative RMS error at t = "
          + ", ".join(f"{t}: {e:.3f}" for t, e in zip(ts, errs)))

    # (b) samples: learnt model, exact-score sampler, a Gaussian fit, and fresh data
    n = 4000
    learnt_fn = lambda x, t: model(x.float(), torch.full((len(x),), t)).double()
    exact_fn = lambda x, t: torch.as_tensor(exact_eps(x.numpy(), np.full(len(x), t), sched, means, std))
    keep = (100, 60, 40, 20, 10, 0)
    S_model, traj = ddpm_sample(learnt_fn, sched, n, seed=1, keep=set(range(0, 101)))
    S_exact, _ = ddpm_sample(exact_fn, sched, n, seed=1)
    ref, _, _ = ring_of_gaussians(n, rng)
    fresh, _, _ = ring_of_gaussians(n, rng)
    mu, cov = X0.mean(0), np.cov(X0.T)
    S_gauss = rng.multivariate_normal(mu, cov, n)
    out = {}
    for name, S in [("fresh data (metric noise floor)", fresh), ("DDPM with exact score", S_exact),
                    ("DDPM, learnt network", S_model), ("Gaussian fit (baseline)", S_gauss)]:
        frac, modes = mode_stats(S, means, std)
        out[name] = dict(mmd=mmd_rbf(S, ref), frac_near_mode=frac, modes_hit=modes)
        print(f"  {name:32s} MMD^2 {out[name]['mmd']:.2e}   within 3 std of a mode {frac:.3f}   modes hit {modes}/8")

    fig, axs = plt.subplots(1, 6, figsize=(15, 2.8), sharex=True, sharey=True)
    for a, t in zip(axs, keep):
        ab = float(sched.abar[t - 1]) if t > 0 else 1.0
        fwd = np.sqrt(ab) * X0[:2000] + np.sqrt(1 - ab) * rng.normal(size=(2000, 2))
        a.scatter(*fwd.T, s=1, alpha=0.4)
        a.set_title(f"forward, t={t}"); a.set_xlim(-3.2, 3.2); a.set_ylim(-3.2, 3.2); a.set_aspect("equal")
    savefig(fig, LESSON, "diffusion_forward.png")

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
    for a, (S, ttl) in zip(axs, [(ref, "data"), (S_gauss, "Gaussian fit"), (S_exact, "DDPM, exact score"),
                                 (S_model, "DDPM, learnt network")]):
        a.scatter(*S[:2000].T, s=1.5, alpha=0.5); a.set_title(ttl); a.set_aspect("equal")
    savefig(fig, LESSON, "diffusion_samples.png")

    # score field: learnt vs exact at an intermediate time
    t = 30
    g1 = np.linspace(-3, 3, 21)
    G = np.array([[a, b] for b in g1 for a in g1])
    with torch.no_grad():
        eL = model(torch.as_tensor(G, dtype=torch.float32), torch.full((len(G),), t)).numpy()
    eE = exact_eps(G, np.full(len(G), t), sched, means, std)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    for a, e, ttl in [(axs[0], eE, "exact"), (axs[1], eL, "learnt")]:
        a.quiver(G[:, 0], G[:, 1], -e[:, 0], -e[:, 1], color="tab:blue")
        a.scatter(*(np.sqrt(float(sched.abar[t])) * means).T, c="r", s=20)
        a.set_title(fr"{ttl} score direction $-\epsilon(x, t={t})$"); a.set_aspect("equal")
    savefig(fig, LESSON, "diffusion_score_field.png")

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(4, 4))
    sc = ax.scatter(*traj[100][:1500].T, s=2, alpha=0.6)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ttl = ax.set_title("")
    frames = list(range(100, -1, -2)) + [0] * 8

    def draw(k):
        sc.set_offsets(traj[k][:1500]); ttl.set_text(f"reverse diffusion, t = {k}")
        return sc,

    save_gif(FuncAnimation(fig, draw, frames=frames), LESSON, "diffusion_reverse.gif", fps=12)
    return {"diffusion": dict(eps_rel_err={str(t): e for t, e in zip(ts, errs)}, samples=out,
                              final_train_loss=hist[-1])}


def main(quick: bool = False) -> dict:
    seed_everything(10)
    set_torch_threads()
    res = vae_part(quick)
    res.update(diffusion_part(quick))
    save_results(LESSON, res)
    return res


if __name__ == "__main__":
    main()
