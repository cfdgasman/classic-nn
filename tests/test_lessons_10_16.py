"""Lessons 10-16: generative models, SOM/Hopfield, DNC, GNN, neural operators,
Hamiltonian nets / neural ODEs, lattice Boltzmann + ROM."""
import numpy as np
import torch

from nnaz.burgers import cole_hopf
from nnaz.data import load_mnist
from nnaz.dnc import DNC, allocation, train_copy
from nnaz.generative import VAE, EpsNet, Schedule, exact_eps, neg_elbo, ring_of_gaussians, train_vae
from nnaz.gnn import MPNN, fem_poisson, make_graph, rel_errors, square_mesh, train_gnn
from nnaz.lbm import C, equilibrium
from nnaz.operators import FNO1d, burgers_spectral, grf_1d, rel_l2, train_operator
from nnaz.physics_nets import HNN, pendulum_data, train_field
from nnaz.rom import dmd_fit, pod
from nnaz.som_hopfield import SOM, hebbian_weights, hopfield_recall, projection_weights

torch.set_num_threads(2)


# ---------------------------------------------------------------- lesson 10
def test_vae_beats_independent_pixel_baseline():
    x, _ = load_mnist("MNIST", True)
    X = torch.as_tensor((x[:4000] > 127).reshape(4000, -1), dtype=torch.float32)
    p = X.mean(0).clamp(1e-4, 1 - 1e-4)
    baseline = float(-(X * p.log() + (1 - X) * (1 - p).log()).sum(1).mean())
    torch.manual_seed(0)
    vae = VAE(k=8, h=256)
    hist, _ = train_vae(vae, X, epochs=4)
    assert hist[-1] < hist[0]
    with torch.no_grad():
        assert float(neg_elbo(vae, X)[0].mean()) < baseline      # an upper bound on the NLL beats the exact baseline


def test_exact_diffusion_score_matches_autograd():
    """eps* = -sqrt(1-abar) grad log q_t for the Gaussian mixture, checked against autograd."""
    rng = np.random.default_rng(0)
    _, means, std = ring_of_gaussians(10, rng)
    sched = Schedule()
    x = rng.normal(size=(50, 2)) * 2
    for t in (0, 30, 99):
        ab = float(sched.abar[t])
        s2 = ab * std ** 2 + 1 - ab
        xt = torch.tensor(x, requires_grad=True)
        mu = torch.tensor(np.sqrt(ab) * means)
        logq = torch.logsumexp(-((xt[:, None] - mu[None]) ** 2).sum(-1) / (2 * s2), 1).sum()
        g = torch.autograd.grad(logq, xt)[0].numpy()
        assert np.abs(exact_eps(x, np.full(50, t), sched, means, std) + np.sqrt(1 - ab) * g).max() < 1e-10


# ---------------------------------------------------------------- lesson 11
def test_som_orders_and_hopfield_recalls():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(1500, 2))
    som = SOM(8, 8, 2, rng)
    som.train_online(X, epochs=2, rng=rng, sigma0=4.0)
    assert som.topographic_error(X) < 0.05 and som.quantisation_error(X) < 0.2
    P = np.where(rng.random((3, 100)) < 0.5, 1.0, -1.0)
    P[1] = P[0].copy(); P[1][:30] *= -1                           # strongly correlated pair
    for W in (projection_weights(P),):
        for mu in range(3):
            probe = P[mu].copy(); probe[rng.random(100) < 0.1] *= -1
            s, E, _ = hopfield_recall(W, probe, rng)
            assert np.all(np.diff(E) <= 1e-9)                    # energy never increases
            assert np.mean(s == P[mu]) > 0.97
    s, E, _ = hopfield_recall(hebbian_weights(P), P[2], rng)
    assert np.all(np.diff(E) <= 1e-9)


# ---------------------------------------------------------------- lesson 12
def test_dnc_allocation_and_learning_signal():
    a = allocation(torch.tensor([[0.9, 0.1, 0.5, 0.0]]))[0]
    assert int(torch.argmax(a)) == 3 and abs(float(a.sum()) - 1) < 1e-3
    torch.manual_seed(0)
    dnc = DNC(7, 6, hidden=32, N=16, W=8)
    h = train_copy(dnc, 6, steps=200, B=16, Lmax=4, lr=3e-3, log=0)
    assert h["loss"][-1] < h["loss"][0]


# ---------------------------------------------------------------- lesson 13
def test_fem_order_and_gnn_beats_zero():
    rng = np.random.default_rng(0)
    errs = []
    for n in (17, 33):
        P, tri, bnd = square_mesh(n, rng)
        ue = np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])
        u = fem_poisson(P, tri, bnd, 2 * np.pi ** 2 * ue)
        errs.append(np.linalg.norm(u - ue) / np.linalg.norm(ue))
    assert 3.0 < errs[0] / errs[1] < 5.5
    graphs = [make_graph(10, rng) for _ in range(30)]
    fs = np.std(np.concatenate([g["f"] for g in graphs])); us = np.std(np.concatenate([g["u"] for g in graphs]))
    torch.manual_seed(0)
    m = MPNN(K=4, h=32)
    hist, _ = train_gnn(m, graphs, fs, us, steps=150, B=4, log=0)
    assert np.mean(hist[-20:]) < np.mean(hist[:20])
    assert np.median(rel_errors(m, graphs[:10], fs, us)) < 0.9      # zero prediction scores 1.0


# ---------------------------------------------------------------- lesson 14
def test_spectral_solver_and_fno():
    x = -1 + 2 * np.arange(256) / 256
    u, _ = burgers_spectral(-np.sin(np.pi * x), 0.05, 1.0, Lx=2.0)
    assert np.abs(u[0] - cole_hopf(x, 1.0, 0.05)).max() < 1e-8
    rng = np.random.default_rng(0)
    u0 = grf_1d(120, 128, rng)
    uT, _ = burgers_spectral(u0, 0.05, 0.5)
    U0, UT = torch.as_tensor(u0, dtype=torch.float32), torch.as_tensor(uT, dtype=torch.float32)
    torch.manual_seed(0)
    fno = FNO1d(modes=8, width=24)
    hist, _ = train_operator(fno, U0[:100], UT[:100], epochs=15, batch=10, lr=3e-3, log=0)
    assert hist[-1] < hist[0]
    with torch.no_grad():
        assert float(rel_l2(fno(U0[100:]), UT[100:]).mean()) < float(rel_l2(U0[100:], UT[100:]).mean())
        assert fno(torch.zeros(2, 256)).shape == (2, 256)            # any resolution


# ---------------------------------------------------------------- lesson 15
def test_hnn_field_conserves_its_energy():
    rng = np.random.default_rng(0)
    X, dX = pendulum_data(10, rng)
    torch.manual_seed(0)
    hnn = HNN()
    hist = train_field(hnn, X, dX, steps=300)
    assert np.mean(hist[-30:]) < np.mean(hist[:30])
    x = torch.randn(64, 2, requires_grad=True)
    H = hnn.H(x).sum()
    gradH = torch.autograd.grad(H, x)[0]
    # dH/dt along the learnt flow = grad H . (dH/dp, -dH/dq) = 0 identically
    assert float((gradH * hnn(x.detach())).sum(1).abs().max()) < 1e-5


# ---------------------------------------------------------------- lesson 16
def test_lbm_equilibrium_moments_and_pod_dmd():
    rng = np.random.default_rng(0)
    rho = 1 + 0.1 * rng.random((5, 4))
    ux, uy = 0.05 * rng.normal(size=(2, 5, 4))
    f = equilibrium(rho, ux, uy)
    assert np.allclose(f.sum(0), rho)
    assert np.allclose((f * C[:, 0, None, None]).sum(0), rho * ux)
    assert np.allclose((f * C[:, 1, None, None]).sum(0), rho * uy)
    # a travelling wave: POD needs exactly 2 modes, DMD recovers the rotation frequency
    x, w = np.linspace(0, 2 * np.pi, 64, endpoint=False), 0.3
    S = np.array([np.sin(x - w * n) for n in range(80)])
    mu, U, s, A = pod(S)
    assert s[2] / s[0] < 1e-10
    # DMD on the uncentred snapshots (mean removal over a non-integer number of periods
    # would make the coefficient dynamics affine instead of linear)
    Uu = np.linalg.svd(S.T, full_matrices=False)[0][:, :2]
    M = dmd_fit(S @ Uu)
    assert abs(np.abs(np.angle(np.linalg.eigvals(M))).max() - w) < 1e-8
