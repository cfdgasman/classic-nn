"""Lesson 16 data source: a lattice-Boltzmann (D2Q9, BGK) solver for flow past a cylinder.

The lattice Boltzmann method evolves particle distribution functions f_i(x, t)
along 9 lattice velocities c_i:

    collide:  f_i* = f_i - (f_i - f_i^eq(rho, u)) / tau            (BGK relaxation)
    stream:   f_i(x + c_i, t + 1) = f_i*(x, t)

with rho = sum_i f_i, rho u = sum_i c_i f_i and the equilibrium
    f_i^eq = w_i rho [1 + 3 c_i.u + 9/2 (c_i.u)^2 - 3/2 |u|^2].
A Chapman-Enskog expansion shows that this recovers the incompressible
Navier-Stokes equations with kinematic viscosity  nu = (tau - 1/2) / 3  (lattice
units) at low Mach number. Boundary conditions: bounce-back on the cylinder
(no-slip), an equilibrium velocity inlet, a zero-gradient outlet and periodic
top/bottom boundaries. It is simple, fully explicit and vectorises well in NumPy.
"""
from __future__ import annotations

import time

import numpy as np

C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def equilibrium(rho, ux, uy):
    cu = 3 * (C[:, 0, None, None] * ux + C[:, 1, None, None] * uy)
    usq = 1.5 * (ux ** 2 + uy ** 2)
    return rho * W[:, None, None] * (1 + cu + 0.5 * cu ** 2 - usq)


def cylinder_flow(nx=400, ny=100, D=16, Re=100.0, U=0.1, n_steps=16000, save_every=25, save_from=6000,
                  probe=None, log=5000):
    """Run the simulation; return saved vorticity snapshots and diagnostics.

    Arrays are indexed [x, y]. The cylinder centre is at (nx/5, ny/2 + 1): the
    one-cell offset breaks the symmetry so vortex shedding starts naturally.
    """
    nu = U * D / Re
    tau = 3 * nu + 0.5
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = nx // 5, ny // 2 + 1
    solid = (X - cx) ** 2 + (Y - cy) ** 2 <= (D / 2) ** 2
    rho = np.ones((nx, ny))
    ux, uy = U * np.ones((nx, ny)), np.zeros((nx, ny))
    # an asymmetric transverse "kick" in the near wake: without it the symmetric wake is a
    # (unstable) steady solution and shedding takes ~25 000 steps to appear from round-off
    kick = np.exp(-((X - cx - 2 * D) ** 2 + (Y - cy - D / 2) ** 2) / (2 * D ** 2))
    uy += 0.5 * U * kick
    ux[solid] = 0
    f = equilibrium(rho, ux, uy)
    probe = probe or (cx + 3 * D, cy)
    snaps, probe_v, t0 = [], [], time.time()
    for n in range(n_steps):
        # outlet: zero-gradient for populations entering from the right
        f[[3, 6, 7], -1] = f[[3, 6, 7], -2]
        rho = f.sum(0)
        ux = (f * C[:, 0, None, None]).sum(0) / rho
        uy = (f * C[:, 1, None, None]).sum(0) / rho
        # inlet: impose the free-stream velocity through the equilibrium
        ux[0], uy[0] = U, 0.0
        rho[0] = (f[[0, 2, 4], 0].sum(0) + 2 * f[[3, 6, 7], 0].sum(0)) / (1 - U)
        feq = equilibrium(rho, ux, uy)
        f[[1, 5, 8], 0] = feq[[1, 5, 8], 0]
        # collision, then bounce-back inside the cylinder (no-slip wall)
        fout = f - (f - feq) / tau
        fout[:, solid] = f[OPP][:, solid]
        # streaming
        for i in range(9):
            f[i] = np.roll(np.roll(fout[i], C[i, 0], axis=0), C[i, 1], axis=1)
        probe_v.append(uy[probe])
        if n >= save_from and (n - save_from) % save_every == 0:
            vort = (np.roll(uy, -1, 0) - np.roll(uy, 1, 0)) / 2 - (np.roll(ux, -1, 1) - np.roll(ux, 1, 1)) / 2
            vort[solid] = 0.0
            snaps.append(vort.astype(np.float32))
        if log and (n + 1) % log == 0:
            print(f"    LBM step {n + 1:6d}/{n_steps}  ({time.time() - t0:.0f}s)")
    probe_v = np.array(probe_v)
    return dict(snaps=np.array(snaps), probe_v=probe_v, nu=nu, tau=tau, D=D, U=U, Re=Re, solid=solid,
                cx=cx, cy=cy, nx=nx, ny=ny, save_every=save_every, time_s=time.time() - t0)


def strouhal(probe_v, D, U, discard=0.5):
    """Dominant frequency of the transverse velocity at a wake probe -> St = f D / U.
    The FFT peak is refined by parabolic interpolation of the log spectrum."""
    v = probe_v[int(discard * len(probe_v)):]
    v = (v - v.mean()) * np.hanning(len(v))
    P = np.abs(np.fft.rfft(v)) ** 2
    k = np.argmax(P[1:]) + 1
    a, b, c = np.log(P[k - 1:k + 2])
    delta = 0.5 * (a - c) / (a - 2 * b + c)
    freq = (k + delta) / len(v)          # cycles per time step
    return freq * D / U


def poiseuille(ny=32, tau=0.8, g=1e-6, n_steps=20000):
    """Body-force-driven channel flow between two walls (verification case).

    Periodic in x, half-way bounce-back walls just outside the first and last
    fluid rows (walls at y = -1/2 and y = ny - 1/2). The force g is added by
    shifting the equilibrium velocity, u_eq = u + tau g (Shan-Chen style forcing);
    the physical velocity is u = (sum c f)/rho + g/2. Exact steady solution:
        u(y) = g / (2 nu) (y + 1/2) (ny - 1/2 - y),   nu = (tau - 1/2) / 3.
    Returns (y, u_numerical, u_exact).
    """
    nx = 3
    rho = np.ones((nx, ny)); ux = np.zeros((nx, ny)); uy = np.zeros((nx, ny))
    f = equilibrium(rho, ux, uy)
    for _ in range(n_steps):
        rho = f.sum(0)
        ux = (f * C[:, 0, None, None]).sum(0) / rho
        uy = (f * C[:, 1, None, None]).sum(0) / rho
        fout = f - (f - equilibrium(rho, ux + tau * g, uy)) / tau
        new = np.empty_like(f)
        for i in range(9):
            # stream in x (periodic); in y stream without wrap-around
            s = np.roll(fout[i], C[i, 0], axis=0)
            if C[i, 1] == 1:
                new[i, :, 1:] = s[:, :-1]
                new[i, :, 0] = fout[OPP[i], :, 0]          # bounce-back at the bottom wall
            elif C[i, 1] == -1:
                new[i, :, :-1] = s[:, 1:]
                new[i, :, -1] = fout[OPP[i], :, -1]        # bounce-back at the top wall
            else:
                new[i] = s
        f = new
    rho = f.sum(0)
    u = (f * C[:, 0, None, None]).sum(0)[1] / rho[1] + g / 2
    nu = (tau - 0.5) / 3
    y = np.arange(ny)
    return y, u, g / (2 * nu) * (y + 0.5) * (ny - 0.5 - y)
