"""Viscous Burgers equation: exact solution (Cole-Hopf) and a finite-difference solver.

    u_t + u u_x = nu u_xx,   x in [-1, 1],  t > 0,
    u(x, 0) = -A sin(pi x),  u(-1, t) = u(1, t) = 0.

Cole-Hopf. The substitution u = -2 nu phi_x / phi turns Burgers into the heat
equation phi_t = nu phi_xx, whose solution is a Gaussian convolution. For this
initial condition (Basdevant et al. 1986):

    u(x,t) = - Int sin(pi(x-eta)) f(x-eta) exp(-eta^2 / 4 nu t) d eta
             / Int f(x-eta) exp(-eta^2 / 4 nu t) d eta,
    f(y) = exp( -A cos(pi y) / (2 pi nu) ).

With eta = sqrt(4 nu t) z both integrals have the weight e^{-z^2}, so
Gauss-Hermite quadrature evaluates them to machine precision. f can be as
large as e^{A/(2 pi nu)} (= e^50 for the classic nu = 0.01/pi), so we work with
log f and subtract its maximum before exponentiating.
"""
from __future__ import annotations

import numpy as np

NU_CLASSIC = 0.01 / np.pi


def cole_hopf(x, t, nu=NU_CLASSIC, A=1.0, n_quad=200):
    """Exact solution on arbitrary broadcastable arrays x, t (t > 0 where needed)."""
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    z, w = np.polynomial.hermite.hermgauss(n_quad)
    out = np.empty(x.shape)
    t0 = t <= 0
    out[t0] = -A * np.sin(np.pi * x[t0])
    xs, ts = x[~t0][..., None], t[~t0][..., None]
    y = xs - np.sqrt(4 * nu * ts) * z                         # x - eta at the quadrature nodes
    logf = -A * np.cos(np.pi * y) / (2 * np.pi * nu)
    wf = w * np.exp(logf - logf.max(-1, keepdims=True))       # overflow-free
    out[~t0] = -A * (wf * np.sin(np.pi * y)).sum(-1) / wf.sum(-1)
    return out


def fd_solve(nu, A, T, N=512, cfl=0.4, save_times=None):
    """Second-order central finite differences + classical RK4 on a uniform grid.

    Because -A sin(pi x) is odd and 2-periodic, the Dirichlet problem on [-1, 1] is
    the same as the periodic one, so we use a periodic grid x_j = -1 + 2 j / N
    (j = 0..N-1). Vectorised over parameters: ``nu`` and ``A`` may be arrays of
    shape [B]; the solution has shape [B, N].

    Semi-discretisation (conservative flux form of the convection term):
        du_j/dt = -(F_{j+1} - F_{j-1}) / (2 dx) + nu (u_{j+1} - 2 u_j + u_{j-1}) / dx^2,
        F = u^2 / 2.
    Central convection is non-oscillatory when the cell Reynolds number
    Re_dx = max|u| dx / nu < 2; the time step obeys both the diffusive
    (dt <= 0.5 dx^2 / nu... RK4 allows ~0.7) and the convective (|u| dt / dx <= ~1) limits.
    """
    nu = np.atleast_1d(np.asarray(nu, float))[:, None]
    A = np.atleast_1d(np.asarray(A, float))[:, None]
    x = -1 + 2 * np.arange(N) / N
    dx = 2 / N
    u = -A * np.sin(np.pi * x)[None]
    re_dx = float((A * dx / nu).max())
    dt = cfl * min(dx ** 2 / nu.max(), dx / A.max())
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps

    def rhs(v):
        F = 0.5 * v * v
        return (-(np.roll(F, -1, 1) - np.roll(F, 1, 1)) / (2 * dx)
                + nu * (np.roll(v, -1, 1) - 2 * v + np.roll(v, 1, 1)) / dx ** 2)

    saves, save_idx = [], set()
    if save_times is not None:
        save_idx = {int(round(s / dt)): s for s in save_times}
    for n in range(n_steps):
        if n in save_idx:
            saves.append(u.copy())
        k1 = rhs(u); k2 = rhs(u + 0.5 * dt * k1); k3 = rhs(u + 0.5 * dt * k2); k4 = rhs(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if save_times is not None and n_steps in save_idx:
        saves.append(u.copy())
    return x, u, dict(dt=dt, n_steps=n_steps, cell_reynolds=re_dx, saves=saves)
