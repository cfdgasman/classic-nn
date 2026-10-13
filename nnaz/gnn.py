"""Lesson 13: spatial graph neural networks on unstructured meshes.

A mesh is a graph: nodes = mesh vertices (with features such as a source term
and coordinates), edges = mesh edges (with features such as the relative
position x_j - x_i). Grids are not required - which is exactly what CFD and FEM
engineers need.

Two families of GNN layers:

* SPECTRAL-derived, e.g. the graph convolutional network (GCN, Kipf & Welling 2017).
  It comes from filtering signals with polynomials of the graph Laplacian and
  reduces to  H' = act( D^-1/2 (A + I) D^-1/2 H W ):  every neighbour is averaged
  with a fixed, *isotropic* weight - the layer cannot tell a neighbour to the
  east from one to the west.
* SPATIAL message passing (Gilmer et al. 2017; Battaglia et al. 2018; MeshGraphNets,
  Pfaff et al. 2021). For each edge a learnt message depends on both nodes AND the
  edge geometry, messages are summed at the receiver, and the node is updated:

      m_ij = phi_e( h_i, h_j, e_ij )          e_ij = (x_j - x_i, |x_j - x_i|)
      h_i <- h_i + phi_v( h_i, sum_j m_ij )   (residual update)

  With edge vectors the layer can build anisotropic stencils - a learnt analogue
  of a finite-volume flux or a finite-element stiffness row.

Information travels one edge per layer, so K layers see K-hop neighbourhoods;
an elliptic problem (Poisson) couples every node to every other node, so depth
and mesh size matter - an honest limitation measured below.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.spatial import Delaunay
from torch import nn


# ================================================================ meshes and FEM
def square_mesh(n, rng, jitter=0.3):
    """Jittered-grid points on [0,1]^2 (boundary nodes kept on the boundary),
    Delaunay-triangulated. n = points per side."""
    g = np.linspace(0, 1, n)
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    bnd = (P[:, 0] == 0) | (P[:, 0] == 1) | (P[:, 1] == 0) | (P[:, 1] == 1)
    h = 1 / (n - 1)
    P[~bnd] += rng.uniform(-jitter * h, jitter * h, size=(int((~bnd).sum()), 2))
    tri = Delaunay(P).simplices
    return P, tri, bnd


def fem_poisson(P, tri, bnd, f):
    """Linear (P1) finite elements for  -Laplace(u) = f  in the domain, u = 0 on the boundary.

    On a triangle with vertices p0, p1, p2 and area |T| the hat functions have
    constant gradients; the local stiffness is K_ab = |T| grad(phi_a).grad(phi_b)
    and the consistent mass matrix is |T|/12 [[2,1,1],[1,2,1],[1,1,2]]. The load
    vector uses the nodal interpolant of f:  F = M f.
    """
    n = len(P)
    rows, cols, kv, mv = [], [], [], []
    Mloc = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    for t in tri:
        p = P[t]
        B = np.array([[p[1, 0] - p[0, 0], p[2, 0] - p[0, 0]], [p[1, 1] - p[0, 1], p[2, 1] - p[0, 1]]])
        area = 0.5 * abs(np.linalg.det(B))
        # gradients of the three hat functions: G = [[-1,-1],[1,0],[0,1]] @ inv(B)
        G = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]) @ np.linalg.inv(B)
        K = area * G @ G.T
        for a in range(3):
            for b in range(3):
                rows.append(t[a]); cols.append(t[b]); kv.append(K[a, b]); mv.append(area * Mloc[a, b])
    K = sp.csr_matrix((kv, (rows, cols)), shape=(n, n))
    M = sp.csr_matrix((mv, (rows, cols)), shape=(n, n))
    F = M @ f
    inner = np.where(~bnd)[0]
    u = np.zeros(n)
    u[inner] = spla.spsolve(K[inner][:, inner].tocsc(), F[inner])
    return u


def mesh_edges(tri):
    """Directed edge list (both directions) of a triangulation."""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    return np.concatenate([e, e[:, ::-1]]).T  # [2, E]: senders, receivers


def random_source(P, rng, n_bumps=(1, 4)):
    """Sum of random Gaussian bumps (positive and negative)."""
    f = np.zeros(len(P))
    for _ in range(rng.integers(*n_bumps)):
        c = rng.uniform(0.15, 0.85, 2)
        w = rng.uniform(0.05, 0.2)
        a = rng.uniform(-1, 1) * 50
        f += a * np.exp(-((P - c) ** 2).sum(1) / (2 * w * w))
    return f


def make_graph(n, rng):
    P, tri, bnd = square_mesh(n, rng)
    f = random_source(P, rng)
    u = fem_poisson(P, tri, bnd, f)
    E = mesh_edges(tri)
    return dict(P=P, tri=tri, bnd=bnd, f=f, u=u, E=E)


# ================================================================ batching
def collate(graphs, f_scale, u_scale):
    """Concatenate graphs into one big disconnected graph (indices shifted)."""
    xs, es, eas, ys, bs, off = [], [], [], [], [], 0
    for g in graphs:
        P = g["P"]
        xs.append(np.c_[g["f"] / f_scale, P, g["bnd"].astype(float)])
        s, r = g["E"]
        d = P[r] - P[s]
        eas.append(np.c_[d, np.linalg.norm(d, axis=1)])
        es.append(g["E"] + off)
        ys.append(g["u"] / u_scale)
        bs.append(g["bnd"])
        off += len(P)
    return dict(x=torch.as_tensor(np.concatenate(xs), dtype=torch.float32),
                edge=torch.as_tensor(np.concatenate(es, 1), dtype=torch.long),
                ea=torch.as_tensor(np.concatenate(eas), dtype=torch.float32),
                y=torch.as_tensor(np.concatenate(ys), dtype=torch.float32),
                bnd=torch.as_tensor(np.concatenate(bs)))


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


# ================================================================ models
class MPNN(nn.Module):
    """Encode - process (K message-passing steps) - decode."""

    def __init__(self, n_in=4, e_in=3, h=64, K=12):
        super().__init__()
        self.enc_v, self.enc_e = mlp(n_in, h, h), mlp(e_in, h, h)
        self.edge_fns = nn.ModuleList([mlp(3 * h, h, h) for _ in range(K)])
        self.node_fns = nn.ModuleList([mlp(2 * h, h, h) for _ in range(K)])
        self.dec = mlp(h, 1, h)

    def forward(self, g):
        s, r = g["edge"]
        h, e = self.enc_v(g["x"]), self.enc_e(g["ea"])
        for fe, fv in zip(self.edge_fns, self.node_fns):
            e = e + fe(torch.cat([h[s], h[r], e], 1))              # message on each edge
            agg = torch.zeros_like(h).index_add_(0, r, e)           # sum at the receiver
            h = h + fv(torch.cat([h, agg], 1))                      # residual node update
        u = self.dec(h).squeeze(1)
        return u * (~g["bnd"])                                     # Dirichlet u = 0 built in


class GCN(nn.Module):
    """Kipf & Welling GCN: isotropic normalised-adjacency propagation, no edge geometry."""

    def __init__(self, n_in=4, h=64, K=12):
        super().__init__()
        self.inp = nn.Linear(n_in, h)
        self.lins = nn.ModuleList([nn.Linear(h, h) for _ in range(K)])
        self.out = mlp(h, 1, h)

    def forward(self, g):
        s, r = g["edge"]
        n = len(g["x"])
        deg = torch.ones(n).index_add_(0, r, torch.ones(len(r)))    # degree incl. self loop
        w = (deg[s] * deg[r]).rsqrt()
        h = torch.relu(self.inp(g["x"]))
        for lin in self.lins:
            hw = lin(h)
            agg = hw / deg[:, None]                                  # self loop term
            agg = agg.index_add(0, r, hw[s] * w[:, None])
            h = h + torch.relu(agg)
        return self.out(h).squeeze(1) * (~g["bnd"])


class PointMLP(nn.Module):
    """Baseline: each node on its own (f, x, y) - no neighbour information at all."""

    def __init__(self, n_in=4, h=128):
        super().__init__()
        self.net = mlp(n_in, 1, h)

    def forward(self, g):
        return self.net(g["x"]).squeeze(1) * (~g["bnd"])


def train_gnn(model, train_graphs, f_scale, u_scale, steps=2000, B=8, lr=1e-3, seed=0, log=500):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist, t0 = [], time.time()
    batches = {}
    for s in range(steps):
        idx = tuple(sorted(rng.choice(len(train_graphs), B, replace=False)))
        g = collate([train_graphs[i] for i in idx], f_scale, u_scale)
        loss = ((model(g) - g["y"]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(loss.item())
        if log and (s + 1) % log == 0:
            print(f"    step {s + 1:5d}  mse {np.mean(hist[-log:]):.2e}  ({time.time() - t0:.0f}s)")
    return hist, time.time() - t0


@torch.no_grad()
def rel_errors(model, graphs, f_scale, u_scale):
    out = []
    for g in graphs:
        b = collate([g], f_scale, u_scale)
        p = model(b).numpy() * u_scale
        out.append(np.linalg.norm(p - g["u"]) / np.linalg.norm(g["u"]))
    return np.array(out)
