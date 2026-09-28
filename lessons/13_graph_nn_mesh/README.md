<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 12_dnc_memory](../12_dnc_memory/README.md) · [course overview](../../README.md) · [14_neural_operators →](../14_neural_operators/README.md)

# Lesson 13 - Spatial graph neural networks on unstructured meshes

📄 [`lessons/13_graph_nn_mesh/lesson.py`](../../lessons/13_graph_nn_mesh/lesson.py) · 📓 [`notebooks/13_graph_nn_mesh.ipynb`](../../notebooks/13_graph_nn_mesh.ipynb) · code: [`nnaz/gnn.py`](../../nnaz/gnn.py)

CFD and FEM meshes are not images, so CNNs do not apply directly. A mesh *is* a **graph**:
vertices carry node features (source term, coordinates, boundary flag), and mesh edges carry
edge features (the relative position $x_j-x_i$ and its length).

### The reference solver: P1 finite elements

We solve $-\Delta u=f$ on the unit square with $u=0$ on the boundary, on jittered Delaunay
meshes. On each triangle $T$ the hat functions have constant gradients $G$, the element
stiffness is $K_T=|T|\,GG^\top$, and the consistent mass matrix is
$M_T=\frac{|T|}{12}\begin{psmallmatrix}2&1&1\\1&2&1\\1&1&2\end{psmallmatrix}$. We assemble,
apply the Dirichlet condition, and solve the sparse system. The solver is **verified** with the
manufactured solution $u=\sin\pi x\sin\pi y$ (so $f=2\pi^2u$): the error drops by 4× per mesh
halving (second order).

### Two families of GNN layers

* **Spectral-derived: GCN** (Kipf and Welling, 2017). Filtering on the graph Laplacian reduces to
  $H'=\sigma(\tilde D^{-1/2}(A+I)\tilde D^{-1/2}HW)$. Each neighbour is averaged with a *fixed,
  isotropic* weight. The layer cannot tell a neighbour to the east from one to the west.
* **Spatial message passing: MPNN / MeshGraphNets** (Gilmer et al. 2017; Pfaff et al. 2021).
  $$ m_{ij}=\phi_e\big(h_i,h_j,e_{ij}\big),\qquad h_i\leftarrow h_i+\phi_v\Big(h_i,\sum_{j\in\mathcal N(i)}m_{ij}\Big) . $$
  The messages see the **edge geometry**, so the layer can build anisotropic, stencil-like
  operators: a learnt analogue of a finite-volume flux or a row of the FEM stiffness matrix.

Both use 12 layers, with the Dirichlet condition built into the output, so information travels
at most 12 edges. The Poisson problem is elliptic: every node influences every other node. On a
20×20 mesh 12 hops cover most of the domain; on a 30×30 mesh they do not. This is a real
limitation, and the finer-mesh test measures it.

<p align="center"><img src="../../docs/lesson13/gnn_poisson.png" width="100%"><br><img src="../../docs/lesson13/gnn_errors.png" width="55%"></p>

<!-- results:13 -->
| model | parameters | median rel. L2 error, test meshes | 90th percentile | median, finer 30x30 meshes | training |
|---|---|---|---|---|---|
| MPNN (spatial, edge vectors), 12 layers | 119553 | 0.198 | 0.329 | 0.783 | 255 s |
| GCN (spectral-derived, isotropic), 12 layers | 14977 | 0.276 | 0.665 | 0.868 | 59 s |
| per-node MLP (no neighbours) | 17281 | 0.531 | 1.312 | 0.667 | 9 s |
| zero prediction | - | 1.000 | - | 1.000 | - |

FEM reference, rel. L2 error on 9² / 17² / 33² / 65² nodes: 4.0e-02 / 8.7e-03 / 2.2e-03 / 5.6e-04 (order 2.14).
<!-- /results:13 -->

**Honest reading.**
* The spatial MPNN, which sees edge vectors, beats the isotropic GCN, and both beat the
  per-node MLP. Neighbour information and geometry matter.
* A median error of about 20 % is far from the FEM solver, which is cheap and exact up to
  $O(h^2)$ for this linear problem. A learnt Poisson solver is not a replacement for a
  sparse direct solve. The value of such models is for nonlinear, expensive physics
  (MeshGraphNets on fluids and cloth), where the same architecture applies.
* **Mesh-refinement failure.** On 30×30 meshes, finer than any seen in training, both GNNs
  degrade badly, even below the neighbour-blind MLP. Two reasons:
  1. 12 message-passing hops no longer span the domain;
  2. the edge vectors are shorter than anything in training. The discrete Laplacian scales like
     $1/h^2$, and the network has only learnt its behaviour at the training $h$.

  Remedies from the literature are multiscale / multigrid GNNs, features normalised by the local
  mesh size, or training on a range of resolutions. This is an important caveat for anyone
  hoping to "train coarse, deploy fine".

## Run it

```bash
python lessons/13_graph_nn_mesh/lesson.py            # script: figures -> docs/lesson13/
jupyter lab notebooks/13_graph_nn_mesh.ipynb       # the same lesson as a notebook
```

---
[← 12_dnc_memory](../12_dnc_memory/README.md) · [course overview](../../README.md) · [14_neural_operators →](../14_neural_operators/README.md)
