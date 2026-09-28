<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 08_attention_gpt](../08_attention_gpt/README.md) · [course overview](../../README.md) · [10_generative →](../10_generative/README.md)

# Lesson 09 - Scientific ML: a PINN for Burgers and a neural surrogate of a solver

📄 [`lessons/09_scientific_ml/lesson.py`](../../lessons/09_scientific_ml/lesson.py) · 📓 [`notebooks/09_scientific_ml.ipynb`](../../notebooks/09_scientific_ml.ipynb) · code: [`nnaz/burgers.py`](../../nnaz/burgers.py), [`nnaz/pinn.py`](../../nnaz/pinn.py)

### The benchmark problem

$$ u_t+u\,u_x=\nu\,u_{xx},\qquad x\in[-1,1],\ t\in[0,1],\qquad u(x,0)=-\sin\pi x,\quad u(\pm1,t)=0,\quad \nu=\tfrac{0.01}{\pi}. $$

The two halves of the sine wave run into each other at $x=0$ and form a viscous shock whose
thickness is only about $\nu/|u|\approx0.003$ by $t\approx0.4$. This is the standard PINN
benchmark (Raissi, Perdikaris and Karniadakis 2019).

### Exact solution (Cole-Hopf) - our ground truth

The **Cole-Hopf** transform $u=-2\nu\,\phi_x/\phi$ turns Burgers into the heat equation
$\phi_t=\nu\phi_{xx}$. For this initial condition the solution is

$$
u(x,t)=-\frac{\displaystyle\int\sin\pi(x-\eta)\,f(x-\eta)\,e^{-\eta^2/4\nu t}\,d\eta}{\displaystyle\int f(x-\eta)\,e^{-\eta^2/4\nu t}\,d\eta},
\qquad f(y)=\exp\!\Big(-\frac{\cos\pi y}{2\pi\nu}\Big).
$$

We substitute $\eta=\sqrt{4\nu t}\,z$ and evaluate both integrals with **Gauss-Hermite
quadrature** (weight $e^{-z^2}$). $f$ reaches $e^{50}$, so we work with $\log f$ and subtract its
maximum before exponentiating. Using 120 or 200 nodes changes the answer by less than $10^{-15}$.

### A classical solver, validated

Second-order central finite differences in conservative form,
$\partial_t u_j=-\frac{F_{j+1}-F_{j-1}}{2\Delta x}+\nu\frac{u_{j+1}-2u_j+u_{j-1}}{\Delta x^2}$
with $F=u^2/2$, and classical RK4 in time. The Dirichlet problem equals the periodic one here,
because $-\sin\pi x$ is odd and 2-periodic. Central differencing of convection is
non-oscillatory only when the **cell Reynolds number** $|u|\Delta x/\nu<2$, which is the classic
CFD constraint. The time step respects both the diffusive limit ($\propto\Delta x^2/\nu$) and the
convective CFL limit. The convergence table below shows the cell-Reynolds constraint at work:
at N = 128 and 256 the cell Reynolds number is 4.9 and 2.5, the scheme wiggles, and the error
barely improves. From N = 512 on it converges at the design order of 2.

### The PINN

A network $u_\theta(x,t)$ is trained to make the **PDE residual** vanish at 7 500
collocation points. No solution data is used:

$$
\mathcal L(\theta)=\frac1{N_c}\sum_{i}\big(\partial_tu_\theta+u_\theta\,\partial_xu_\theta-\nu\,\partial_{xx}u_\theta\big)^2\Big|_{(x_i,t_i)} .
$$

All derivatives are **exact** derivatives of the network, computed by autograd
(`create_graph=True` twice for $u_{xx}$). Design choices, each a common PINN "trick":

* **Hard constraints**: $u_\theta=-\sin\pi x+t(1-x^2)N_\theta(x,t)$ satisfies the initial and
  boundary conditions for *any* $\theta$. There are no competing loss terms to weight.
* **float64**: second derivatives of a deep tanh network are sensitive to round-off.
* **Adam, then L-BFGS**: Adam is robust far from a solution (it stalls at about 10-20 % error
  here). L-BFGS, a quasi-Newton method, converges much faster near one; the error drops by two
  orders of magnitude.
* extra collocation points near $x=0$, where the steep front forms.

<p align="center"><img src="../../docs/lesson09/pinn_fields.png" width="95%"><br><img src="../../docs/lesson09/pinn_slices.png" width="95%"><br>
<img src="../../docs/lesson09/pinn_training.png" width="45%"> <img src="../../docs/lesson09/pinn_burgers.gif" width="45%"></p>

### A neural surrogate of the solver

A **surrogate** replaces an expensive simulation with a cheap learnt map, here
$(\nu, A)\mapsto u(x, T=0.5)$ for $u(x,0)=-A\sin\pi x$, with $\nu\in[0.005,0.05]$ (log-uniform)
and $A\in[0.5,1.5]$. We run the FD solver 400 times (vectorised: all parameters are advanced
together) and fit an MLP that outputs the 128-point solution. It is tested on 200 new
parameter pairs.

<p align="center"><img src="../../docs/lesson09/surrogate.png" width="95%"></p>

<!-- results:09 -->
**Reference solver**

| FD grid N | rel. L2 error at t = 1 vs Cole-Hopf | CPU time |
|---|---|---|
| 128 | 5.2e-03 | 0.03 s |
| 256 | 4.8e-03 | 0.06 s |
| 512 | 1.3e-03 | 0.13 s |
| 1024 | 3.2e-04 | 0.46 s |
| 2048 | 8.1e-05 | 2.24 s |

Observed order of accuracy: **1.97** (second-order scheme).

**PINN** (ν = 0.01/π)

| quantity | value |
|---|---|
| PINN rel. L2 error, whole space-time grid | **1.3e-03** |
| PINN max abs error | 6.7e-03 |
| PINN rel. L2 error at t = 0.25 / 0.5 / 0.75 / 1 | 9.9e-04 / 1.2e-03 / 1.6e-03 / 2.9e-03 |
| PINN parameters / training time (CPU) | 5409 / 15.8 min |
| Raissi et al. (2019), 9x20 net, 10 000 points (published) | 6.7e-04 |

**Surrogate**

| model (test set, rel. L2 error) | median | 95th percentile |
|---|---|---|
| mean solution (trivial baseline) | 2.1e-01 | - |
| nearest training run | 1.3e-02 | 4.9e-02 |
| piecewise-linear interpolation (Delaunay in parameter space) | 7.2e-04 | 2.2e-03 |
| neural surrogate (MLP 2 -> 128 grid values) | 1.5e-03 | 4.5e-03 |

Solver: 111 ms per solution (vectorised over parameters); its own error at the parameter-space corners is 1.6e-03. Surrogate: 4.2 µs per solution after 19 s of training on 400 solver runs.
<!-- /results:09 -->

**Honest reading.**
* The PINN reaches about $10^{-3}$ relative error without seeing any solution data, the same
  order as the published result. It costs **minutes** of CPU time. The finite-difference solver
  reaches $3\times10^{-4}$ in **half a second** (N = 1024). For a single forward problem like this
  one, a classical solver wins by orders of magnitude. PINNs earn their keep for inverse problems
  (unknown $\nu$ or unknown sources from sparse data), for irregular data assimilation, or when a
  mesh is inconvenient.
* On a 2-parameter problem with 400 runs, **piecewise-linear interpolation beats the neural
  surrogate**, and both are below the solver's own discretisation error (about $10^{-3}$).
  Neural surrogates pay off when the parameter space has many dimensions (interpolation tables
  scale exponentially) or when the output is a large field. The lesson that transfers: always
  benchmark a surrogate against simple interpolation and nearest-neighbour baselines.
* A surrogate evaluation costs microseconds against about 100 ms for a solve. That is the
  real value in design loops, uncertainty quantification and optimisation.

## Run it

```bash
python lessons/09_scientific_ml/lesson.py            # script: figures -> docs/lesson09/
jupyter lab notebooks/09_scientific_ml.ipynb       # the same lesson as a notebook
```

---
[← 08_attention_gpt](../08_attention_gpt/README.md) · [course overview](../../README.md) · [10_generative →](../10_generative/README.md)
