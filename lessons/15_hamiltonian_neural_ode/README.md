<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 14_neural_operators](../14_neural_operators/README.md) · [course overview](../../README.md) · [16_rom_cfd →](../16_rom_cfd/README.md)

# Lesson 15 - Physics-structured networks: Hamiltonian neural networks and neural ODEs

📄 [`lessons/15_hamiltonian_neural_ode/lesson.py`](../../lessons/15_hamiltonian_neural_ode/lesson.py) · 📓 [`notebooks/15_hamiltonian_neural_ode.ipynb`](../../notebooks/15_hamiltonian_neural_ode.ipynb) · code: [`nnaz/physics_nets.py`](../../nnaz/physics_nets.py)

### Hamiltonian neural networks

A frictionless mechanical system is described by one scalar, the energy $H(q,p)$, through
Hamilton's equations $\dot q=\partial H/\partial p,\ \dot p=-\partial H/\partial q$. Two ways to
learn the pendulum ($H=p^2/2+1-\cos q$) from 50 noisy trajectories:

* **baseline**: an MLP maps $(q,p)\mapsto(\dot q,\dot p)$. It can represent *any* vector field,
  including ones that slowly gain or lose energy;
* **HNN** (Greydanus et al., 2019): an MLP outputs the scalar $H_\theta(q,p)$. The vector field is
  its **symplectic gradient**, computed by autograd: $(\partial_pH_\theta,-\partial_qH_\theta)$.
  Along this field $\dot H_\theta=\partial_qH_\theta\,\dot q+\partial_pH_\theta\,\dot p=0$
  **identically**. Energy conservation is built in, not learnt.

Both fields are integrated with the same RK4 scheme for 100 time units (about 15 periods) from
unseen initial states.

<p align="center"><img src="../../docs/lesson15/hnn_pendulum.png" width="100%"><br><img src="../../docs/lesson15/pendulums.gif" width="40%"></p>

### Neural ODEs

A **neural ODE** (Chen et al., 2018) models $\dot x=f_\theta(x)$ and is fitted to *trajectory
samples*, with no derivatives needed. We integrate the ODE with a differentiable RK4 solver and
backpropagate through the solver steps. Here the data are **120 irregularly timed**, noisy
observations of the Lotka-Volterra predator-prey system
($\dot u=au-buv,\ \dot v=-cv+duv$). A continuous-time model handles irregular sampling
naturally. We use **multiple shooting**: short windows, each started from an observed state,
which keeps the gradients well-behaved. The model works in log-populations for positivity. The
baseline is the best **linear** ODE $\dot z=Az+b$ in log space, fitted by least squares to finite
differences.

<p align="center"><img src="../../docs/lesson15/neural_ode_lv.png" width="100%"></p>

<!-- results:15 -->
| pendulum, 100 time units, 3 unseen initial states | relative change of the true energy | mean state error vs exact |
|---|---|---|
| baseline MLP | 0.122 / 0.106 / 0.484 | 0.129 / 0.188 / 2.297 |
| HNN | 0.021 / 0.020 / 0.009 | 0.159 / 0.123 / 0.019 |
| exact field + RK4 (integrator error only) | 0.000 / 0.000 / 0.000 | 0 (reference) |

| Lotka-Volterra model | log-RMSE on [0, 30] (data window) | log-RMSE on (30, 60] (extrapolation) | max change of the LV invariant |
|---|---|---|---|
| neural ODE | 0.024 | 0.077 | 0.034 |
| linear ODE in log-space | 0.657 | 1.108 | 0.667 |
<!-- /results:15 -->

**Reading the numbers.**
* The plain MLP's small vector-field errors accumulate. Over 15 periods the true energy of its
  rollouts changes by 11-48 %: the pendulum slowly speeds up or dies out.
* The HNN changes it by only 1-2 %. What it conserves exactly is its *learnt* energy $H_\theta$,
  which differs slightly from the true $H$. The level-set plot shows how close they are.
* Conservation does **not** guarantee phase accuracy. For one start the HNN's state error is
  slightly larger than the baseline's, because a small error in the period accumulates as a
  phase shift. Structure removes a whole class of errors (drift in energy), not all errors.
* The neural ODE, trained only on 120 noisy, irregularly timed samples, tracks the true
  trajectory and extrapolates to twice the training window with a log-error below 0.08. It nearly
  conserves the Lotka-Volterra invariant, even though nothing forces it to (it has no built-in
  structure, unlike the HNN). The best linear ODE cannot represent the nonlinear predator-prey
  coupling at all.

## Run it

```bash
python lessons/15_hamiltonian_neural_ode/lesson.py            # script: figures -> docs/lesson15/
jupyter lab notebooks/15_hamiltonian_neural_ode.ipynb       # the same lesson as a notebook
```

---
[← 14_neural_operators](../14_neural_operators/README.md) · [course overview](../../README.md) · [16_rom_cfd →](../16_rom_cfd/README.md)
