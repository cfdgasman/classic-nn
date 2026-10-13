<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 15_hamiltonian_neural_ode](../15_hamiltonian_neural_ode/README.md) · [course overview](../../README.md)

# Lesson 16 - Reduced-order modelling for CFD: POD vs a convolutional autoencoder

📄 [`lessons/16_rom_cfd/lesson.py`](../../lessons/16_rom_cfd/lesson.py) · 📓 [`notebooks/16_rom_cfd.ipynb`](../../notebooks/16_rom_cfd.ipynb) · code: [`nnaz/lbm.py`](../../nnaz/lbm.py), [`nnaz/rom.py`](../../nnaz/rom.py)

### The CFD data: our own lattice-Boltzmann solver

The **lattice Boltzmann method** evolves particle distributions $f_i$ along the 9 velocities of a
D2Q9 lattice, alternating a **collision** (BGK relaxation to a local equilibrium) and a
**streaming** step:

$$
f_i^*=f_i-\frac{1}{\tau}\big(f_i-f_i^{\rm eq}(\rho,\mathbf u)\big),\qquad f_i(\mathbf x+\mathbf c_i,t+1)=f_i^*(\mathbf x,t),\qquad
f_i^{\rm eq}=w_i\rho\Big[1+3\,\mathbf c_i\!\cdot\!\mathbf u+\tfrac92(\mathbf c_i\!\cdot\!\mathbf u)^2-\tfrac32|\mathbf u|^2\Big].
$$

A Chapman-Enskog expansion shows that this recovers the incompressible Navier-Stokes equations
with $\nu=(\tau-\frac12)/3$. We simulate flow past a cylinder ($D=16$ cells, $U=0.1$, Re = 100,
400×100 grid) with bounce-back on the cylinder (no-slip), a velocity inlet, a zero-gradient
outlet and periodic side boundaries. **Validation:** the Strouhal number $St=fD/U$ of the vortex
shedding, measured by FFT at a wake probe, is compared with Williamson's (1988) fit for
unconfined flow, $St=0.2175-5.1064/\mathrm{Re}$.

<p align="center"><img src="../../docs/lesson16/cylinder_wake.gif" width="80%"></p>

### Compression: POD vs a convolutional autoencoder

**POD** (proper orthogonal decomposition) is PCA of the snapshot matrix. It is the workhorse of
CFD model reduction. The SVD of the mean-subtracted snapshots gives orthonormal spatial modes
ranked by energy, and truncating to $r$ modes is the optimal *linear* compression (lesson 07). A
periodic wake is a **limit cycle**: its state is essentially one phase angle. POD needs a
**pair** of modes (sine and cosine) for every harmonic of the shedding frequency, as the paired
spectrum below shows. A *nonlinear* autoencoder can map the cycle onto a closed curve in a 2-D
latent space. We train on the first 75 % of the snapshots and test on the last 25 %.

<p align="center"><img src="../../docs/lesson16/pod_modes.png" width="90%"><br><img src="../../docs/lesson16/rom_compression.png" width="100%"><br><img src="../../docs/lesson16/rom_reconstruction.png" width="100%"></p>

### Forecasting in the reduced space

* **DMD** (dynamic mode decomposition): the best *linear* propagator $a_{n+1}=a_nM$ of the POD
  coefficients, fitted by least squares. Its eigenvalues give the shedding frequency and growth
  rates.
* **Neural latent map**: $z_{n+1}=z_n+g_\theta(z_n)$ in the autoencoder's 2-D latent space,
  trained on 5-step rollouts, then decoded.

<!-- results:16 -->
| LBM validation | value |
|---|---|
| Strouhal number St = f D / U (FFT of a wake probe) | 0.1885 |
| Williamson (1988), unconfined cylinder, Re = 100 | 0.1664 |
| difference | +13.2 % |
| lateral spacing / blockage D/H | 0.16 (periodic sides) |
| LBM CPU time | 158 s |

**Compression** (relative error w.r.t. the fluctuation, unseen snapshots)

| latent dimension | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| POD (linear), test error | 0.756 | 0.411 | 0.295 | 0.077 | 0.010 |
| conv autoencoder, test error | 0.133 | 0.067 | 0.040 | - | - |

**Forecasting**

| 100-snapshot forecast (last 25 % of the run) | relative error |
|---|---|
| DMD on 2 POD modes | 0.412 |
| DMD on 8 POD modes | 0.082 |
| autoencoder (latent 2) + neural latent map | 0.324 |
| autoencoder reconstruction (no forecasting) | 0.067 |
| mean flow (trivial baseline) | 1.000 |
<!-- /results:16 -->

**Reading the numbers.**
* **Validation.** The Strouhal number is within about 13 % of Williamson's value for an
  *unconfined* cylinder. The difference is physical, not a bug. Our side boundaries are periodic,
  so we actually simulate an infinite row of cylinders 6.25 D apart. Neighbouring wakes interact,
  and confinement is known to raise the shedding frequency. A wider domain (more cells) would
  close the gap at the price of CPU time.
* **Compression.** A single autoencoder latent variable beats 4 POD modes, and a 2-D latent
  matches about 8 POD modes. This is the limit-cycle argument in numbers: the flow lives on a
  closed curve, and POD has to spend two modes per harmonic to describe it linearly. The 2-D code
  indeed traces a closed loop (right panel above).
* **Forecasting.** For this *periodic* flow, the linear DMD model with 8 POD modes is the best
  forecaster: a limit cycle is well described by a few pure oscillations, which is exactly DMD's
  model. The neural latent map is accurate for a few periods but accumulates a **phase drift**,
  so its error over 100 snapshots is larger. Its floor is the autoencoder's reconstruction
  error (0.067). This honest result generalises: nonlinear ROMs shine for compression. For
  long-time forecasting they need extra structure (phase/frequency conditioning, or training on
  long rollouts), especially against DMD on problems that are nearly periodic.

## Run it

```bash
python lessons/16_rom_cfd/lesson.py            # script: figures -> docs/lesson16/
jupyter lab notebooks/16_rom_cfd.ipynb       # the same lesson as a notebook
```

---
[← 15_hamiltonian_neural_ode](../15_hamiltonian_neural_ode/README.md) · [course overview](../../README.md)
