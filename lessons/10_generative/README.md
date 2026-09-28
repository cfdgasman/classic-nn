<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 09_scientific_ml](../09_scientific_ml/README.md) · [course overview](../../README.md) · [11_som_hopfield →](../11_som_hopfield/README.md)

# Lesson 10 - Generative models: a VAE on MNIST and a minimal diffusion model

📄 [`lessons/10_generative/lesson.py`](../../lessons/10_generative/lesson.py) · 📓 [`notebooks/10_generative.ipynb`](../../notebooks/10_generative.ipynb) · code: [`nnaz/generative.py`](../../nnaz/generative.py)

### Variational autoencoder

A latent-variable model: draw $z\sim\mathcal N(0,I)$, then $x\sim p_\theta(x\mid z)$ (the
*decoder*, here independent Bernoulli pixels). The likelihood
$p(x)=\int p(x\mid z)p(z)\,dz$ is intractable, so we introduce an *encoder*
$q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}\sigma^2_\phi(x))$ and maximise the
**evidence lower bound**:

$$
\log p(x)\;\ge\;\underbrace{\mathbb E_{q}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}}-\underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,\mathcal N(0,I)\big)}_{\frac12\sum(\mu^2+\sigma^2-\log\sigma^2-1)} .
$$

* **Reparameterisation trick**: $z=\mu+\sigma\odot\varepsilon$ with $\varepsilon\sim\mathcal N(0,I)$
  makes the sample a differentiable function of $(\mu,\sigma)$, so the ELBO can be
  back-propagated.
* The KL term pulls every code towards the prior. This is what makes the latent space
  *continuous* and lets us **sample**: decode $z\sim\mathcal N(0,I)$.
* The data are binarised (pixel > 127), so the numbers are true log-likelihoods in nats. The
  trivial baseline is independent Bernoulli pixels (an exact likelihood). The
  **importance-weighted bound** (IWAE, $K=100$) tightens the ELBO, which is how VAE likelihoods are
  reported in papers.

<p align="center"><img src="../../docs/lesson10/vae_latent.png" width="90%"><br><img src="../../docs/lesson10/vae_samples.png" width="60%"></p>

### Denoising diffusion (DDPM)

**Forward process** (fixed, no learning): add a little Gaussian noise at each of $T=100$ steps.
In closed form,

$$ x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \bar\alpha_t=\prod_{s\le t}(1-\beta_s), $$

with $\beta$ linear from $10^{-4}$ to $0.1$, so $x_T$ is essentially pure noise.
**Training**: a network $\varepsilon_\theta(x_t,t)$, an MLP with a sinusoidal time embedding,
learns to predict the added noise by minimising $\lVert\varepsilon_\theta(x_t,t)-\varepsilon\rVert^2$.
**Sampling** runs the chain backwards from pure noise:

$$ x_{t-1}=\frac{1}{\sqrt{1-\beta_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\Big)+\sqrt{\beta_t}\,z . $$

**What the network learns, checked exactly.** The optimal noise predictor is the **score** of
the noised data distribution, $\varepsilon^*(x,t)=-\sqrt{1-\bar\alpha_t}\,\nabla_x\log q_t(x)$.
Our data are a mixture of 8 Gaussians, so $q_t$ is again a Gaussian mixture, with means
$\sqrt{\bar\alpha_t}\mu_k$ and variance $\bar\alpha_t\sigma^2+1-\bar\alpha_t$. Its score is known
in closed form, which lets us:
1. measure the error of the learnt $\varepsilon_\theta$ against $\varepsilon^*$ at every $t$;
2. run the sampler with the **exact** score, separating *sampler* error from *learning* error.

<p align="center"><img src="../../docs/lesson10/diffusion_forward.png" width="95%"><br>
<img src="../../docs/lesson10/diffusion_samples.png" width="95%"><br>
<img src="../../docs/lesson10/diffusion_score_field.png" width="60%"> <img src="../../docs/lesson10/diffusion_reverse.gif" width="32%"></p>

<!-- results:10 -->
| model (binarised MNIST, test set) | −ELBO (nats) | reconstruction + KL | IWAE NLL, K = 100 (nats) |
|---|---|---|---|
| independent Bernoulli pixels (exact NLL, trivial baseline) | - | - | 200.1 |
| VAE, latent 2 | 137.2 | 130.3 + 6.9 | 132.4 |
| VAE, latent 16 | 83.9 | 61.1 + 22.9 | 76.8 |

| sampler (4000 samples) | MMD² vs data | within 3σ of a mode | modes covered |
|---|---|---|---|
| fresh data (metric noise floor) | -2.0e-04 | 98.8% | 8/8 |
| DDPM with exact score | -9.1e-05 | 98.1% | 8/8 |
| DDPM, learnt network | 1.0e-03 | 93.9% | 8/8 |
| Gaussian fit (baseline) | 1.6e-01 | 6.0% | 8/8 |

Relative RMS error of the learnt noise predictor vs the exact $\epsilon^*$: t=0: 2.241, t=10: 0.165, t=20: 0.083, t=30: 0.074, t=40: 0.040, t=50: 0.030, t=60: 0.023, t=70: 0.027, t=80: 0.019, t=90: 0.021, t=99: 0.040.
<!-- /results:10 -->

**Reading the numbers.** The VAE with a 16-D latent reaches about 77 nats against 200 for the
independent-pixel model. The 2-D VAE is worse but gives the interpretable manifold shown above.
The diffusion sampler with the exact score is indistinguishable from the data: its MMD² is at the
metric's noise floor, and negative values are possible for the unbiased estimator. The learnt
model covers all 8 modes but puts about 6 % of its samples between them. The largest score
errors are at the smallest noise levels ($t=0$). There the exact score is extremely steep
(noise std 0.01 against a mode width of 0.1) and hard to fit, but those steps barely move the
samples. The single Gaussian baseline, of course, fails completely.

## Run it

```bash
python lessons/10_generative/lesson.py            # script: figures -> docs/lesson10/
jupyter lab notebooks/10_generative.ipynb       # the same lesson as a notebook
```

---
[← 09_scientific_ml](../09_scientific_ml/README.md) · [course overview](../../README.md) · [11_som_hopfield →](../11_som_hopfield/README.md)
