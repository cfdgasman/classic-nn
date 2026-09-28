<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 02_mlp_numpy](../02_mlp_numpy/README.md) · [course overview](../../README.md) · [04_pytorch_basics →](../04_pytorch_basics/README.md)

# Lesson 03 - Training craft

📄 [`lessons/03_training_craft/lesson.py`](../../lessons/03_training_craft/lesson.py) · 📓 [`notebooks/03_training_craft.ipynb`](../../notebooks/03_training_craft.ipynb)

### 1. Initialisation and signal propagation

For $z=\sum_{i=1}^{n}w_i h_i$ with independent zero-mean weights,
$\operatorname{Var}z = n\,\operatorname{Var}w\;\mathbb E[h^2]$. To keep the scale constant from
layer to layer, we need

* **Xavier/LeCun** (tanh, which is roughly linear near 0): $\operatorname{Var}w = 1/n$,
* **He/Kaiming** (ReLU, which zeroes half its inputs, so $\mathbb E[h^2]=\tfrac12\operatorname{Var}z$): $\operatorname{Var}w = 2/n$.

With weights of size 0.01 the activations shrink by about $0.01\sqrt{256}\approx0.16$ per layer
and vanish. The gradients are then about $10^{-7}$, so nothing learns. With ReLU + Xavier the signal
decays slowly; with ReLU + He it stays at order 1.

<p align="center"><img src="../../docs/lesson03/init_signal_propagation.png" width="90%"></p>

### 2. Activations

Since $\sigma'(z)\le 1/4$, every sigmoid layer multiplies the backward signal by at most 1/4.
Six layers therefore attenuate it by up to $4^{-6}\approx 2\times10^{-4}$. tanh ($\tanh'(0)=1$) and
ReLU ($\text{ReLU}'=1$ where active) do not have this problem.

<p align="center"><img src="../../docs/lesson03/activations.png" width="50%"></p>

### 3. Optimisers

| method | update |
|---|---|
| SGD | $\theta\leftarrow\theta-\eta g$ |
| momentum | $v\leftarrow\mu v-\eta g,\;\theta\leftarrow\theta+v$ (a heavy ball: it accumulates speed along consistent directions and damps oscillations across a valley) |
| Adam | $m\leftarrow\beta_1m+(1-\beta_1)g,\;s\leftarrow\beta_2s+(1-\beta_2)g^2,\;\theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat s}+\epsilon)$ with bias corrections $\hat m=m/(1-\beta_1^t)$, $\hat s=s/(1-\beta_2^t)$ |

The Rosenbrock function $f=(1-x)^2+100(y-x^2)^2$ is a curved, narrow valley. Plain GD needs a tiny
step to stay stable across the valley, so it crawls along it. Adam divides each coordinate's step
by its own gradient scale and reaches the minimum.

<p align="center"><img src="../../docs/lesson03/optimisers_rosenbrock.png" width="48%"> <img src="../../docs/lesson03/optimisers_mlp.png" width="46%"></p>

### 4. Learning-rate schedules

Step decay (÷10 at 50 % and 75 % of training), cosine decay
$\eta_t=\tfrac12\eta_0(1+\cos\pi t/T)$, and linear warm-up. On this small problem all schedules
end within a fraction of a percent of each other. That is an honest negative result: schedules
matter most for large models and long training runs.

<p align="center"><img src="../../docs/lesson03/lr_schedules.png" width="90%"></p>

### 5. Batch normalisation

BN re-centres and re-scales each feature using batch statistics. This keeps the pre-activations
in the useful, non-saturated range of the sigmoid, whatever the previous layers do. At test time
it uses running averages. A 10-layer sigmoid network does not train at all without BN; with BN,
it does.

<p align="center"><img src="../../docs/lesson03/batchnorm.png" width="50%"></p>

### 6. Overfitting, regularisation, early stopping

150 training points with **20 % flipped labels** and a network with 67 000 parameters. Without
regularisation the network memorises the noise: training accuracy approaches 100 % while the
validation loss climbs. The remedies:

* **weight decay**, $L+\frac\lambda2\lVert W\rVert^2$, which prefers small weights and therefore
  smooth functions;
* **dropout**, which trains an ensemble of thinned networks;
* **early stopping** at the minimum of the validation loss.

Note that the "training loss" plotted with weight decay includes the penalty term.

<p align="center"><img src="../../docs/lesson03/overfitting.png" width="95%"></p>

**Results:**

<!-- results:03 -->
**Activations** (6 hidden layers, Adam 3e-3, 150 epochs)

| activation | final train loss | train accuracy |
|---|---|---|
| sigmoid | 5.34e-01 | 79.8% |
| tanh | 1.75e-04 | 100.0% |
| relu | 1.05e-05 | 100.0% |

**Optimisers**

| run | final value |
|---|---|
| Rosenbrock, GD  (lr 1e-3) | 2.5e-02 |
| Rosenbrock, momentum 0.9 (lr 1e-4) | 2.4e-02 |
| Rosenbrock, Adam (lr 2e-2) | 2.8e-07 |
| spiral MLP loss, SGD lr 0.1 | 0.0822 |
| spiral MLP loss, momentum lr 0.01 | 0.0842 |
| spiral MLP loss, momentum lr 0.1 | 0.0016 |
| spiral MLP loss, Adam lr 3e-3 | 0.0041 |

**LR schedules** (SGD + momentum, 100 epochs, mean of 3 seeds)

| schedule | final train loss | test accuracy |
|---|---|---|
| constant | 0.0025 | 99.7% |
| step | 0.0062 | 99.6% |
| cosine | 0.0070 | 99.6% |
| cosine + warm-up | 0.0065 | 99.6% |

**Batch norm** (10 sigmoid hidden layers, SGD + momentum)

| network | final train loss | train accuracy |
|---|---|---|
| plain | 1.104 | 33.3% |
| with BatchNorm | 0.246 | 86.3% |

**Overfitting** (150 points, 20 % label noise, 2-256-256-3, 400 epochs)

| regularisation | train acc | val acc (final) | val loss (final) | early stopping: epoch / val acc |
|---|---|---|---|---|
| no regularisation | 99.3% | 76.0% | 1.165 | 58 / 83.5% |
| weight decay 1e-3 | 94.0% | 79.7% | 0.542 | 122 / 83.2% |
| dropout 0.3 | 95.3% | 79.5% | 0.605 | 197 / 85.9% |
| dropout 0.3 + wd 1e-3 | 91.3% | 81.9% | 0.452 | 199 / 89.2% |
<!-- /results:03 -->

## Run it

```bash
python lessons/03_training_craft/lesson.py            # script: figures -> docs/lesson03/
jupyter lab notebooks/03_training_craft.ipynb       # the same lesson as a notebook
```

---
[← 02_mlp_numpy](../02_mlp_numpy/README.md) · [course overview](../../README.md) · [04_pytorch_basics →](../04_pytorch_basics/README.md)
