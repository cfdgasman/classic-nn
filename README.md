# Neural networks from A to Z

[![CI](https://github.com/cfdgasman/classic-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/cfdgasman/classic-nn/actions/workflows/ci.yml)

A step-by-step, hands-on course that takes you from **a single artificial neuron** to
**transformers** and **scientific machine learning** (physics-informed networks, neural
surrogates of PDE solvers, VAEs and diffusion models). Every lesson is a numbered folder with

* a runnable, heavily commented **script** (`lessons/NN_topic/lesson.py`) that explains the idea,
  derives the maths and implements it,
* a **Jupyter notebook** (`notebooks/NN_topic.ipynb`) that walks through the same lesson cell by
  cell, with the figures inline,
* **figures and GIFs** saved to `docs/lessonNN/` and the numbers of the run saved to
  `docs/results/lessonNN.json` (the tables below are generated from these files by `run.py`).

Everything runs on a laptop **CPU in minutes**. The first three lessons use only NumPy, so every
gradient is derived and coded by hand before we let PyTorch's autograd do it for us.

<p align="center">
  <img src="docs/lesson02/spiral_training.gif" width="30%" alt="MLP learning the spiral">
  <img src="docs/lesson01/perceptron.gif" width="30%" alt="perceptron updates">
  <img src="docs/lesson03/optimisers.gif" width="36%" alt="optimisers on Rosenbrock">
</p>

---

## Contents

1. [Learning path](#learning-path) - which lesson answers which question
2. [Quick start](#quick-start)
3. [Notation and the three ideas everything rests on](#notation-and-the-three-ideas-everything-rests-on)
4. [Lesson 01 - The single neuron](#lesson-01--the-single-neuron-perceptron-and-logistic-regression)
5. [Lesson 02 - An MLP from scratch](#lesson-02--a-multilayer-perceptron-from-scratch-in-numpy)
6. [Lesson 03 - Training craft](#lesson-03--training-craft)
7. [Lessons 04-10](#lessons-04-10)
8. [Lessons learned / bugs found](#lessons-learned--bugs-found)
9. [Repository layout and tests](#repository-layout-and-tests)
10. [References](#references)

## Learning path

| # | Lesson | The question it answers | Key result (real run) | Status |
|---|---|---|---|---|
| 01 | [Perceptron / single neuron](lessons/01_perceptron) | What does one neuron compute, how does it learn, and how do I derive and *check* a gradient by hand? | hand gradient = finite differences to 1e-9; GD reaches the exact L-BFGS optimum; fails on XOR | ✅ |
| 02 | [MLP from scratch (NumPy)](lessons/02_mlp_numpy) | What *is* backpropagation, and how do I prove my implementation is right? | gradient check < 1e-6 for every layer type; spirals 33 % → 100 % | ✅ |
| 03 | [Training craft](lessons/03_training_craft) | Why does a deep net not train, and what do initialisation, ReLU, momentum/Adam, schedules, batch norm, dropout fix? | BN rescues a 10-layer sigmoid net; regularisation +6 % val. accuracy under label noise | ✅ |
| 04 | PyTorch basics | What does autograd do, and how do I write a clean training loop? | reproduces lesson 02 to machine precision | 🚧 |
| 05 | CNNs on MNIST / Fashion-MNIST | Why convolutions for images, and what do the filters learn? | target > 98 % on MNIST | 🚧 |
| 06 | RNN / LSTM forecasting | Can a recurrent net forecast a nonlinear oscillator better than a linear model? | vs. persistence and linear AR | 🚧 |
| 07 | Autoencoders | What does a bottleneck learn; is a linear autoencoder just PCA? | linear AE = PCA subspace | 🚧 |
| 08 | Attention → tiny GPT | How does self-attention work, and how does a GPT generate text? | char-level GPT on Shakespeare | 🚧 |
| 09 | Scientific ML: PINN + surrogate | Can a network solve a PDE (Burgers) without data, and emulate a solver? | vs. exact Cole-Hopf solution | 🚧 |
| 10 | Generative models: VAE + diffusion | How do we *sample* new data? | VAE on MNIST, DDPM on 2-D data | 🚧 |

## Quick start

```bash
git clone https://github.com/cfdgasman/classic-nn && cd classic-nn
python -m venv .venv && source .venv/bin/activate          # Python 3.12+
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python lessons/01_perceptron/lesson.py      # one lesson (figures -> docs/lesson01/)
python run.py                               # every lesson + README tables + notebooks
python run.py --lessons 1 2 --quick         # a fast subset
jupyter lab notebooks/                      # the same lessons as notebooks
pytest                                      # the test-suite (< 2 min)
```

## Notation and the three ideas everything rests on

A data set is a matrix $X\in\mathbb R^{N\times D}$ (one sample per **row**) with targets $y$.
A network is a parametrised function $f_\theta$; training minimises an average loss

$$
L(\theta)=\frac1N\sum_{n=1}^N \ell\big(f_\theta(x_n),\,y_n\big).
$$

Three ideas carry the entire course:

1. **Gradient descent.** $\theta \leftarrow \theta-\eta\,\nabla_\theta L$. With mini-batches the
   gradient is estimated on $B\ll N$ samples (stochastic gradient descent).
2. **The chain rule, organised as backpropagation.** A network is a composition
   $f = f_K\circ\dots\circ f_1$. If each layer knows how to turn $\partial L/\partial(\text{output})$
   into $\partial L/\partial(\text{input})$ and $\partial L/\partial(\text{its parameters})$, one
   backward sweep gives every gradient at about the cost of one forward pass.
3. **Validation against something you trust.** Every gradient is checked with finite differences,
   every optimiser against a known optimum, every model against a trivial baseline, and every PDE
   solution against an exact one.

---

## Lesson 01 - The single neuron: perceptron and logistic regression

📄 [`lessons/01_perceptron/lesson.py`](lessons/01_perceptron/lesson.py) · 📓 [`notebooks/01_perceptron.ipynb`](notebooks/01_perceptron.ipynb) · code: [`nnaz/neuron.py`](nnaz/neuron.py)

**What a neuron computes.** A weighted sum followed by a non-linearity:

$$ z = w\cdot x + b, \qquad a = \varphi(z). $$

The set $\{x : w\cdot x+b=0\}$ is a straight line (a hyperplane in $D$ dimensions): **a single
neuron can only draw one linear decision boundary**.

### The perceptron (Rosenblatt, 1958)

With $\varphi=\operatorname{sign}$ and labels $y\in\{-1,+1\}$, the learning rule is: visit the
samples, and whenever one is misclassified ($y\,(w\cdot x+b)\le 0$) move the boundary towards it,

$$ w \leftarrow w + y\,x, \qquad b \leftarrow b + y. $$

**Why it converges (Novikoff, 1962).** Append a constant 1 to every input so the bias is just a
weight. Suppose some unit vector $u$ separates the data with margin
$\gamma=\min_n y_n\,u\cdot x_n>0$, and $R=\max_n\lVert x_n\rVert$. Each mistake increases
$w\cdot u$ by at least $\gamma$, while $\lVert w\rVert^2$ grows by at most $R^2$. After $k$
mistakes, $k\gamma \le w\cdot u\le\lVert w\rVert\le\sqrt{k}R$, so

$$ k \le (R/\gamma)^2 . $$

The script generates separable data from a known line, so it can compute this bound and compare it
with the actual number of mistakes.

<p align="center"><img src="docs/lesson01/perceptron.gif" width="38%"> <img src="docs/lesson01/xor_failure.png" width="36%"></p>

### The logistic neuron and its gradient, derived by hand

With the sigmoid $\sigma(z)=1/(1+e^{-z})$ the output is a probability, and we minimise the
binary cross-entropy (plus an L2 penalty $\tfrac\lambda2\lVert w\rVert^2$):

$$ \ell = -\big[y\log a + (1-y)\log(1-a)\big], \qquad a=\sigma(z). $$

Chain rule, one factor at a time:

$$
\frac{\partial\ell}{\partial a}=\frac{a-y}{a(1-a)},\qquad
\frac{\partial a}{\partial z}=\sigma'(z)=a(1-a)
\;\;\Longrightarrow\;\;
\boxed{\frac{\partial\ell}{\partial z}=a-y},\qquad
\frac{\partial\ell}{\partial w}=(a-y)\,x,\quad \frac{\partial\ell}{\partial b}=a-y .
$$

Over a batch this is one matrix product, $\nabla_w L = X^\top(a-y)/N+\lambda w$. Two numerical
details from the code:

* the loss is evaluated as $\log(1+e^{z})-yz$ with `np.logaddexp`, which never overflows;
* the gradient is **checked** with central differences
  $\big(L(\theta+\epsilon e_k)-L(\theta-\epsilon e_k)\big)/2\epsilon$, which have error $O(\epsilon^2)$.

Because the regularised problem is strictly convex, it has one minimiser. We compute that
minimiser independently with SciPy's L-BFGS and check that plain gradient descent converges to it.

<p align="center"><img src="docs/lesson01/logistic_neuron.png" width="80%"></p>

**Results** (from `docs/results/lesson01.json`):

<!-- results:01 -->
| quantity | value |
|---|---|
| perceptron mistakes before convergence | 14 |
| Novikoff bound (R/γ)² with R, γ | 1813  (R = 4.08, γ = 0.096) |
| hand gradient vs finite differences (rel. error) | 1.3e-09 |
| GD vs L-BFGS optimum: max parameter difference | 1.2e-08 |
| logistic neuron test accuracy | 92.6% |
| majority-class baseline | 50.2% |
| single neuron on XOR (train accuracy) | 36.8% |
<!-- /results:01 -->

**Take-aways.** The perceptron made far fewer mistakes than the worst-case bound. Gradient descent
converges *linearly*: a straight line on the log plot, as theory predicts for a strongly convex
problem. On XOR, a single neuron is no better than chance; the accuracy is even below 50 %,
because it minimises cross-entropy, not the error count. Lesson 02 fixes this with a hidden layer.

---

## Lesson 02 - A multilayer perceptron from scratch in NumPy

📄 [`lessons/02_mlp_numpy/lesson.py`](lessons/02_mlp_numpy/lesson.py) · 📓 [`notebooks/02_mlp_numpy.ipynb`](notebooks/02_mlp_numpy.ipynb) · code: [`nnaz/mlp_numpy.py`](nnaz/mlp_numpy.py)

### Forward pass

An MLP alternates affine maps and element-wise non-linearities. For a mini-batch $X$ (rows = samples):

$$ H_0=X,\qquad Z_k = H_{k-1}W_k + b_k,\qquad H_k=\varphi(Z_k),\qquad \text{logits}=Z_K . $$

For $C$ classes, the loss is the softmax cross-entropy
$L=-\frac1N\sum_n\log\operatorname{softmax}(z_n)_{y_n}$.
Its gradient is again "prediction minus target":

$$ \frac{\partial L}{\partial Z_K} = \frac1N\big(\operatorname{softmax}(Z_K)-\operatorname{onehot}(y)\big). $$

### Backward pass: every layer is a small chain-rule machine

Each layer caches what it needs during `forward` and implements `backward(dY) -> dX`:

| layer | forward | backward (given $dY=\partial L/\partial Y$) |
|---|---|---|
| Linear | $Y = XW+b$ | $dW=X^\top dY$, $\;db=\sum_n dY_n$, $\;dX = dY\,W^\top$ |
| ReLU | $Y=\max(0,X)$ | $dX = dY\odot\mathbb 1[X>0]$ |
| tanh | $Y=\tanh X$ | $dX = dY\odot(1-Y^2)$ |
| sigmoid | $Y=\sigma(X)$ | $dX = dY\odot Y(1-Y)$ |
| BatchNorm | $\hat X=(X-\mu)/\sqrt{\sigma^2+\varepsilon}$, $Y=\gamma\hat X+\beta$ | $dX=\frac{1}{N\sigma}\big(N\,d\hat X-\sum d\hat X-\hat X\sum d\hat X\odot\hat X\big)$ |
| Dropout | $Y = X\odot M/(1-p)$ | $dX = dY\odot M/(1-p)$ |

Backprop is then just `for layer in reversed(layers): dY = layer.backward(dY)`. Note how
$dW=X^\top dY$ automatically **sums the per-sample gradients**, so there is no loop over the batch.

### Proving it is right: the gradient check

For every parameter entry $\theta_k$ we compare backprop with the central difference and report
the norm-wise relative error $\lVert g_{bp}-g_{fd}\rVert/(\lVert g_{bp}\rVert+\lVert g_{fd}\rVert)$
per tensor. In float64 the truncation error is $O(\epsilon^2)$ and the round-off error is
$O(10^{-16}/\epsilon)$, so the sweet spot is $\epsilon\approx10^{-5}$ to $10^{-6}$. A correct
implementation gives errors of $10^{-8}$ to $10^{-10}$, far below the required $10^{-6}$.

<p align="center"><img src="docs/lesson02/gradcheck_eps.png" width="55%"></p>

The V-shape is the textbook picture: round-off error on the left, truncation error on the right.
The ReLU curve jumps once $\epsilon$ is large enough for a perturbation to cross a kink
($z=0$), where the finite difference no longer measures the derivative.

<!-- results:02 -->
| network 2-8-8-3 | max rel. error, parameters | max rel. error, inputs | < 1e-6 ? |
|---|---|---|---|
| tanh | 3.4e-08 | 1.3e-09 | PASS |
| relu | 1.4e-09 | 5.7e-10 | PASS |
| sigmoid | 1.9e-08 | 2.2e-08 | PASS |
| tanh + batchnorm + dropout | 2.9e-09 | 8.6e-10 | PASS |

| model (spirals, 3 classes) | test accuracy |
|---|---|
| majority class | 33.3% |
| softmax regression (no hidden layer) | 33.0% |
| MLP 2-64-64-3, ReLU, Adam | 100.0% |
| XOR with 4 hidden tanh units (train) | 93.2% |
<!-- /results:02 -->

<p align="center"><img src="docs/lesson02/spiral_mlp.png" width="95%"></p>

---

## Lesson 03 - Training craft

📄 [`lessons/03_training_craft/lesson.py`](lessons/03_training_craft/lesson.py) · 📓 [`notebooks/03_training_craft.ipynb`](notebooks/03_training_craft.ipynb)

### 1. Initialisation and signal propagation

For $z=\sum_{i=1}^{n}w_i h_i$ with independent zero-mean weights,
$\operatorname{Var}z = n\,\operatorname{Var}w\;\mathbb E[h^2]$. To keep the scale constant from
layer to layer, we need

* **Xavier/LeCun** (tanh, which is roughly linear near 0): $\operatorname{Var}w = 1/n$,
* **He/Kaiming** (ReLU, which zeroes half its inputs, so $\mathbb E[h^2]=\tfrac12\operatorname{Var}z$): $\operatorname{Var}w = 2/n$.

With weights of size 0.01 the activations shrink by about $0.01\sqrt{256}\approx0.16$ per layer
and vanish. The gradients are then about $10^{-7}$, so nothing learns. With ReLU + Xavier the signal
decays slowly; with ReLU + He it stays at order 1.

<p align="center"><img src="docs/lesson03/init_signal_propagation.png" width="90%"></p>

### 2. Activations

Since $\sigma'(z)\le 1/4$, every sigmoid layer multiplies the backward signal by at most 1/4.
Six layers therefore attenuate it by up to $4^{-6}\approx 2\times10^{-4}$. tanh ($\tanh'(0)=1$) and
ReLU ($\text{ReLU}'=1$ where active) do not have this problem.

<p align="center"><img src="docs/lesson03/activations.png" width="50%"></p>

### 3. Optimisers

| method | update |
|---|---|
| SGD | $\theta\leftarrow\theta-\eta g$ |
| momentum | $v\leftarrow\mu v-\eta g,\;\theta\leftarrow\theta+v$ (a heavy ball: it accumulates speed along consistent directions and damps oscillations across a valley) |
| Adam | $m\leftarrow\beta_1m+(1-\beta_1)g,\;s\leftarrow\beta_2s+(1-\beta_2)g^2,\;\theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat s}+\epsilon)$ with bias corrections $\hat m=m/(1-\beta_1^t)$, $\hat s=s/(1-\beta_2^t)$ |

The Rosenbrock function $f=(1-x)^2+100(y-x^2)^2$ is a curved, narrow valley. Plain GD needs a tiny
step to stay stable across the valley, so it crawls along it. Adam divides each coordinate's step
by its own gradient scale and reaches the minimum.

<p align="center"><img src="docs/lesson03/optimisers_rosenbrock.png" width="48%"> <img src="docs/lesson03/optimisers_mlp.png" width="46%"></p>

### 4. Learning-rate schedules

Step decay (÷10 at 50 % and 75 % of training), cosine decay
$\eta_t=\tfrac12\eta_0(1+\cos\pi t/T)$, and linear warm-up. On this small problem all schedules
end within a fraction of a percent of each other. That is an honest negative result: schedules
matter most for large models and long training runs.

<p align="center"><img src="docs/lesson03/lr_schedules.png" width="90%"></p>

### 5. Batch normalisation

BN re-centres and re-scales each feature using batch statistics. This keeps the pre-activations
in the useful, non-saturated range of the sigmoid, whatever the previous layers do. At test time
it uses running averages. A 10-layer sigmoid network does not train at all without BN; with BN,
it does.

<p align="center"><img src="docs/lesson03/batchnorm.png" width="50%"></p>

### 6. Overfitting, regularisation, early stopping

150 training points with **20 % flipped labels** and a network with 67 000 parameters. Without
regularisation the network memorises the noise: training accuracy approaches 100 % while the
validation loss climbs. The remedies:

* **weight decay**, $L+\frac\lambda2\lVert W\rVert^2$, which prefers small weights and therefore
  smooth functions;
* **dropout**, which trains an ensemble of thinned networks;
* **early stopping** at the minimum of the validation loss.

Note that the "training loss" plotted with weight decay includes the penalty term.

<p align="center"><img src="docs/lesson03/overfitting.png" width="95%"></p>

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

---

## Lessons 04-10

Lessons 04 (PyTorch), 05 (CNNs), 06 (LSTMs), 07 (autoencoders), 08 (attention and GPT),
09 (PINNs and surrogates) and 10 (VAE and diffusion) are being added lesson by lesson through pull
requests. Each one gets a section like the ones above.

## Lessons learned / bugs found

* **A gradient checker can be wrong too.** The first version of `gradient_check` copied the
  backprop gradient of each tensor *inside* the loop, after the previous tensor's
  finite-difference evaluations had re-run `backward` at a **perturbed** point. The reported
  error scaled like $\epsilon$ (1e-6 at $\epsilon$=1e-5, 1e-7 at $\epsilon$=1e-6) instead of
  $\epsilon^2$. The fix is to snapshot all analytic gradients before perturbing anything. The
  tell-tale sign: an error that is proportional to $\epsilon$.
* **Element-wise relative errors mislead.** A gradient entry of $10^{-7}$ has a relative round-off
  error of about $10^{-3}$ at $\epsilon=10^{-6}$ even when the code is right. We therefore compare
  whole tensors (norm-wise).
* **A bias before BatchNorm has an exactly zero gradient.** BN subtracts the batch mean, so the
  bias cancels. The relative error of $0/0$ then reads as 1.0. The bias is redundant, so it is
  removed.
* **ReLU kinks.** With $\epsilon=10^{-5}$ one ReLU check failed (error 2e-2): a pre-activation was
  within $10^{-5}$ of zero. Smaller steps, or a smooth activation, avoid this.
* **Overfitting needs a reason to happen.** With clean, angle-noise-only spirals, even a huge
  network generalised (99 % validation accuracy), so the first demo showed nothing. Label noise
  makes memorisation visible.

## Repository layout and tests

```
nnaz/                 shared, importable course code (one module per topic)
lessons/NN_topic/     the runnable lesson scripts
notebooks/            the lessons as Jupyter notebooks (generated by run.py, executed)
docs/lessonNN/        figures and GIFs;  docs/results/*.json: numbers of the last run
tests/                pytest: gradient checks, loss decreases, beats-baseline, error thresholds
run.py                regenerates every figure, GIF, table and notebook
```

The tests (`pytest`, under 2 minutes on CPU) check the following:

* hand gradients and backprop against finite differences (< 1e-6);
* that the loss decreases in smoke runs;
* that every model beats a trivial baseline;
* the scientific-ML error thresholds.

## References

* F. Rosenblatt (1958), *The perceptron: a probabilistic model for information storage and organization in the brain*, Psychological Review 65.
* A. Novikoff (1962), *On convergence proofs for perceptrons*, Symposium on the Mathematical Theory of Automata.
* D. Rumelhart, G. Hinton, R. Williams (1986), *Learning representations by back-propagating errors*, Nature 323.
* X. Glorot, Y. Bengio (2010), *Understanding the difficulty of training deep feedforward neural networks*, AISTATS.
* K. He, X. Zhang, S. Ren, J. Sun (2015), *Delving deep into rectifiers*, ICCV.
* S. Ioffe, C. Szegedy (2015), *Batch normalization*, ICML.
* N. Srivastava et al. (2014), *Dropout: a simple way to prevent neural networks from overfitting*, JMLR 15.
* D. Kingma, J. Ba (2015), *Adam: a method for stochastic optimization*, ICLR.
* I. Loshchilov, F. Hutter (2017), *SGDR: stochastic gradient descent with warm restarts*, ICLR.
* A. Karpathy, *CS231n notes* and *char-rnn* (Tiny Shakespeare data).
