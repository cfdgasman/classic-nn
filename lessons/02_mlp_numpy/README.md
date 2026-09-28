<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 01_perceptron](../01_perceptron/README.md) · [course overview](../../README.md) · [03_training_craft →](../03_training_craft/README.md)

# Lesson 02 - A multilayer perceptron from scratch in NumPy

📄 [`lessons/02_mlp_numpy/lesson.py`](../../lessons/02_mlp_numpy/lesson.py) · 📓 [`notebooks/02_mlp_numpy.ipynb`](../../notebooks/02_mlp_numpy.ipynb) · code: [`nnaz/mlp_numpy.py`](../../nnaz/mlp_numpy.py)

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

<p align="center"><img src="../../docs/lesson02/gradcheck_eps.png" width="55%"></p>

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

<p align="center"><img src="../../docs/lesson02/spiral_mlp.png" width="95%"></p>

## Run it

```bash
python lessons/02_mlp_numpy/lesson.py            # script: figures -> docs/lesson02/
jupyter lab notebooks/02_mlp_numpy.ipynb       # the same lesson as a notebook
```

---
[← 01_perceptron](../01_perceptron/README.md) · [course overview](../../README.md) · [03_training_craft →](../03_training_craft/README.md)
