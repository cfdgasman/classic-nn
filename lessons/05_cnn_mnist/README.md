<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 04_pytorch_basics](../04_pytorch_basics/README.md) · [course overview](../../README.md) · [06_rnn_lstm →](../06_rnn_lstm/README.md)

# Lesson 05 - Convolutional networks on MNIST and Fashion-MNIST

📄 [`lessons/05_cnn_mnist/lesson.py`](../../lessons/05_cnn_mnist/lesson.py) · 📓 [`notebooks/05_cnn_mnist.ipynb`](../../notebooks/05_cnn_mnist.ipynb) · code: [`nnaz/cnn.py`](../../nnaz/cnn.py)

### The convolution

A convolution layer slides a small kernel $K$ over the image. With $C_\text{in}$ input channels:

$$ Y[o,i,j] = b_o + \sum_{c=1}^{C_\text{in}}\sum_{u,v=0}^{k-1} K[o,c,u,v]\;X[c,\,i+u,\,j+v]. $$

(Deep-learning libraries compute a cross-correlation and call it a convolution.) Compared with
a dense layer it builds in two assumptions that suit images and fields:

* **locality**: an output pixel depends only on a $k\times k$ neighbourhood, like a finite-difference stencil;
* **weight sharing**: the same kernel is applied everywhere, so a detector learnt in one place
  works everywhere (translation **equivariance**), and the parameter count does not depend on
  the image size.

A hand-made **Sobel** kernel is a vertical-edge detector, and in fact a central-difference
approximation of $\partial/\partial x$ smoothed in $y$. A CNN *learns* such stencils. The lesson
computes the same convolution three ways: explicit loops, **im2col** (every patch unrolled into a
column, so the convolution becomes one big matrix product, which is how libraries do it), and
`torch.nn.functional.conv2d`. All three agree to round-off.

<p align="center"><img src="../../docs/lesson05/convolution.gif" width="45%"></p>

### Architecture and shapes

```
input 1x28x28 → conv5x5 (16) → ReLU → 16x24x24 → maxpool2 → 16x12x12
              → conv5x5 (32) → ReLU → 32x8x8   → maxpool2 → 32x4x4 → flatten 512
              → Linear 128 → ReLU → dropout 0.3 → Linear 10 (logits)
```

Output size of a "valid" convolution is $n-k+1$; of a $2\times2$ max-pool, $n/2$. **Max-pooling**
adds a little translation *invariance* and halves the resolution, so deeper layers "see" larger
regions: the **receptive field** grows from 5×5 to 16×16 pixels after the second conv/pool pair.

### Results

The comparison uses models with the **same parameter budget** (MLP 76 k vs CNN 80 k), trained
with Adam (lr 1e-3, batch 128) for 5 epochs:

<!-- results:05 -->
| data set | model | parameters | test accuracy | CPU training time |
|---|---|---|---|---|
| MNIST | majority class | - | 11.3% | - |
| MNIST | logistic regression | 7850 | 92.4% | 2 s |
| MNIST | MLP 784-96-10 | 76330 | 97.7% | 4 s |
| MNIST | CNN | 80202 | 99.2% | 33 s |
| FashionMNIST | majority class | - | 10.0% | - |
| FashionMNIST | logistic regression | 7850 | 84.1% | 2 s |
| FashionMNIST | MLP 784-96-10 | 76330 | 87.6% | 3 s |
| FashionMNIST | CNN | 80202 | 89.6% | 31 s |

Convolution implementations agree: loops vs `F.conv2d` 1.4e-14, im2col vs `F.conv2d` 0.0e+00.
<!-- /results:05 -->

<p align="center"><img src="../../docs/lesson05/mnist_confusion.png" width="95%"></p>

The learnt first-layer filters are oriented edge and stroke detectors. The feature maps show
which parts of the digit each filter responds to. The second-layer maps are coarser and more
abstract: corners, stroke ends, curvature.

<p align="center"><img src="../../docs/lesson05/conv1_filters.png" width="70%"><br><img src="../../docs/lesson05/feature_maps.png" width="90%"></p>

**Fashion-MNIST** is much harder: shirt, T-shirt, pullover and coat look alike at 28×28 (see its
confusion matrix). The CNN's advantage over the MLP is smaller there, and about 90 % is typical
for a small network and a few epochs.

<p align="center"><img src="../../docs/lesson05/fashionmnist_confusion.png" width="95%"></p>

## Run it

```bash
python lessons/05_cnn_mnist/lesson.py            # script: figures -> docs/lesson05/
jupyter lab notebooks/05_cnn_mnist.ipynb       # the same lesson as a notebook
```

---
[← 04_pytorch_basics](../04_pytorch_basics/README.md) · [course overview](../../README.md) · [06_rnn_lstm →](../06_rnn_lstm/README.md)
