<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 10_generative](../10_generative/README.md) · [course overview](../../README.md) · [12_dnc_memory →](../12_dnc_memory/README.md)

# Lesson 11 - Self-organising maps and Hopfield networks

📄 [`lessons/11_som_hopfield/lesson.py`](../../lessons/11_som_hopfield/lesson.py) · 📓 [`notebooks/11_som_hopfield.ipynb`](../../notebooks/11_som_hopfield.ipynb) · code: [`nnaz/som_hopfield.py`](../../nnaz/som_hopfield.py)

Two classics that learn **without backpropagation**. Both are still useful to engineers:
SOMs for visualising and clustering high-dimensional data (sensor data, flow regimes, design
spaces), and Hopfield networks as the ancestor of associative memory and of attention.

### Self-organising map (Kohonen, 1982)

A 2-D grid of units, each holding a prototype vector $w_j$ in data space. For each input $x$:

$$
b=\arg\min_j\lVert x-w_j\rVert \quad\text{(competition)},\qquad
w_j\leftarrow w_j+\eta(t)\,e^{-\lVert r_j-r_b\rVert^2/2\sigma(t)^2}\,(x-w_j)\quad\text{(cooperation)},
$$

where $r_j$ is the unit's position **on the grid**. The best-matching unit *and its grid
neighbours* move towards the input, and $\eta$ and $\sigma$ shrink over time. As a result,
units that are neighbours on the grid end up with similar prototypes. The map is a
topology-preserving, discretised 2-D projection of the data: a nonlinear cousin of PCA, or
"k-means with a map". The **batch SOM** (used for MNIST) replaces each prototype by the
neighbourhood-weighted mean of the data. It is deterministic and vectorises well.

Two quality measures:
* **quantisation error**: mean distance from a sample to its best prototype (lower = better fit);
* **topographic error**: fraction of samples whose best and second-best units are *not*
  adjacent on the grid (lower = better ordered).

<p align="center"><img src="../../docs/lesson11/som_unfolding.gif" width="32%"> <img src="../../docs/lesson11/som_mnist.png" width="62%"></p>

The U-matrix shows the average distance between neighbouring prototypes. Dark ridges are
cluster borders; the digit label of each unit is overlaid.

### Hopfield networks: a network with feedback loops

$N$ binary neurons $s_i=\pm1$ with symmetric weights and no self-connections. Every neuron feeds
back into all the others, and asynchronous updates $s_i\leftarrow\operatorname{sign}(\sum_jW_{ij}s_j)$
can never increase the **energy**

$$ E(s)=-\tfrac12\,s^\top Ws . $$

The state therefore rolls downhill into a local minimum. If the memories are minima, this is a
**content-addressable memory**: start from a corrupted pattern and the dynamics complete it.

* **Hebbian rule** ("fire together, wire together"): $W=\frac1N\sum_\mu\xi^\mu\xi^{\mu\top}$.
  Capacity for random patterns is about $0.138N$ (Amit, Gutfreund and Sompolinsky, 1985). It
  assumes nearly **uncorrelated** patterns.
* **Projection (pseudo-inverse) rule**: $W=P^\top(PP^\top)^{-1}P$. Every stored pattern is an
  exact fixed point, even for strongly correlated patterns.
* **Modern Hopfield network** (Ramsauer et al., 2020):
  $s\leftarrow P^\top\operatorname{softmax}(\beta Ps)$. This is exactly the **attention** of
  lesson 08: the stored patterns are keys and values, the probe is the query. Its capacity grows
  exponentially with $N$.

Our letters share most of their white background: the maximum pairwise overlap is about 0.9. The
Hebbian network therefore falls into mixture states and recalls **none** of them exactly. The
projection rule and the modern network recall all of them. On random patterns, the measured
capacity curve collapses near the theoretical $0.138N$.

<p align="center"><img src="../../docs/lesson11/hopfield_patterns.png" width="80%"><br>
<img src="../../docs/lesson11/hopfield_recall.gif" width="30%"> <img src="../../docs/lesson11/hopfield_capacity.png" width="45%"></p>

<!-- results:11 -->
| 15x15 SOM on MNIST (test digits) | SOM | k-means (225 centres) | random data points as prototypes |
|---|---|---|---|
| quantisation error | 5.253 | 5.053 | 6.307 |
| topographic error | 0.085 | n/a (no grid) | 0.971 |
| digit accuracy, unit majority label | 78.4% | 85.1% | majority class: 11.4% |

| Hopfield rule (8 letters, 20 % flipped pixels) | correct pixels after recall, per letter |
|---|---|
| Hebbian | 0.88 0.88 0.95 0.96 0.81 0.98 0.84 0.89 |
| projection (pseudo-inverse) | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 |
| modern (attention) | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 |

| P/N (N = 200 random patterns) | 0.02 | 0.05 | 0.08 | 0.10 | 0.12 | 0.14 | 0.16 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|---|---|---|
| classic (Hebbian) recalled | 1.00 | 1.00 | 1.00 | 0.95 | 0.85 | 0.70 | 0.40 | 0.10 | 0.00 | 0.00 |
| modern (attention) recalled | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
<!-- /results:11 -->

## Run it

```bash
python lessons/11_som_hopfield/lesson.py            # script: figures -> docs/lesson11/
jupyter lab notebooks/11_som_hopfield.ipynb       # the same lesson as a notebook
```

---
[← 10_generative](../10_generative/README.md) · [course overview](../../README.md) · [12_dnc_memory →](../12_dnc_memory/README.md)
