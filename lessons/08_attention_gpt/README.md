<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 07_autoencoders](../07_autoencoders/README.md) · [course overview](../../README.md) · [09_scientific_ml →](../09_scientific_ml/README.md)

# Lesson 08 - Attention from scratch, then a tiny GPT

📄 [`lessons/08_attention_gpt/lesson.py`](../../lessons/08_attention_gpt/lesson.py) · 📓 [`notebooks/08_attention_gpt.ipynb`](../../notebooks/08_attention_gpt.ipynb) · code: [`nnaz/attention.py`](../../nnaz/attention.py)

### Attention is a soft lookup table

Each position $i$ emits a **query** $q_i$, a **key** $k_i$ and a **value** $v_i$, which are
linear projections of its embedding. Position $i$ reads a weighted average of all values,
weighted by how well its query matches each key:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(rac{QK^	op}{\sqrt{d_k}}+M\Big)V,
\qquad M_{ij}=egin{cases}0 & j\le i\ -\infty & j>i\end{cases}
$$

* The **causal mask** $M$ stops a position from looking at the future. This is what makes
  next-token prediction a fair game.
* **Why $1/\sqrt{d_k}$?** For random unit-variance $q,k$, the dot product $q\cdot k$ has
  variance $d_k$. Without scaling, the logits grow with the head size and the softmax saturates
  to one-hot, which also kills its gradient. The right panel below measures this.
* **Multi-head**: $h$ attention maps run in parallel on $d/h$-dimensional slices, so different
  heads can track different relations (the previous character, the start of the word, the
  speaker's name, ...).
* Attention alone is **permutation-equivariant**: it does not know the order of its inputs.
  Order enters through **positional embeddings** added to the token embeddings.

<p align="center"><img src="../../docs/lesson08/attention_basics.png" width="85%"></p>

### The GPT block

```
x = token_embedding[idx] + position_embedding[0..T-1]
repeat n_layer times:
    x = x + MultiHeadCausalAttention(LayerNorm(x))   # positions exchange information
    x = x + MLP(LayerNorm(x))                        # per-position computation, 4x wider, GELU
logits = LayerNorm(x) @ token_embedding^T            # weight tying
loss = cross_entropy(logits[t], idx[t+1])            # predict the NEXT character
```

The residual connections $x+f(x)$ give the gradient a direct path through the whole stack,
just like the LSTM cell state does through time. Generation is **autoregressive**: predict a
distribution over the next character, sample from it (temperature 0.8), append, repeat.

Our model has 4 layers, 4 heads, width 128 and a context of 64 characters (0.81 M parameters).
It is trained with AdamW, warm-up and cosine decay for 3000 steps of 32×64 characters, about
5 minutes on a CPU.

<!-- results:08 -->
| model | validation cross-entropy (nats/char) |
|---|---|
| uniform over the 65 characters | 4.174 |
| unigram (character frequencies) | 3.347 |
| bigram (add-one smoothing) | 2.482 |
| TinyGPT, 0.81 M params, 3000 steps (5.0 min CPU) | **1.605** (train 1.404) |

From-scratch attention vs PyTorch: 1.1e-16 (NumPy, float64), 8.9e-08 (multi-head module, float32).
<!-- /results:08 -->

<p align="center"><img src="../../docs/lesson08/gpt_training.png" width="48%"> <img src="../../docs/lesson08/gpt_samples.gif" width="48%"></p>

Validation cross-entropy of 1.6 nats/char means the model is on average as uncertain as a choice
among $e^{1.6}pprox5$ characters, against 12 for the bigram model. The samples have the
*form* of a play: speaker names in capitals, line breaks, archaic words. The sense is poor,
which is expected at 0.8 M parameters and 5 CPU-minutes. The training/validation gap
(1.40 vs 1.61) shows the beginning of overfitting on a 1 MB corpus.
A sample ([`docs/lesson08/sample.txt`](../../docs/lesson08/sample.txt)):

```
ROMEO:
My lord, sir, as I would now you, that noble counter
I take her lamp of quit of the man's plant heart,
...
KING RICHARD III:
O, the should of you have flower cares to no be; though we
```

Attention maps of the trained network show distinct heads. Some attend to the immediately
preceding characters, like an n-gram model. Others attend to earlier positions such as the
start of the current word or the previous line.

<p align="center"><img src="../../docs/lesson08/gpt_attention_maps.png" width="95%"></p>

## Run it

```bash
python lessons/08_attention_gpt/lesson.py            # script: figures -> docs/lesson08/
jupyter lab notebooks/08_attention_gpt.ipynb       # the same lesson as a notebook
```

---
[← 07_autoencoders](../07_autoencoders/README.md) · [course overview](../../README.md) · [09_scientific_ml →](../09_scientific_ml/README.md)
