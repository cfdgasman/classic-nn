<!-- generated from the root README by nnaz/report.py - edit the root README instead -->
[← 05_cnn_mnist](../05_cnn_mnist/README.md) · [course overview](../../README.md) · [07_autoencoders →](../07_autoencoders/README.md)

# Lesson 06 - RNN and LSTM forecasting of a nonlinear oscillator

📄 [`lessons/06_rnn_lstm/lesson.py`](../../lessons/06_rnn_lstm/lesson.py) · 📓 [`notebooks/06_rnn_lstm.ipynb`](../../notebooks/06_rnn_lstm.ipynb) · code: [`nnaz/rnn.py`](../../nnaz/rnn.py)

### Data: the Van der Pol oscillator

$$ \ddot x-\mu(1-x^2)\dot x+x=0,\qquad \mu=2 . $$

The nonlinear damping pumps energy in for $|x|<1$ and removes it for $|x|>1$. Every trajectory
therefore converges to a **limit cycle** of period about 7.6, with slow phases and fast jumps
(relaxation oscillations). We integrate it with SciPy's RK45 at tight tolerances
(`rtol = atol = 1e-10`), sample every $\Delta t = 0.1$, and add Gaussian observation noise
with $\sigma = 0.1$.

<p align="center"><img src="../../docs/lesson06/vdp_data.png" width="85%"></p>

### Recurrent networks

A vanilla RNN reads the window one value at a time and updates a hidden state,
$h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$. Backpropagation *through time* multiplies $T$ Jacobians
$\partial h_t/\partial h_{t-1}$. The product vanishes or explodes geometrically, which is the
same mechanism as the deep sigmoid network of lesson 03, only now along time.
The **LSTM** fixes this with a separate cell state that is updated *additively* and controlled
by gates:

$$
\begin{aligned}
f_t&=\sigma(W_f[h_{t-1},x_t]+b_f), & i_t&=\sigma(W_i[h_{t-1},x_t]+b_i), & o_t&=\sigma(W_o[h_{t-1},x_t]+b_o),\\
g_t&=\tanh(W_g[h_{t-1},x_t]+b_g), & c_t&=f_t\odot c_{t-1}+i_t\odot g_t, & h_t&=o_t\odot\tanh(c_t).
\end{aligned}
$$

When the forget gate $f_t\approx1$, the gradient flows through $c_t$ almost unchanged: a
"gradient highway". Our models read a window of $p = 40$ noisy samples and predict the
**increment** $x_{t+1}-x_t$ from the last hidden state (a residual formulation). They are trained
with Adam, a cosine learning-rate schedule and gradient-norm clipping at 1.0, which guards
against exploding gradients.

### Baselines

* **persistence**: $\hat x_{t+1}=x_t$;
* **linear AR($p$)**: $\hat x_{t+1}=a\cdot(x_{t-p+1},\dots,x_t)+c$, solved *exactly* by least
  squares. This is the optimal *linear* one-step predictor and a classical, strong baseline.

### One-step vs closed-loop forecasting

A **one-step** forecast always sees real observations. A **closed-loop** (autoregressive)
forecast feeds each prediction back in as the next input. Errors then compound, and the model
must have learnt the *dynamics*, not just local smoothing.

<p align="center"><img src="../../docs/lesson06/rollout.png" width="95%"><br><img src="../../docs/lesson06/forecast.gif" width="55%"></p>

<!-- results:06 -->
| model | 1-step RMSE vs noisy obs. | 1-step RMSE vs true state | closed loop, 10 steps | 50 steps | 150 steps |
|---|---|---|---|---|---|
| persistence | 0.2023 | 0.1755 | 1.292 | 2.585 | 0.406 |
| linear AR(40) | 0.1076 | 0.0457 | 0.047 | 0.053 | 0.056 |
| RNN | 0.1037 | 0.0331 | 0.060 | 0.048 | 0.022 |
| LSTM | 0.1025 | 0.0300 | 0.047 | 0.038 | 0.020 |

The noise floor is 0.100: even a perfect model scores 0.100 against the noisy observations.
<!-- /results:06 -->

**Honest reading of the numbers.** On noisy observations, all learnt models are close to the
noise floor (0.100), so the fair comparison is against the **true state**. There the LSTM is
the most accurate, both one step ahead and in closed loop. It keeps the phase and the sharp
jumps of the limit cycle for two full periods. The linear AR model is a *much* stronger
baseline than people expect: a periodic signal is a sum of harmonics, and any sum of $m$
sinusoids obeys an exact linear recurrence of order $2m$. A 40-tap linear filter captures most
of it, so the LSTM's advantage is real but modest (1.4-3× lower closed-loop error depending on the
horizon). With a shorter window, or on a less regular (e.g. chaotic) signal, the gap would be larger.

## Run it

```bash
python lessons/06_rnn_lstm/lesson.py            # script: figures -> docs/lesson06/
jupyter lab notebooks/06_rnn_lstm.ipynb       # the same lesson as a notebook
```

---
[← 05_cnn_mnist](../05_cnn_mnist/README.md) · [course overview](../../README.md) · [07_autoencoders →](../07_autoencoders/README.md)
