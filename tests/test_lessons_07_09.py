"""Lessons 07-09: autoencoders vs PCA, attention/GPT, PINN and surrogate."""
import numpy as np
import torch

from nnaz.attention import CharData, ScratchMHA, TinyGPT, attention_numpy, ngram_baselines, train_gpt
from nnaz.autoencoder import LinearAE, pca_fit, train_ae
from nnaz.burgers import cole_hopf, fd_solve
from nnaz.data import load_mnist, load_text
from nnaz.pinn import BurgersPINN, pinn_eval_grid, train_pinn

torch.set_num_threads(2)


def test_linear_autoencoder_reaches_pca_optimum():
    x, _ = load_mnist("MNIST", True)
    X = x[:3000].reshape(3000, -1).astype(np.float64) / 255
    mu, V, s = pca_fit(X)
    k = 8
    optimum = (s[k:] ** 2).sum() / X.size                    # Eckart-Young
    Xc = torch.as_tensor(X - mu, dtype=torch.float32)
    torch.manual_seed(0)
    ae = LinearAE(784, k)
    hist, _ = train_ae(ae, Xc, epochs=40, lr=3e-3)
    with torch.no_grad():
        mse = float(((ae(Xc) - Xc) ** 2).mean())
    assert hist[-1] < hist[0]
    assert optimum <= mse * (1 + 1e-4)                       # nothing beats the SVD
    assert mse < 1.05 * optimum                              # and the linear AE gets there


def test_attention_matches_torch():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(3, 5, 4))
    out, A = attention_numpy(Q, K, V, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(torch.from_numpy(a)[None] for a in (Q, K, V)), is_causal=True)[0].numpy()
    assert np.abs(out - ref).max() < 1e-12 and np.allclose(np.triu(A, 1), 0)
    torch.manual_seed(0)
    mine, ref_mha = ScratchMHA(16, 4), torch.nn.MultiheadAttention(16, 4, batch_first=True)
    with torch.no_grad():
        ref_mha.in_proj_weight.copy_(mine.qkv.weight); ref_mha.in_proj_bias.copy_(mine.qkv.bias)
        ref_mha.out_proj.weight.copy_(mine.proj.weight); ref_mha.out_proj.bias.copy_(mine.proj.bias)
        x = torch.randn(2, 7, 16)
        mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
        assert (mine(x) - ref_mha(x, x, x, attn_mask=mask, need_weights=False)[0]).abs().max() < 1e-5


def test_tiny_gpt_beats_unigram_quickly():
    data = CharData(load_text(100_000))
    base = ngram_baselines(data)
    torch.manual_seed(0)
    model = TinyGPT(data.vocab, block_size=32, d=64, n_layer=2, n_head=4)
    hist = train_gpt(model, data, steps=150, B=16, eval_every=150)
    assert hist["val"][-1] < hist["val"][0]
    assert hist["val"][-1] < base["unigram"]


def test_fd_solver_second_order_against_cole_hopf():
    errs = []
    for N in (512, 1024):
        x, u, _ = fd_solve(0.01 / np.pi, 1.0, 1.0, N=N)
        ex = cole_hopf(x, 1.0)
        errs.append(np.linalg.norm(u[0] - ex) / np.linalg.norm(ex))
    assert errs[1] < 2e-3
    assert 3.0 < errs[0] / errs[1] < 5.0                     # factor ~4 = second order


def test_pinn_error_below_threshold():
    """Smooth case nu = 0.1/pi so the test stays fast; threshold on the rel. L2 error."""
    nu = 0.1 / np.pi
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        model = BurgersPINN(width=20, depth=4)
        x, t = np.linspace(-1, 1, 101), np.linspace(0, 1, 51)
        exact = cole_hopf(x[None], t[:, None], nu)
        hist = train_pinn(model, nu, n_colloc=1500, adam_steps=500, lbfgs_steps=300, eval_every=100, log=False)
        err = np.linalg.norm(pinn_eval_grid(model, x, t) - exact) / np.linalg.norm(exact)
    finally:
        torch.set_default_dtype(torch.float32)
    assert hist["loss"][-1] < hist["loss"][0]
    assert err < 1e-2, err
