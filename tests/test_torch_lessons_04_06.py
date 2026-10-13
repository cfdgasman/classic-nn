"""Lessons 04-06: PyTorch equivalence, CNN on MNIST, recurrent forecasting."""
import numpy as np
import torch
import torch.nn.functional as F

from nnaz.cnn import LogReg, SmallCNN, accuracy, conv2d_im2col, fit, to_tensor_images
from nnaz.data import load_mnist, spirals, van_der_pol
from nnaz.mlp_numpy import MLP
from nnaz.rnn import SeqRegressor, fit_linear_ar, make_windows, net_predict, train_seq
from nnaz.torch_basics import MySoftplus, numpy_vs_torch_gd

torch.set_num_threads(2)


def test_numpy_and_torch_agree_to_round_off():
    X, y = spirals(150, np.random.default_rng(0))
    net = MLP([2, 16, 16, 3], np.random.default_rng(0), act="tanh", init="xavier")
    ln, lt, gdiff = numpy_vs_torch_gd(net, X, y, steps=50, lr=0.1)
    assert gdiff < 1e-13
    assert np.abs(ln - lt).max() < 1e-12
    assert ln[-1] < ln[0]


def test_custom_autograd_function():
    z = torch.randn(10, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(MySoftplus.apply, (z,), eps=1e-6, atol=1e-8)


def test_im2col_convolution_matches_torch():
    rng = np.random.default_rng(0)
    X, K, b = rng.normal(size=(3, 10, 10)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    ref = F.conv2d(torch.from_numpy(X)[None], torch.from_numpy(K), torch.from_numpy(b))[0].numpy()
    assert np.abs(conv2d_im2col(X, K, b) - ref).max() < 1e-12


def test_cnn_beats_baselines_on_mnist_subset():
    xtr, ytr = load_mnist("MNIST", True)
    xte, yte = load_mnist("MNIST", False)
    Xtr, m, s = to_tensor_images(xtr[:4000])
    Xte, _, _ = to_tensor_images(xte[:1000], m, s)
    ytr_t, yte_t = torch.as_tensor(ytr[:4000]), torch.as_tensor(yte[:1000])
    torch.manual_seed(0)
    cnn = SmallCNN()
    h = fit(cnn, Xtr, ytr_t, epochs=3, log_every=5)
    assert h["loss"][-1] < 0.5 * h["loss"][0]                 # the loss decreases
    acc = accuracy(cnn, Xte, yte_t)
    majority = np.bincount(yte[:1000]).max() / 1000
    assert acc > 0.9 and acc > majority + 0.7                  # far above the trivial baseline


def test_lstm_beats_persistence():
    t, clean, noisy = van_der_pol(2500, dt=0.1, noise=0.1, rng=np.random.default_rng(0))
    s = noisy.std()
    x, c = noisy / s, clean / s
    P = 20
    Xtr, ytr = make_windows(x[:1800], P)
    Xte, _ = make_windows(x[1800:], P)
    _, yte = make_windows(c[1800:], P)
    torch.manual_seed(0)
    model = SeqRegressor("lstm", hidden=32)
    h = train_seq(model, Xtr, ytr, Xte, yte, steps=300, log=0)
    assert h["val"][-1] < h["val"][0]
    rmse_lstm = np.sqrt(np.mean((net_predict(model, Xte) - yte) ** 2))
    rmse_persist = np.sqrt(np.mean((Xte[:, -1] - yte) ** 2))
    coef = fit_linear_ar(x[:1800], P)
    assert rmse_lstm < rmse_persist
    assert np.isfinite(coef).all()
