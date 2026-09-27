"""Lessons 01-03: hand-derived gradients, backprop, and training smoke tests."""
import numpy as np
import pytest

from nnaz.data import blobs, linearly_separable, spirals
from nnaz.mlp_numpy import MLP, gradient_check, input_gradient_check, train
from nnaz.neuron import (bce_grad, bce_loss, logistic_gd, logistic_reference, novikoff_bound,
                         numerical_grad, perceptron_train, predict, rel_error)


def test_perceptron_converges_within_novikoff_bound():
    rng = np.random.default_rng(0)
    X, y = linearly_separable(200, rng)
    w, b, mistakes, _ = perceptron_train(X, y, rng=rng)
    assert (predict(np.r_[w, b], X) == y).all()
    _, gamma, bound = novikoff_bound(X, y, np.array([1.0, 2.0]) / np.sqrt(5), -0.5)
    assert gamma > 0 and mistakes <= bound


def test_logistic_hand_gradient_and_optimum():
    rng = np.random.default_rng(1)
    X, y = blobs(300, rng)
    p = rng.normal(size=3)
    assert rel_error(bce_grad(p, X, y, 0.01), numerical_grad(lambda q: bce_loss(q, X, y, 0.01), p)) < 1e-6
    p_ref, _ = logistic_reference(X, y, 0.01)
    p_gd, losses, _ = logistic_gd(X, y, lr=1.0, steps=400, lam=0.01)
    assert np.abs(p_gd - p_ref).max() < 1e-6          # GD reaches the exact optimum
    assert losses[-1] < losses[0]
    assert (predict(p_gd, X) == y).mean() > 0.85        # far above the 50% baseline


@pytest.mark.parametrize("kw", [dict(act="tanh"), dict(act="relu"), dict(act="sigmoid"),
                                dict(act="tanh", batchnorm=True, dropout=0.3)])
def test_mlp_gradient_check(kw):
    rng = np.random.default_rng(2)
    X, y = spirals(30, rng)
    net = MLP([2, 8, 8, 3], np.random.default_rng(3), **kw)
    worst, _ = gradient_check(net, X, y, eps=1e-6, weight_decay=1e-3)
    assert worst < 1e-6
    assert input_gradient_check(net, X.copy(), y) < 1e-6


def test_mlp_training_beats_linear_baseline():
    rng = np.random.default_rng(4)
    X, y = spirals(300, rng)
    Xte, yte = spirals(600, rng)
    lin = MLP([2, 3], np.random.default_rng(0))
    train(lin, X, y, epochs=40, lr=1e-2, rng=rng)
    net = MLP([2, 64, 64, 3], np.random.default_rng(0))
    h = train(net, X, y, epochs=60, lr=1e-2, rng=rng)
    assert h["loss"][-1] < 0.2 * h["loss"][0]
    acc_lin = (lin.predict(Xte) == yte).mean()
    acc_mlp = (net.predict(Xte) == yte).mean()
    assert acc_mlp > 0.9 and acc_mlp > acc_lin + 0.3
