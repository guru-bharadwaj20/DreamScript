"""Phase 6.2.7 - the hand-derived gradient, against finite differences and against autograd.

Every claim in `docs/backprop_derivation.md` is checked here. That is the whole point of the
task: a derivation nobody has run beside a reference implementation is not a result, and the
errors it is prone to - a transpose, a missing sum over the batch, a `1/n` in the wrong place -
are invisible to reading and obvious to a numerical check.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import backprop

ACTIVATIONS = ("relu", "leaky_relu", "gelu", "tanh")


@pytest.fixture
def problem():
    rng = np.random.default_rng(42)
    return {
        "X": rng.normal(size=(64, 12)),
        "y": rng.integers(0, 5, size=64),
        "params": backprop.initialise(12, 8, 5, seed=42),
    }


# -- the forward pass ----------------------------------------------------------------------


def test_softmax_rows_sum_to_one():
    z = np.random.default_rng(0).normal(size=(20, 5)) * 10
    probabilities = backprop.softmax(z)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities > 0).all()


def test_softmax_subtracts_the_row_max_so_large_logits_do_not_overflow():
    """A converged network's logits reach 20-plus; `exp(800)` is inf and the row becomes nan."""
    huge = np.array([[800.0, 799.0, 0.0]])
    probabilities = backprop.softmax(huge)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_the_loss_matches_torchs_cross_entropy(problem):
    torch = pytest.importorskip("torch")

    cache = backprop.forward(problem["X"], problem["params"], "relu")
    mine = backprop.cross_entropy(cache["p"], problem["y"])
    theirs = torch.nn.functional.cross_entropy(
        torch.tensor(cache["z2"]), torch.tensor(problem["y"])
    )
    assert mine == pytest.approx(float(theirs), abs=1e-12)


def test_the_forward_pass_keeps_the_pre_activation(problem):
    """`a1` alone cannot recover `g'(z1)`, so caching `z1` is correctness, not speed."""
    cache = backprop.forward(problem["X"], problem["params"], "relu")
    assert set(cache) >= {"z1", "a1", "z2", "p"}
    assert cache["z1"].shape == cache["a1"].shape


@pytest.mark.parametrize("kind", ACTIVATIONS)
def test_each_activation_matches_torchs_layer(kind):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    layers = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }
    z = np.linspace(-4, 4, 401)
    mine = backprop.activate(z, kind)
    theirs = layers[kind](torch.tensor(z)).numpy()
    assert np.abs(mine - theirs).max() < 1e-12


def test_gelu_uses_the_exact_erf_form_not_the_tanh_approximation():
    """The two differ by 4.7e-04, far above the 1e-6 these tests hold the gradient to."""
    from math import pi, sqrt

    z = np.linspace(-4, 4, 401)
    approximation = 0.5 * z * (1 + np.tanh(sqrt(2 / pi) * (z + 0.044715 * z**3)))
    assert np.abs(backprop.activate(z, "gelu") - approximation).max() > 1e-4


# -- the collapse in section 3 --------------------------------------------------------------


def test_the_collapsed_output_error_equals_the_uncollapsed_jacobian_product(problem):
    """Section 3's cancellation, verified rather than trusted: the (k, k) Jacobian, built."""
    cache = backprop.forward(problem["X"], problem["params"], "relu")
    p, y = cache["p"], problem["y"]
    n, k = p.shape

    # dL/dp: only the true-class entry is nonzero.
    dL_dp = np.zeros_like(p)
    dL_dp[np.arange(n), y] = -1.0 / (n * p[np.arange(n), y])

    # The full softmax Jacobian per row, diag(p) - p p^T, composed with dL/dp.
    uncollapsed = np.empty_like(p)
    for i in range(n):
        jacobian = np.diag(p[i]) - np.outer(p[i], p[i])
        uncollapsed[i] = dL_dp[i] @ jacobian

    assert np.abs(backprop.softmax_cross_entropy_grad(p, y) - uncollapsed).max() < 1e-12


def test_the_output_error_carries_the_one_over_n_from_the_mean(problem):
    """Dropping the 1/n still trains, and makes every gradient n times too large."""
    cache = backprop.forward(problem["X"], problem["params"], "relu")
    delta = backprop.softmax_cross_entropy_grad(cache["p"], problem["y"])
    # Each row of (p - onehot) sums to zero, so the check is on the magnitude, not the sum.
    unscaled = cache["p"].copy()
    unscaled[np.arange(len(problem["y"])), problem["y"]] -= 1.0
    assert np.abs(delta * len(problem["y"]) - unscaled).max() < 1e-12


# -- the gradients, two independent references -----------------------------------------------


@pytest.mark.parametrize("kind", ACTIVATIONS)
def test_the_analytic_gradient_matches_central_differences(kind):
    result = backprop.check(kind=kind)
    assert result["passes_at_1e-6"], result["max_relative_error"]


@pytest.mark.parametrize("kind", ACTIVATIONS)
def test_the_analytic_gradient_matches_torch_autograd(kind, problem):
    """The reference with no truncation error - agreement here is to machine precision."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    layers = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }
    params = problem["params"]
    tensors = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in params.items()
    }
    X = torch.tensor(problem["X"], dtype=torch.float64)
    z1 = X @ tensors["W1"] + tensors["b1"]
    z2 = layers[kind](z1) @ tensors["W2"] + tensors["b2"]
    nn.functional.cross_entropy(z2, torch.tensor(problem["y"])).backward()

    mine = backprop.backward(
        problem["X"], problem["y"], params, backprop.forward(problem["X"], params, kind), kind
    )
    for name in ("W1", "b1", "W2", "b2"):
        assert np.abs(mine[name] - tensors[name].grad.numpy()).max() < 1e-12, name


def test_every_gradient_has_the_shape_of_its_parameter(problem):
    params = problem["params"]
    grads = backprop.backward(
        problem["X"], problem["y"], params, backprop.forward(problem["X"], params), "relu"
    )
    for name in ("W1", "b1", "W2", "b2"):
        assert grads[name].shape == params[name].shape


def test_the_bias_gradient_is_a_sum_over_rows_not_a_mean(problem):
    """The 1/n is already inside delta2; averaging again would shrink the biases by n."""
    params = problem["params"]
    cache = backprop.forward(problem["X"], params, "relu")
    grads = backprop.backward(problem["X"], problem["y"], params, cache, "relu")
    assert np.abs(grads["b2"] - grads["delta2"].sum(axis=0)).max() < 1e-15
    assert np.abs(grads["b1"] - grads["delta1"].sum(axis=0)).max() < 1e-15


def test_the_hidden_error_uses_g_prime_at_the_pre_activation(problem):
    """Section 5's warning: for ReLU `a1` coincidentally works, for tanh it is simply wrong."""
    params = problem["params"]
    cache = backprop.forward(problem["X"], params, "tanh")
    correct = backprop.backward(problem["X"], problem["y"], params, cache, "tanh")
    wrong = (correct["delta2"] @ params["W2"].T) * backprop.activate_grad(cache["a1"], "tanh")
    assert np.abs(correct["delta1"] - wrong).max() > 1e-6


# -- the properties section 7 asserts about the four activations ------------------------------


def test_relu_and_leaky_relu_differ_only_below_zero():
    z = np.linspace(-3, 3, 601)
    same = z > 0
    assert np.allclose(
        backprop.activate_grad(z, "relu")[same], backprop.activate_grad(z, "leaky_relu")[same]
    )
    assert (backprop.activate_grad(z, "relu")[~same] == 0).all()
    assert (backprop.activate_grad(z, "leaky_relu")[~same] == 0.01).all()


def test_tanh_saturates_where_the_vanishing_gradient_argument_says_it_does():
    assert backprop.activate_grad(np.array([3.5]), "tanh")[0] < 0.005
    assert backprop.activate_grad(np.array([0.0]), "tanh")[0] == pytest.approx(1.0)


def test_an_unknown_activation_is_refused_in_both_directions():
    for function in (backprop.activate, backprop.activate_grad):
        with pytest.raises(ValueError, match="unknown activation"):
            function(np.zeros(3), "swish")


# -- section 8's parameter count ---------------------------------------------------------------


def test_the_worked_parameter_count_in_section_8_is_right():
    """161 -> 512 -> 256 -> 5 is 215,557, which is the figure 6.2.1 reported."""
    from src.classify.mlp import parameter_count

    assert parameter_count(161, (512, 256), 5) == 215_557


def test_the_document_exists_and_states_the_shapes_it_derives():
    from pathlib import Path

    text = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("docs", "backprop_derivation.md")
        .read_text(encoding="utf-8")
    )
    assert "delta2 = (p - onehot(y)) / n" in text
    assert "215,557" in text
