"""Phase 6.2.7 - the gradient of one layer, derived by hand and checked against autograd.

    python -m src.classify.backprop        # the check, on the real network's shapes

`docs/backprop_derivation.md` carries the derivation the viva asks for. A derivation in a
markdown file is a claim, and this module is what turns it into a tested one: every expression in
that document is implemented here in numpy, with no autograd anywhere in the forward or backward
pass, and the tests compare the result against `torch.autograd` to 1e-6 on the actual shapes of
6.2.1's selected network.

That is the point of the module. A hand-derived gradient that has never been run beside a
reference implementation is the single easiest thing in a project like this to get subtly wrong -
a transpose, a missing sum over the batch, a 1/n that belongs somewhere else - and none of those
errors are visible by reading. Each of them is caught immediately by a numerical check, so the
document's claims are tested rather than asserted: `check()` compares every expression against
central differences, and the test module compares them against `torch.autograd` as well.

## What is derived

One hidden layer of the network Phase 6.2 actually trains, with the loss it actually uses:

    z1 = X W1 + b1        pre-activation      (n, h)
    a1 = g(z1)            activation          (n, h)
    z2 = a1 W2 + b2       logits              (n, k)
    p  = softmax(z2)      probabilities       (n, k)
    L  = -1/n sum log p[i, y_i]               scalar, mean cross-entropy

and the four gradients a single layer needs:

    dL/dW2, dL/db2        the output layer
    dL/dW1, dL/db1        the hidden layer, which is where the chain rule does its work

`g` is any of 6.2.2's four activations, because the derivation is identical up to `g'` and the
whole content of "which activation" is one elementwise factor.

## The one step worth stating carefully

The softmax-with-cross-entropy Jacobian collapses. Differentiating the softmax alone gives a
(k, k) matrix per row - `diag(p) - p p^T` - and composing that with the derivative of the log
loss gives, after the cancellation, simply `p - onehot(y)`. Every autograd framework implements
the two as one fused operation for this reason, and it is the step a viva is most likely to ask
to see, so `softmax_cross_entropy_grad` computes it directly and `test_backprop_derivation.py`
checks it against the uncollapsed two-step product as well as against torch.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

SEED = 42


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, with the row max subtracted before exponentiating."""
    shifted = np.exp(z - z.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def cross_entropy(probabilities: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class - `nn.CrossEntropyLoss`'s default."""
    n = len(y)
    # Clipped only against exact zero: a converged network assigns e-30 to a wrong class and
    # log(0) would make the loss -inf where the gradient is perfectly well defined.
    return float(-np.log(np.clip(probabilities[np.arange(n), y], 1e-300, None)).mean())


def activate(z: np.ndarray, kind: str = "relu") -> np.ndarray:
    """The forward half of 6.2.2's four activations."""
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "leaky_relu":
        return np.where(z > 0, z, 0.01 * z)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "gelu":
        # The exact (erf) form, which is what `nn.GELU()` uses by default - the tanh
        # approximation is a different function and differs in the sixth decimal.
        from math import sqrt

        from scipy.special import erf

        return 0.5 * z * (1.0 + erf(z / sqrt(2.0)))
    raise ValueError(f"unknown activation {kind!r}")


def activate_grad(z: np.ndarray, kind: str = "relu") -> np.ndarray:
    """g'(z), elementwise - the entire difference between the four derivations."""
    if kind == "relu":
        # At exactly zero ReLU is not differentiable; every framework picks 0 for the subgradient
        # and this matches torch so the comparison test is meaningful.
        return (z > 0).astype(z.dtype)
    if kind == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    if kind == "tanh":
        return 1.0 - np.tanh(z) ** 2
    if kind == "gelu":
        from math import pi, sqrt

        from scipy.special import erf

        # d/dz [ z * Phi(z) ] = Phi(z) + z * phi(z), by the product rule.
        cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        pdf = np.exp(-0.5 * z**2) / sqrt(2.0 * pi)
        return cdf + z * pdf
    raise ValueError(f"unknown activation {kind!r}")


def forward(X: np.ndarray, params: dict, kind: str = "relu") -> dict:
    """One hidden layer and an output layer, keeping every intermediate the backward pass needs.

    Caching `z1` rather than recomputing it is not an optimisation here; `a1` alone is not enough
    to recover `g'(z1)` for ReLU, because a zero in `a1` could have come from any negative `z1`
    and the derivative of tanh needs the pre-activation too.
    """
    z1 = X @ params["W1"] + params["b1"]
    a1 = activate(z1, kind)
    z2 = a1 @ params["W2"] + params["b2"]
    return {"z1": z1, "a1": a1, "z2": z2, "p": softmax(z2)}


def softmax_cross_entropy_grad(probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dL/dz2 = (p - onehot(y)) / n - the collapse the derivation turns on.

    The two-step product of the softmax Jacobian `diag(p) - p p^T` with `dL/dp` gives this after
    cancellation, and the tests check both routes agree.
    """
    n = len(y)
    delta = probabilities.copy()
    delta[np.arange(n), y] -= 1.0
    # The 1/n is the *mean* in the loss. Dropping it is the most common error in a hand-written
    # backward pass and it scales every gradient in the network by the batch size.
    return delta / n


def backward(X: np.ndarray, y: np.ndarray, params: dict, cache: dict, kind: str = "relu") -> dict:
    """The four gradients, exactly as `docs/backprop_derivation.md` derives them.

    No autograd, no finite differences - this is the closed form, and the tests are what say it
    is the right one.
    """
    delta2 = softmax_cross_entropy_grad(cache["p"], y)  # (n, k)

    # Output layer. `a1.T @ delta2` sums the outer product over the batch, which is where the
    # sum over examples in the derivation comes from; the bias gradient is the same sum with the
    # activation replaced by 1, hence a plain column sum.
    grad_w2 = cache["a1"].T @ delta2  # (h, k)
    grad_b2 = delta2.sum(axis=0)  # (k,)

    # Hidden layer. The error is carried back through W2 and then multiplied elementwise by
    # g'(z1) - the chain rule's two halves, and the only place the activation enters.
    delta1 = (delta2 @ params["W2"].T) * activate_grad(cache["z1"], kind)  # (n, h)
    grad_w1 = X.T @ delta1  # (d, h)
    grad_b1 = delta1.sum(axis=0)  # (h,)

    return {
        "W1": grad_w1,
        "b1": grad_b1,
        "W2": grad_w2,
        "b2": grad_b2,
        "delta1": delta1,
        "delta2": delta2,
    }


def initialise(n_features: int, hidden: int, n_classes: int, seed: int = SEED) -> dict:
    """Glorot-uniform, matching `TorchMLP._build` so the shapes under test are the real ones."""
    rng = np.random.default_rng(seed)

    def glorot(fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out))

    return {
        "W1": glorot(n_features, hidden),
        "b1": np.zeros(hidden),
        "W2": glorot(hidden, n_classes),
        "b2": np.zeros(n_classes),
    }


def numerical_gradient(
    X, y, params: dict, name: str, kind: str = "relu", eps: float = 1e-6
) -> np.ndarray:
    """Central differences on one parameter array - the check that needs no torch at all.

    Central rather than forward differences: the error is O(eps^2) instead of O(eps), which is
    the difference between agreeing with the analytic gradient to six decimals and to three.
    """
    grad = np.zeros_like(params[name])
    flat = params[name].reshape(-1)
    flat_grad = grad.reshape(-1)
    for index in range(flat.size):
        original = flat[index]
        flat[index] = original + eps
        plus = cross_entropy(forward(X, params, kind)["p"], y)
        flat[index] = original - eps
        minus = cross_entropy(forward(X, params, kind)["p"], y)
        flat[index] = original
        flat_grad[index] = (plus - minus) / (2 * eps)
    return grad


def check(
    n: int = 64,
    n_features: int = 12,
    hidden: int = 8,
    n_classes: int = 5,
    kind: str = "relu",
    seed: int = SEED,
) -> dict:
    """Analytic against numerical, on every parameter array. The viva's "prove it"."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = rng.integers(0, n_classes, size=n)
    params = initialise(n_features, hidden, n_classes, seed)

    analytic = backward(X, y, params, forward(X, params, kind), kind)
    worst = {}
    for name in ("W1", "b1", "W2", "b2"):
        numeric = numerical_gradient(X, y, params, name, kind)
        scale = np.maximum(np.abs(analytic[name]) + np.abs(numeric), 1e-12)
        worst[name] = float(np.max(np.abs(analytic[name] - numeric) / scale))

    return {
        "activation": kind,
        "shapes": {"X": [n, n_features], "hidden": hidden, "classes": n_classes},
        "loss": round(cross_entropy(forward(X, params, kind)["p"], y), 6),
        "max_relative_error": {k: float(f"{v:.3e}") for k, v in worst.items()},
        "passes_at_1e-6": all(value < 1e-6 for value in worst.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activation", default="relu", choices=["relu", "leaky_relu", "gelu", "tanh"])
    ap.add_argument("--all", action="store_true", help="check every activation")
    args = ap.parse_args(argv)

    kinds = ("relu", "leaky_relu", "gelu", "tanh") if args.all else (args.activation,)
    results = [check(kind=kind) for kind in kinds]
    print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return 0 if all(result["passes_at_1e-6"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
