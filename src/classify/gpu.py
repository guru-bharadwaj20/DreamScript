"""Phase 5 - the GPU paths, and an honest account of when they are worth using.

This machine has an RTX 4500 Ada with 24 GB. Phase 5's models are classical and its corpus is
small - 1,340 rows by 33 features - so the interesting question is not "can these run on the
GPU" but **"where does it actually pay, and where is saying so more useful than a speed-up
claim"**. Two operations are implemented here and both are measured in the modules that use
them:

* `logreg_fit` - multinomial logistic regression by full-batch L-BFGS on the GPU. It solves the
  same penalised objective sklearn does, and 5.1.1 checks that it agrees with sklearn rather
  than assuming it.
* `knn_predict` - brute-force k-nearest-neighbours by one dense distance matrix. This is the
  one that genuinely wins: 5.1.2's sweep is 6 values of k by 3 metrics by 2 weightings across 15
  folds, and every one of those 540 fits is a fresh distance computation on the CPU while on the
  GPU **the distance matrix is computed once per fold and every k reads from it.**

## What the measurements said

Recorded here so the claim and the evidence sit together; the runs are in 5.1.1 and 5.1.2.

    operation                              CPU (32 cores)   GPU        speed-up
    logistic regression sweep, 5.1.1           (see 5.1.1)
    kNN sweep, 5.1.2                           (see 5.1.2)

`available()` is checked at import time by every caller, and every GPU path has a CPU
equivalent that produces the same answer, because a result that only exists on one machine is
not a result. The tests run both and compare.

## Determinism

`torch.use_deterministic_algorithms` is not forced globally - it would slow the rest of the
project's CUDA work - but everything here is written to be deterministic anyway: L-BFGS from a
fixed zero initialisation, and a distance matrix whose reduction order does not depend on
scheduling. Ties in kNN are broken by the lowest class index, matching sklearn.
"""

from __future__ import annotations

import numpy as np

#: Below this many rows the kernel launch and the host-to-device copy cost more than the work.
#: Measured, not guessed - see 5.1.2's benchmark.
MIN_ROWS_FOR_GPU = 512


def available() -> bool:
    """True when a CUDA device is present and usable."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a hard dependency of this project
        return False


def device_name() -> str:
    if not available():
        return "cpu"
    import torch

    return torch.cuda.get_device_name(0)


def _tensor(array, dtype=None):
    import torch

    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype or torch.float32).cuda()


def logreg_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-9,
    class_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression on the GPU. Returns (coef, intercept).

    The objective is sklearn's: mean cross-entropy plus `1 / (2 C)` times the squared L2 norm of
    the coefficients, with the intercept unpenalised. Solved by L-BFGS from zeros, which is
    deterministic and converges in tens of iterations on a problem this size.
    """
    import torch

    features = _tensor(X, torch.float64)
    classes = np.unique(y)
    index = torch.as_tensor(np.searchsorted(classes, y).astype(np.int64), dtype=torch.int64).cuda()
    weights = None
    if class_weight is not None:
        weights = torch.as_tensor(np.asarray(class_weight), dtype=torch.float64).cuda()

    n_features = features.shape[1]
    n_classes = len(classes)
    coef = torch.zeros((n_classes, n_features), dtype=torch.float64, device="cuda")
    intercept = torch.zeros(n_classes, dtype=torch.float64, device="cuda")
    coef.requires_grad_(True)
    intercept.requires_grad_(True)

    optimiser = torch.optim.LBFGS(
        [coef, intercept],
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimiser.zero_grad(set_to_none=True)
        logits = features @ coef.T + intercept
        loss = torch.nn.functional.cross_entropy(logits, index, weight=weights, reduction="mean")
        loss = loss + coef.pow(2).sum() / (2.0 * C * len(features))
        loss.backward()
        return loss

    optimiser.step(closure)
    return (
        coef.detach().cpu().numpy(),
        intercept.detach().cpu().numpy(),
    )


def logreg_decision(X: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.asarray(X, float) @ np.asarray(coef, float).T + np.asarray(intercept, float)


def pairwise(train: np.ndarray, test: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Dense distance matrix (n_test, n_train) on the GPU.

    This is the whole GPU story for kNN: one matrix, reused by every k and every weighting.
    """
    import torch

    a = _tensor(test, torch.float32)
    b = _tensor(train, torch.float32)
    if metric == "euclidean":
        distance = torch.cdist(a, b, p=2)
    elif metric == "manhattan":
        distance = torch.cdist(a, b, p=1)
    elif metric == "cosine":
        a_normalised = torch.nn.functional.normalize(a, dim=1, eps=1e-12)
        b_normalised = torch.nn.functional.normalize(b, dim=1, eps=1e-12)
        distance = 1.0 - a_normalised @ b_normalised.T
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return distance


def knn_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    *,
    k: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    classes: np.ndarray | None = None,
    distances=None,
) -> tuple[np.ndarray, np.ndarray]:
    """(predicted labels, class probabilities) by brute-force kNN.

    `distances` lets a caller compute the matrix once and sweep k over it, which is the reason
    this exists at all.
    """
    import torch

    if classes is None:
        classes = np.unique(train_y)
    labels = torch.as_tensor(
        np.searchsorted(classes, train_y).astype(np.int64), dtype=torch.int64
    ).cuda()

    if distances is None:
        distances = pairwise(train_X, test_X, metric)
    k = min(k, distances.shape[1])
    nearest, index = torch.topk(distances, k, dim=1, largest=False, sorted=True)
    neighbour_labels = labels[index]

    if weights == "distance":
        # sklearn's rule: an exact match takes the whole vote.
        weight = 1.0 / nearest.clamp_min(1e-12)
        exact = nearest <= 1e-12
        rows_with_exact = exact.any(dim=1)
        weight[rows_with_exact] = exact[rows_with_exact].to(weight.dtype)
    else:
        weight = torch.ones_like(nearest)

    votes = torch.zeros((distances.shape[0], len(classes)), dtype=weight.dtype, device="cuda")
    votes.scatter_add_(1, neighbour_labels, weight)
    probability = votes / votes.sum(dim=1, keepdim=True).clamp_min(1e-12)
    # `argmax` breaks ties by the lowest index, which is sklearn's behaviour on sorted classes.
    chosen = probability.argmax(dim=1).cpu().numpy()
    return np.asarray(classes)[chosen], probability.cpu().numpy()
