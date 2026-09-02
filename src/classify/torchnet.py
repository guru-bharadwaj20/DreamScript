"""Phase 6.2.2's engine - the training loop 6.2.1 said would be needed, built one task early.

    from src.classify.torchnet import TorchMLP, pipeline
    python -m src.classify.torchnet --smoke        # one fit, prints the loss curve's shape

6.2.1 ran its topology sweep on `sklearn.neural_network.MLPClassifier` and said in its docstring
that a hand-written loop earns its place at 6.2.4, where the optimizer comparison needs the loss
at every epoch. **It is needed one task earlier than that.** The plan's 6.2.2 asks for ReLU vs
LeakyReLU vs GELU vs tanh, and sklearn's `activation` parameter accepts exactly four values:
`identity`, `logistic`, `tanh`, `relu`. Two of the four activations the plan names cannot be
expressed in the estimator 6.2.1 used, so either the task is silently reduced to the two sklearn
happens to have, or the network is rewritten. This is the rewrite.

Once it exists, 6.2.3's dropout, 6.2.4's five optimizers, 6.2.5's schedules and 6.2.6's batch
sizes are all one constructor argument each, and every one of them is a knob sklearn either does
not expose or exposes only partially. So the cost is paid once, here.

## What it has to be compatible with

Everything in Phase 5 is built around scikit-learn's estimator protocol: 5.2.1's `out_of_fold`,
`cross_val_predict`, `GridSearchCV`, and the `feature_scaler` pipeline that 4.2.3's imputation
and 4.2.4's scaling live in. So `TorchMLP` is a `BaseEstimator`/`ClassifierMixin` with `fit`,
`predict`, `predict_proba` and `classes_`, and it takes string labels directly - the problem
6.2.1 needed `StringSafeMLP` to work around does not arise here, because the label encoder is
part of `fit` by construction rather than bolted onto someone else's.

It also reproduces sklearn's defaults where they are defaults rather than accidents: Adam at
1e-3, early stopping on a 12% internal validation split with 15 epochs of patience,
Glorot-uniform initialisation, and the same 600-epoch ceiling. That is deliberate - 6.2.2's
activation table is only readable if its ReLU row can be checked against 6.2.1's sklearn
number, and it can: 0.9325 here against 0.9310 there, on the hybrid table.

## The GPU does not win this workload, and the measurement says so

1,340 rows of 161 float32 columns is 863 KB. The whole training set fits in cache, let alone in
the card's 24 GB, and a (512, 256) forward pass on a batch of 64 is roughly 200,000
multiply-adds - a rounding error for a device that does 10^13 of them a second. So the fit is
paced by Python's per-epoch and per-batch overhead, not by arithmetic, and the expected result
is that the accelerator buys nothing. Measured, one 5-fold CV of one configuration on the hybrid
table:

    device                       seconds   macro F1
    cuda (RTX 4500 Ada)            3.4      0.9411
    cpu                            2.8      0.9411

**The card is slower**, by 20%, for identical output. Every batch is a kernel launch whose
launch cost exceeds its compute cost at this size, and there are twenty epochs of twenty of them
per fold.

That is not an argument against having the hardware; it is an argument about where the
parallelism belongs. A single card cannot run 32 configurations at once, and 32 cores can - so
`sweep` runs *configurations* in parallel on the CPU, one thread each, and 6.2.3 through 6.2.6
get their speed from there. Measured on the same machine: **twelve configurations in 11.4 s,
against 67.8 s of serial fitting**, a 6x wall-clock gain on a workload the GPU would have run
one at a time and slightly slower each.

`device` therefore defaults to `cuda` when a card is present - it is where a bigger network in
Phase 9 will want to be, and it costs 0.6 s here - and `sweep` overrides it to `cpu` on purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from src.features.scaling import feature_scaler

# Set before torch is imported anywhere, because the OpenMP runtime reads it once at load time
# and ignores `torch.set_num_threads` for the pool it has already built. This is what makes a
# multi-process sweep of a single-threaded fit safe rather than a lottery.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SEED = 42

#: Worker cap for `sweep`, and it is a cap rather than `-1` for a measured reason. On this
#: machine `n_jobs=-1` starts 32 loky processes that each initialise a fresh OpenMP runtime, and
#: the pool wedges: 36 processes alive, the CPU at 3%, and no progress in ten minutes. At 8
#: workers the same grid runs to completion every time. The arithmetic also says the cap costs
#: little - each fit is single-threaded and memory-bound on a 863 KB table, so the speed-up was
#: never going to be linear in cores - and `OMP_NUM_THREADS` is pinned below for the same reason.
MAX_WORKERS = 8

#: The four the plan names. `relu` is the reference point against 6.2.1's sklearn sweep; the
#: other three are the reason this module exists.
ACTIVATIONS = ("relu", "leaky_relu", "gelu", "tanh")

#: Every optimizer 6.2.4 compares, named here because the estimator has to accept them and
#: 6.2.4 should not have to keep its own copy of the list.
OPTIMIZERS = ("sgd", "momentum", "rmsprop", "adam", "adamw")

#: 6.2.5's schedules. `none` is the constant-rate control.
SCHEDULES = ("none", "step", "cosine", "one_cycle")


def resolve_device(requested: str | None = None) -> str:
    """`cuda` when there is a card and the caller did not ask for otherwise."""
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _activation(name: str):
    import torch.nn as nn

    layers = {
        "relu": nn.ReLU,
        # 0.01 is the slope from the original paper. The point of including it is that it has no
        # dead-unit region, which is the failure mode ReLU is usually blamed for.
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    if name not in layers:
        raise ValueError(f"activation must be one of {list(ACTIVATIONS)}; got {name!r}")
    return layers[name]()


def _optimizer(name: str, parameters, lr: float, weight_decay: float):
    import torch

    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        # The whole point of AdamW is that the decay is *not* folded into the gradient, so
        # passing the same `weight_decay` to it and to Adam applies two different penalties -
        # which is exactly what 6.2.3 is set up to measure rather than assume away.
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}; got {name!r}")


def _schedule(name: str, optimizer, epochs: int, steps_per_epoch: int, lr: float):
    import torch

    if name == "none":
        return None, "epoch"
    if name == "step":
        step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
        return step, "epoch"
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), "epoch"
    if name == "one_cycle":
        # One-cycle is defined per *batch*, not per epoch. Stepping it once an epoch would
        # traverse 1/steps_per_epoch of the cycle and never reach the annealing half - a bug
        # that presents as "one-cycle just underperforms" rather than as an error.
        cycle = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 10, epochs=epochs, steps_per_epoch=max(1, steps_per_epoch)
        )
        return cycle, "batch"
    raise ValueError(f"schedule must be one of {list(SCHEDULES)}; got {name!r}")


class TorchMLP(ClassifierMixin, BaseEstimator):
    """A dense network with every knob Phase 6.2 asks for, behind sklearn's estimator protocol.

    The constructor stores its arguments and does nothing else, because `get_params`/`set_params`
    - which `GridSearchCV` and `clone` depend on - require exactly that. All construction
    happens in `fit`.

    `loss_curve_`, `val_curve_` and `lr_curve_` are the artefacts 6.2.4 and 6.2.5 exist to plot:
    one entry per epoch. sklearn's `MLPClassifier` exposes only the first of the three, and only
    for some solvers.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (512, 256),
        activation: str = "relu",
        dropout: float = 0.0,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        schedule: str = "none",
        batch_size: int = 64,
        max_epochs: int = 600,
        early_stopping: bool = True,
        n_iter_no_change: int = 15,
        validation_fraction: float = 0.12,
        device: str | None = None,
        random_state: int = SEED,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.schedule = schedule
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state

    # -- construction -------------------------------------------------------------------

    def _build(self, n_features: int, n_classes: int):
        import torch.nn as nn

        sizes = [n_features, *list(self.hidden_layer_sizes)]
        layers: list = []
        for a, b in zip(sizes[:-1], sizes[1:], strict=True):
            layers.append(nn.Linear(a, b))
            layers.append(_activation(self.activation))
            if self.dropout > 0:
                # After the activation, not before: dropping the pre-activations of a ReLU
                # zeroes units that were already going to be zero, and the effective rate is
                # then not the rate that was asked for.
                layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(sizes[-1], n_classes))
        net = nn.Sequential(*layers)
        for module in net:
            if isinstance(module, nn.Linear):
                # Glorot-uniform, which is what sklearn's MLPClassifier uses - so 6.2.1's
                # numbers and this module's are separated by the training loop and not by the
                # initialisation.
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        return net

    def parameter_count(self) -> int:
        from src.classify.mlp import parameter_count

        return parameter_count(
            int(self.n_features_in_), tuple(self.hidden_layer_sizes), len(self.classes_)
        )

    # -- fit ----------------------------------------------------------------------------

    def fit(self, X, y):  # noqa: N803 - sklearn's parameter name
        import torch
        import torch.nn as nn
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        self.encoder_ = LabelEncoder().fit(np.asarray(y).astype(str))
        self.classes_ = self.encoder_.classes_
        self.n_features_in_ = X.shape[1]
        encoded = self.encoder_.transform(np.asarray(y).astype(str))

        self.device_ = resolve_device(self.device)
        device = self.device_

        # The hold-out is carved whenever `validation_fraction` is set, *independently* of
        # `early_stopping`. That is deliberate and it is a correctness fix: if the un-stopped arm
        # trained on 100% while the stopped arm trained on 88%, then "early stopping costs
        # 0.021" would really be "12% more training data is worth 0.021" on a corpus 5.2.9
        # measured as data-limited. Both arms now see exactly the same rows, and the only
        # difference between them is whether the run is cut short and the best weights restored.
        if self.validation_fraction > 0:
            counts = np.bincount(encoded)
            # A stratified hold-out needs two rows of every class, and the 40-row circuit class
            # at a 12% split inside a 4/5 training fold is close enough to that floor that the
            # guard is not theoretical. An unstratified fallback beats crashing a sweep at fold
            # three.
            stratify = encoded if counts.min() >= 2 else None
            train_idx, val_idx = train_test_split(
                np.arange(len(encoded)),
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
        else:
            train_idx = np.arange(len(encoded))
            val_idx = np.array([], dtype=int)

        Xt = torch.as_tensor(X[train_idx], device=device)
        yt = torch.as_tensor(encoded[train_idx], dtype=torch.long, device=device)
        has_val = len(val_idx) > 0
        if has_val:
            Xv = torch.as_tensor(X[val_idx], device=device)
            yv_true = self.encoder_.inverse_transform(encoded[val_idx])

        net = self._build(X.shape[1], len(self.classes_)).to(device)
        criterion = nn.CrossEntropyLoss()
        opt = _optimizer(self.optimizer, net.parameters(), self.learning_rate, self.weight_decay)

        n_train = len(train_idx)
        batch = min(self.batch_size, n_train) if self.batch_size else n_train
        steps = int(np.ceil(n_train / batch))
        sched, sched_unit = _schedule(
            self.schedule, opt, self.max_epochs, steps, self.learning_rate
        )

        generator = torch.Generator(device="cpu").manual_seed(self.random_state)
        self.loss_curve_, self.val_curve_, self.lr_curve_ = [], [], []
        best_score, best_state, bad_epochs = -np.inf, None, 0

        for _epoch in range(self.max_epochs):
            net.train()
            order = torch.randperm(n_train, generator=generator).to(device)
            total = 0.0
            for start in range(0, n_train, batch):
                idx = order[start : start + batch]
                opt.zero_grad(set_to_none=True)
                loss = criterion(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                if sched is not None and sched_unit == "batch":
                    sched.step()
                total += float(loss.item()) * len(idx)
            self.lr_curve_.append(float(opt.param_groups[0]["lr"]))
            if sched is not None and sched_unit == "epoch":
                sched.step()
            self.loss_curve_.append(total / n_train)

            if not has_val:
                continue
            net.eval()
            with torch.no_grad():
                predicted = net(Xv).argmax(1).cpu().numpy()
            score = f1_score(
                yv_true,
                self.encoder_.inverse_transform(predicted),
                average="macro",
                zero_division=0,
            )
            self.val_curve_.append(float(score))

            if not self.early_stopping:
                continue
            if score > best_score + 1e-6:
                best_score, bad_epochs = score, 0
                # Detached and cloned, so the restore below is not reading tensors that later
                # epochs have since mutated in place.
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.n_iter_no_change:
                    break

        if best_state is not None:
            # Early stopping that keeps the *last* weights rather than the best ones is not
            # early stopping, it is a shorter run - and the difference between the two is one
            # of the things 6.2.3 measures.
            net.load_state_dict(best_state)

        self.net_ = net
        self.epochs_run_ = len(self.loss_curve_)
        self.best_validation_score_ = float(best_score) if best_score > -np.inf else None
        return self

    # -- predict ------------------------------------------------------------------------

    def _logits(self, X):  # noqa: N803
        import torch

        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            return self.net_(torch.as_tensor(X, device=self.device_)).cpu().numpy()

    def predict(self, X):  # noqa: N803
        return self.encoder_.inverse_transform(self._logits(X).argmax(1))

    def predict_proba(self, X):  # noqa: N803
        logits = self._logits(X)
        # Softmax with the row max subtracted. The raw logits of a converged 5-class network on
        # an easy page reach 20-plus, and the shift is what keeps `exp` in range regardless.
        shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)

    def training_score(self, X, y) -> float:  # noqa: N803
        """Macro F1 on the rows this estimator was fitted on - 6.2.3's overfit gap needs it."""
        from sklearn.metrics import f1_score

        return float(f1_score(y, self.predict(X), average="macro", zero_division=0))


def pipeline(**kwargs) -> Pipeline:
    """4.2.3's imputation and 4.2.4's scaling, then the network - 6.2.1's pipeline, new engine."""
    return Pipeline([("prepare", feature_scaler()), ("model", TorchMLP(**kwargs))])


def cross_validate(data, folds: int = 5, seed: int = SEED, **kwargs) -> dict:
    """One configuration, scored under 5.2.1's fold protocol - the harness 6.2.2-6.2.6 share.

    Deliberately not `cross_val_predict(n_jobs=-1)`. That would fork 32 processes, each of which
    would build its own CUDA context on one card, and the result is slower than running the five
    fits in sequence as well as being a way to exhaust device memory for no gain. The
    parallelism in this phase belongs one level up, across configurations, not across folds.

    Returns the fold scores *and* the training scores, because 6.2.3's whole subject is the
    distance between the two, and refitting to get it later would be a different network.
    """
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    estimator = pipeline(**kwargs)
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)

    started = time.perf_counter()
    predicted = np.empty(len(data.y), dtype=object)
    train_scores, epochs, curves = [], [], []
    for train_idx, test_idx in splitter.split(data.X, data.y):
        fitted = clone(estimator).fit(data.X[train_idx], data.y[train_idx])
        predicted[test_idx] = fitted.predict(data.X[test_idx])
        model = fitted.named_steps["model"]
        # The scaler is part of the pipeline, so the training score goes through `fitted` rather
        # than through `model` - scoring the network on unscaled columns would read as a
        # catastrophic overfit gap that is really a missing transform.
        train_scores.append(
            f1_score(
                data.y[train_idx],
                fitted.predict(data.X[train_idx]),
                average="macro",
                zero_division=0,
            )
        )
        epochs.append(model.epochs_run_)
        curves.append(
            {
                "loss": list(model.loss_curve_),
                "val": list(model.val_curve_),
                "lr": list(model.lr_curve_),
            }
        )

    per_fold = [
        f1_score(data.y[test], predicted[test], average="macro", zero_division=0)
        for _, test in splitter.split(data.X, data.y)
    ]
    return {
        "macro_f1": round(float(np.mean(per_fold)), 4),
        "std": round(float(np.std(per_fold)), 4),
        "train_macro_f1": round(float(np.mean(train_scores)), 4),
        "overfit_gap": round(float(np.mean(train_scores) - np.mean(per_fold)), 4),
        "epochs": round(float(np.mean(epochs)), 1),
        "seconds": round(time.perf_counter() - started, 1),
        "predicted": predicted,
        "curves": curves,
    }


def sweep(data, configs: list[dict], n_jobs: int | None = None, folds: int = 5) -> list[dict]:
    """Many configurations, each 5-fold cross-validated - the outer loop of 6.2.3 through 6.2.6.

    The parallelism is here rather than inside `cross_validate`, and it is on the CPU, and both
    of those are measured choices rather than habits. See the module docstring: one 5-fold CV of
    a (512, 256) network on the hybrid table takes 3.4 s on the card and 2.8 s on one core, so
    there is nothing for the GPU to win at this scale - while 32 cores running 32 *different*
    configurations at once is a real 32x that a single card cannot offer.

    Each worker sets `torch.set_num_threads(1)`: the default is to grab every core per process,
    and 32 processes each doing that is 1,024 threads fighting over 32 cores.
    """
    from joblib import Parallel, delayed

    def one(config: dict) -> dict:
        import torch

        # One thread per worker. The default is for each process to claim every core, and N
        # processes each doing that is N x 32 threads fighting over 32 cores.
        torch.set_num_threads(1)
        result = cross_validate(data, folds=folds, device="cpu", **config)
        return {**config, **{k: v for k, v in result.items() if k != "predicted"}}

    workers = MAX_WORKERS if n_jobs is None else n_jobs
    return list(
        Parallel(n_jobs=workers, backend="loky")(delayed(one)(config) for config in configs)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=["handcrafted", "embedding", "hybrid"])
    ap.add_argument("--activation", default="relu", choices=list(ACTIVATIONS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    from src.embed.hybrid import dataset

    data = dataset(args.table, "real")
    estimator = pipeline(activation=args.activation, device=args.device).fit(data.X, data.y)
    model = estimator.named_steps["model"]
    print(
        json.dumps(
            {
                "table": args.table,
                "activation": args.activation,
                "device": model.device_,
                "parameters": model.parameter_count(),
                "epochs_run": model.epochs_run_,
                "first_loss": round(model.loss_curve_[0], 4),
                "final_loss": round(model.loss_curve_[-1], 4),
                "best_validation_macro_f1": round(model.best_validation_score_, 4),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
