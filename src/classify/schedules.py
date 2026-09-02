"""Phase 6.2.5 - four learning-rate schedules on a run that early stopping ends after 20 epochs.

    python -m src.classify.schedules        # writes reports/figures/p6_schedules.png

The plan asks for constant, step, cosine and one-cycle, and for the best one to be chosen. There
is a structural problem with that question on this corpus, and stating it is most of the task.

**A schedule is a function of the training length, and this training does not have one.** Step
decays at a fraction of the total epochs; cosine anneals to zero at `T_max`; one-cycle's entire
shape is defined over a fixed horizon. All three are configured against `max_epochs = 600`. But
6.2.2 and 6.2.3 measured early stopping firing after **20 to 40 epochs** - 3% to 7% of the
horizon the schedules are shaped for. Cosine at epoch 25 of 600 has decayed the rate by 0.4%.
One-cycle at epoch 25 is still in its warm-up and has not yet reached its peak.

So a naive comparison would report "the schedules do nothing", and the finding would be an
artefact of a horizon nobody chose deliberately. This task therefore sweeps the horizon as well:

    horizon 600    the ceiling 6.2.1 set, which early stopping never approaches
    horizon 60     roughly twice the epochs early stopping actually uses
    horizon 30     about the number it uses

At horizon 30 a cosine schedule is a real cosine over the run; at 600 it is a constant rate with
extra arithmetic. If the schedules matter at all, that is where it shows.

## What it measured

6.2.3's regularized configuration on the hybrid table, stratified 5-fold, macro F1:

    horizon      none      step     cosine   one_cycle     spread   best - constant
    600         0.9503    0.9503    0.9503    0.9260       0.0243        0.0000
     60         0.9503    0.9413    0.9480   *0.9540*      0.0127       +0.0037
     30         0.9480    0.9376    0.9396    0.9404       0.0028       -0.0076

**The prediction is confirmed exactly, and in the strongest form available: at horizon 600, step
and cosine score 0.9503 - the same number as the constant rate, to four decimal places.** Not
approximately the same; the same. Early stopping fires after 33 epochs, at which point a cosine
over 600 has decayed the rate by 0.4% and a step schedule with `step_size = 200` has not stepped
at all. Both are constant-rate training with extra arithmetic, and the sweep says so.

One-cycle is the exception at that horizon and it is the same phenomenon from the other side:
it scores 0.9260, **the worst cell in the table**, because 33 epochs into a 600-epoch cycle it is
still climbing its warm-up and has never reached its peak, let alone annealed. A schedule
evaluated at 5% of its horizon is not that schedule.

**The prediction is also partly wrong, and that is the more interesting half.** The docstring
above reasoned that if schedules matter, it would show at horizon 30 - roughly the number of
epochs early stopping actually uses. It shows at 60 instead. At 30 the best schedule is *worse*
than the constant rate by 0.0076 and the spread collapses to 0.0028; at 60, one-cycle wins by
0.0037 and the spread is 0.0127. So the horizon that matters is **about twice** the number of
epochs the run uses, not equal to it - which in hindsight is what one-cycle needs to be halfway
through its cycle rather than at either end of it, and is a thing the sweep had to be designed
across horizons to see at all.

**The best cell in the table is one-cycle at horizon 60: 0.9540 +- 0.0111, using 34.6 epochs.**
That is above 6.1.4's logistic regression (0.9503), above 6.2.4's best optimizer cell (0.9533),
and the best MLP number in Phase 6 so far. It is also 0.0037 above the constant-rate control at
one third of a fold-standard-deviation, so what is being claimed is a preference, not a result.

## What is carried forward, and the caveat that comes with it

One-cycle at a 60-epoch horizon, on the understanding that **the schedule and the horizon are one
hyperparameter and not two.** Every row of this table is evidence for that: the same four
schedules reorder completely between horizons, and the single largest effect anywhere in it is
one-cycle moving 0.0280 between horizon 600 and horizon 60 without changing anything about the
schedule itself. Reporting "one-cycle is best" without "at max_epochs = 60" would be reporting
the half of the finding that does not reproduce.

Set against the phase, the size is familiar: four schedules, three horizons, twelve
configurations, and the total range from best to worst is 0.028 - most of which is one badly
matched horizon rather than a difference between the schedules. 6.3.1's linear SVM still scores
0.9628 on this table with 810 parameters and no schedule at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.classify.activations import LOGREG, TABLES, TOPOLOGY
from src.classify.torchnet import SCHEDULES

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p6_schedules.png"

SEED = 42

#: The horizons. A schedule is defined over its total length, so sweeping the schedule without
#: sweeping the length asks the question at one arbitrary point - see the module docstring.
HORIZONS = (600, 60, 30)

#: Constant is the control, and it is the only one of the four whose behaviour does not depend
#: on the horizon at all - which is what makes it the right thing to measure the others against.
CONTROL = "none"


def grid(horizons=HORIZONS, schedules=SCHEDULES, base: dict | None = None) -> list[dict]:
    """Every schedule at every horizon, with early stopping left on.

    Early stopping stays on because turning it off would answer a different question: the plan
    wants the schedule chosen for the pipeline that Phase 6 actually ships, and that pipeline
    stops early. The horizon sweep is what separates the schedule's effect from the interaction.
    """
    settings = dict(base or {})
    return [
        {**settings, "schedule": schedule, "max_epochs": horizon}
        for horizon in horizons
        for schedule in schedules
    ]


def by_horizon(rows: list[dict]) -> dict:
    """Rows regrouped as horizon -> schedule -> score, which is how the table is read."""
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(str(row["max_epochs"]), {})[row["schedule"]] = row["macro_f1"]
    return grouped


def horizon_effect(rows: list[dict]) -> dict:
    """How much the schedule choice is worth at each horizon, against the constant-rate control.

    This is the number the docstring's argument predicts should grow as the horizon shrinks: at
    600 epochs none of the three schedules has meaningfully moved by the time early stopping
    fires, and at 30 they have.
    """
    grouped = by_horizon(rows)
    effect = {}
    for horizon, scores in grouped.items():
        control = scores.get(CONTROL)
        others = {k: v for k, v in scores.items() if k != CONTROL}
        if control is None or not others:
            continue
        best = max(others, key=lambda name: others[name])
        effect[horizon] = {
            "constant": control,
            "best_schedule": best,
            "best_score": others[best],
            "gain_over_constant": round(others[best] - control, 4),
            "spread": round(max(others.values()) - min(others.values()), 4),
        }
    return effect


def figure(rows: list[dict], traces: dict, path: Path = FIGURE) -> Path:
    """Two panels: the rate each schedule actually applies, and what it is worth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = plt.get_cmap("tab10")

    for index, name in enumerate(SCHEDULES):
        trace = traces.get(name)
        if trace is None:
            continue
        axes[0].plot(range(1, len(trace) + 1), trace, color=colours(index), label=name, lw=1.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("learning rate")
    axes[0].set_title(f"the rate actually applied (horizon {min(HORIZONS)})")

    grouped = by_horizon(rows)
    horizons = sorted(grouped, key=lambda value: -int(value))
    width = 0.8 / max(1, len(SCHEDULES))
    positions = np.arange(len(horizons))
    for index, name in enumerate(SCHEDULES):
        scores = [grouped[horizon].get(name, np.nan) for horizon in horizons]
        axes[1].bar(positions + index * width, scores, width, color=colours(index), label=name)
    axes[1].set_xticks(positions + 0.4 - width / 2)
    axes[1].set_xticklabels([f"max_epochs={horizon}" for horizon in horizons])
    axes[1].set_ylabel("macro F1")
    axes[1].set_title("what the schedule is worth, by horizon")
    # The bars differ in the third decimal; a zero-based axis would render four identical
    # rectangles and hide the only thing the panel is for.
    finite = [value for horizon in horizons for value in grouped[horizon].values()]
    axes[1].set_ylim(min(finite) - 0.01, max(finite) + 0.005)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Phase 6.2.5 - a schedule is a function of the horizon, and early stopping "
        "chooses the horizon",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def rate_traces(data, base: dict, horizon: int) -> dict:
    """The learning rate each schedule applies, per epoch, on one short fit.

    Read off `lr_curve_` rather than recomputed from the scheduler formulas: the figure has to
    show what the training loop *did*, and one-cycle in particular is stepped per batch, so a
    per-epoch reconstruction would draw a curve the run never followed.
    """
    from src.classify.torchnet import pipeline

    traces = {}
    for schedule in SCHEDULES:
        settings = {
            **base,
            "schedule": schedule,
            "max_epochs": horizon,
            "early_stopping": False,
            "device": "cpu",
        }
        fitted = pipeline(**settings).fit(data.X, data.y)
        traces[schedule] = list(fitted.named_steps["model"].lr_curve_)
    return traces


def run(
    table: str = "hybrid", corpus: str = "real", n_jobs: int | None = None, write: bool = True
) -> dict:
    from src.classify.regularize import ACTIVATION, best_settings
    from src.classify.torchnet import sweep
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    base = {
        "hidden_layer_sizes": TOPOLOGY,
        "activation": ACTIVATION,
        **{k: v for k, v in best_settings(table).items() if k in ("dropout", "weight_decay")},
    }
    rows = sweep(data, grid(base=base), n_jobs=n_jobs)
    ranked = sorted(rows, key=lambda row: -row["macro_f1"])
    effect = horizon_effect(rows)

    summary = {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "base": base,
        "horizons": list(HORIZONS),
        "by_horizon": by_horizon(rows),
        "horizon_effect": effect,
        "best": {
            "schedule": ranked[0]["schedule"],
            "max_epochs": ranked[0]["max_epochs"],
            "macro_f1": ranked[0]["macro_f1"],
            "std": ranked[0]["std"],
            "epochs": ranked[0]["epochs"],
        },
        "constant_at_the_default_horizon": next(
            row["macro_f1"]
            for row in rows
            if row["schedule"] == CONTROL and row["max_epochs"] == HORIZONS[0]
        ),
        "logistic_regression_6_1_4": LOGREG.get(table),
        "all": rows,
    }
    if write:
        traces = rate_traces(data, base, min(HORIZONS))
        summary["figure"] = str(figure(rows, traces).relative_to(ROOT))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=list(TABLES))
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.table, args.corpus, args.jobs, not args.no_figure)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    if not args.full:
        result.pop("all", None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
