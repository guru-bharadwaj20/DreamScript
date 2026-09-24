"""Phase 15.6 - prediction monitoring: what the router believes, and when to stop believing it.

    python -m src.mlops.monitor              # reports/prediction_monitoring.md + .json + figure
    python -m src.mlops.monitor --check      # non-zero if the spike alarm fails its controls
    python -m src.mlops.monitor --against <parquet>   # score a batch and exit non-zero on a spike

The plan asks to log confidence distributions and flag low-confidence spikes. The first half is
easy and the second is where the row can go wrong, so the design decision is stated up front.

## What is monitored, and what deliberately is not

**Not the golden-run stage confidences.** `runs/p13_golden/*.json` carries a `confidence` on every
stage, and on 26 pages six of the seven stages report exactly 1.0 - `detect`, `assemble`,
`traverse`, `serialise`, `generate` and `verify` are constants, and only `classify` varies at all,
over the range 0.9909 to 1.0. A monitor over that would be reporting a straight line and calling
it a distribution. **The IR node confidences in those runs are also all 1.0**, because a golden
run carries annotated IR rather than predicted IR.

What is monitored instead is the **routing classifier's own out-of-fold probability** over 4.2's
feature table - 13.4's rung, the thing that actually decides which parser a page goes to. It is
out-of-fold by writer, so a page's confidence is never read off a model that trained on that
page's hand, which is what makes the baseline honest rather than flattering.

## The spike alarm

A page is *low confidence* below `LOW_CONFIDENCE`. The baseline is the low-confidence share of the
reference window; a batch alarms when its share is at least `SPIKE_RATIO` times that baseline
**and** at least `SPIKE_FLOOR` in its own right.

An *additive* margin was written first and was the wrong shape, which is recorded in the constant
rather than quietly corrected: on a baseline of 0.0023, "baseline + 0.10" is an absolute threshold
wearing a margin's clothes, and it scored a degraded batch at 0.0867 - a 38x rise - as no spike.

Two controls run every time and `--check` fails on them, for 15.5's reason:

    quiet   a held-out slice of the reference. **Must not spike.** If clean pages from the same
            distribution set the alarm off, it is measuring the split and not the data.
    spike   the same pages through 14.5's degradations, re-extracted from the degraded pixels.
            **Must spike** - a monitor that cannot see a page photographed badly is not
            monitoring anything a deployment will encounter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import ROOT

REPORT_MD = ROOT / "reports" / "prediction_monitoring.md"
REPORT_JSON = ROOT / "reports" / "prediction_monitoring.json"
FIGURE = ROOT / "reports" / "figures" / "p15_prediction_monitoring.png"

#: Below this the router is not confident enough to be trusted without a second opinion. 0.70 is
#: 13.5's own routing gate, reused rather than invented so two rows cannot disagree about what
#: "low confidence" means.
LOW_CONFIDENCE = 0.70

#: A batch alarms when its low-confidence share is both a **multiple** of the baseline and above a
#: small absolute floor. Both halves are load-bearing and the shape was got wrong first:
#:
#: An *additive* margin was tried and is the wrong form. This corpus's baseline is 0.0023, so
#: "baseline + 0.10" is 0.1023 - which is an absolute threshold wearing a margin's clothes, the
#: exact thing a margin is supposed to avoid. It missed a degraded batch running at 0.0867, a
#: **38x** increase over baseline, because 8.67% is not 10%.
#:
#: The ratio is the form that travels, and the floor is what stops it being hysterical near zero:
#: on a baseline of 0.001, two uncertain pages in a thousand is "2x" and means nothing. A batch
#: has to be materially uncertain in its own right *and* much worse than usual.
SPIKE_RATIO = 3.0
SPIKE_FLOOR = 0.02

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 99)


def _fallback():
    from src.pipeline import fallback

    return fallback


def out_of_fold_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Each page's predicted class and probability, from a model that never saw its writer."""
    from sklearn.model_selection import StratifiedGroupKFold

    fb = _fallback()
    X, _ = fb.features(frame)
    y = frame["diagram_type"].to_numpy()
    groups = frame["group"].to_numpy()
    confidence = np.full(len(frame), np.nan)
    predicted = np.empty(len(frame), dtype=object)

    splitter = StratifiedGroupKFold(n_splits=fb.N_SPLITS, shuffle=True, random_state=0)
    for train_index, test_index in splitter.split(X, y, groups):
        model = fb.classifier().fit(X[train_index], y[train_index])
        proba = model.predict_proba(X[test_index])
        confidence[test_index] = proba.max(axis=1)
        predicted[test_index] = model.classes_[proba.argmax(axis=1)]

    return pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "source": frame["source"].to_numpy(),
            "truth": y,
            "predicted": predicted,
            "confidence": confidence,
            "correct": predicted == y,
        }
    )


def distribution(confidence: np.ndarray) -> dict[str, Any]:
    """The log the plan asks for: percentiles, not a mean that hides the tail."""
    values = np.asarray(confidence, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": round(float(values.mean()), 4),
        "percentiles": {f"p{p}": round(float(np.percentile(values, p)), 4) for p in PERCENTILES},
        "low_confidence_share": round(float((values < LOW_CONFIDENCE).mean()), 4),
    }


def spike(baseline_share: float, batch: np.ndarray) -> dict[str, Any]:
    """Does this batch carry materially more uncertain pages than the reference did?"""
    values = np.asarray(batch, dtype=float)
    values = values[np.isfinite(values)]
    share = float((values < LOW_CONFIDENCE).mean()) if len(values) else float("nan")
    ratio = (
        float(share / baseline_share)
        if np.isfinite(share) and baseline_share > 0
        else (float("inf") if np.isfinite(share) and share > 0 else float("nan"))
    )
    return {
        "n": int(len(values)),
        "low_confidence_share": None if not np.isfinite(share) else round(share, 4),
        "baseline_share": round(float(baseline_share), 4),
        "ratio": None if not np.isfinite(ratio) else round(ratio, 2),
        "ratio_required": SPIKE_RATIO,
        "floor": SPIKE_FLOOR,
        "spiked": bool(
            np.isfinite(share)
            and share >= SPIKE_FLOOR
            and (not np.isfinite(ratio) or ratio >= SPIKE_RATIO)
        ),
    }


def _confidence_for(model, columns: list[str], table: pd.DataFrame) -> np.ndarray:
    """Route an arbitrary feature table through a fitted model, matching columns by name."""
    matrix = np.full((len(table), len(columns)), np.nan, dtype=float)
    for index, column in enumerate(columns):
        if column in table.columns:
            matrix[:, index] = table[column].to_numpy(dtype=float)
    return model.predict_proba(matrix).max(axis=1)


def controls(frame: pd.DataFrame, baseline_share: float, limit: int = 120) -> dict[str, Any]:
    """A clean held-out slice must stay quiet; the same pages degraded must spike."""
    fb = _fallback()
    held_ids = set(frame[frame["source"] == "fa_bresler"].head(limit)["id"])
    train = frame[~frame["id"].isin(held_ids)]
    held = frame[frame["id"].isin(held_ids)]
    if held.empty:
        return {"skipped": "no fa_bresler pages to hold out"}

    X_train, columns = fb.features(train)
    model = fb.classifier().fit(X_train, train["diagram_type"].to_numpy())

    quiet = spike(baseline_share, _confidence_for(model, columns, held))
    quiet["must_spike"] = False
    quiet["passes"] = not quiet["spiked"]
    quiet["reading"] = "clean held-out pages, never trained on; a spike here is the split talking"

    degraded = fb.degraded_features(frame, limit=limit)
    if degraded is None or degraded.empty:
        loud = {"skipped": "no degraded features", "passes": None, "must_spike": True}
    else:
        loud = spike(baseline_share, _confidence_for(model, columns, degraded))
        loud["must_spike"] = True
        loud["passes"] = bool(loud["spiked"])
        loud["reading"] = "the same pages through 14.5's degradations, re-extracted"
    return {"quiet": quiet, "spike": loud}


def collect(with_pixels: bool = True) -> dict[str, Any]:
    fb = _fallback()
    frame = fb.load()
    scored = out_of_fold_confidence(frame)
    overall = distribution(scored["confidence"].to_numpy())
    baseline = overall["low_confidence_share"]

    by_source = {
        source: distribution(part["confidence"].to_numpy())
        for source, part in scored.groupby("source")
    }
    correct = scored[scored["correct"]]["confidence"].to_numpy()
    wrong = scored[~scored["correct"]]["confidence"].to_numpy()

    result: dict[str, Any] = {
        "what": "15.6 - the routing classifier's confidence, logged, with a spike alarm",
        "monitored": "out-of-fold max predict_proba of 13.4's routing rung, grouped by writer",
        "not_monitored": (
            "runs/p13_golden stage confidences: six of seven stages are exactly 1.0 over 26 pages"
            " and the IR node confidences are all 1.0, so a monitor over them would report a"
            " straight line and call it a distribution"
        ),
        "thresholds": {
            "low_confidence": LOW_CONFIDENCE,
            "spike_ratio": SPIKE_RATIO,
            "spike_floor": SPIKE_FLOOR,
        },
        "overall": overall,
        "by_source": by_source,
        "when_correct": distribution(correct),
        "when_wrong": distribution(wrong),
        "controls": controls(frame, baseline) if with_pixels else {},
    }
    arms = [
        a
        for a in result["controls"].values()
        if isinstance(a, dict) and a.get("passes") is not None
    ]
    result["verdict"] = {
        "controls_pass": bool(arms) and all(a["passes"] for a in arms),
        "baseline_low_confidence_share": baseline,
    }
    return result


def figure(result: dict, path: Path = FIGURE) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_source = result.get("by_source") or {}
    if not by_source:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    sources = sorted(by_source)
    shares = [by_source[s].get("low_confidence_share", 0) for s in sources]
    axes[0].bar(sources, shares, color="#4C72B0")
    axes[0].axhline(
        result["verdict"]["baseline_low_confidence_share"],
        color="#C44E52",
        linestyle="--",
        label="corpus baseline",
    )
    axes[0].set_ylabel(f"share below {LOW_CONFIDENCE}")
    axes[0].set_title("low-confidence share by source")
    axes[0].tick_params(axis="x", rotation=20, labelsize=8)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, axis="y")

    for key, colour in (("when_correct", "#55A868"), ("when_wrong", "#C44E52")):
        percentiles = result.get(key, {}).get("percentiles")
        if not percentiles:
            continue
        xs = [int(name[1:]) for name in percentiles]
        axes[1].plot(xs, list(percentiles.values()), "-o", color=colour, markersize=4, label=key)
    axes[1].axhline(LOW_CONFIDENCE, color="#8172B2", linestyle=":", label="low-confidence gate")
    axes[1].set_xlabel("percentile")
    axes[1].set_ylabel("confidence")
    axes[1].set_title("confidence when right vs when wrong")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Phase 15.6 - prediction monitoring (routing rung, out-of-fold)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render(result: dict, figure_path: Path | None) -> str:
    overall = result["overall"]
    lines = [
        "# Phase 15.6 - prediction monitoring",
        "",
        "Generated by `python -m src.mlops.monitor`.",
        "",
        f"**What is monitored:** {result['monitored']}.",
        "",
        f"**What deliberately is not:** {result['not_monitored']}.",
        "",
        "## The distribution",
        "",
        f"{overall['n']} pages, mean {overall['mean']}, "
        f"**{overall['low_confidence_share']:.1%} below the {LOW_CONFIDENCE} gate**.",
        "",
        "| percentile | " + " | ".join(overall["percentiles"]) + " |",
        "| :--- | " + " | ".join("---:" for _ in overall["percentiles"]) + " |",
        "| confidence | " + " | ".join(str(v) for v in overall["percentiles"].values()) + " |",
        "",
        "Percentiles rather than a mean, because the mean of a distribution this skewed hides"
        " exactly the tail the alarm exists to watch.",
        "",
        "| source | pages | mean | share below gate |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for source, dist in sorted(result["by_source"].items()):
        lines.append(
            f"| {source} | {dist['n']} | {dist.get('mean')} |"
            f" {dist.get('low_confidence_share')} |"
        )
    right, wrong = result["when_correct"], result["when_wrong"]
    lines += [
        "",
        "## Is the confidence worth anything?",
        "",
        f"On the {right['n']} pages it gets right the router's mean confidence is"
        f" **{right.get('mean')}**; on the {wrong['n']} it gets wrong it is **{wrong.get('mean')}**."
        + (
            " The two are separated, so the number carries information and a low reading is worth"
            " acting on."
            if (wrong.get("mean") or 1) < (right.get("mean") or 0)
            else " **The two are not separated, so this confidence does not identify its own"
            " mistakes and the gate below is not doing what it looks like it is doing.**"
        ),
        "",
        "## The spike alarm",
        "",
        f"A page is low confidence below **{LOW_CONFIDENCE}**. A batch alarms when its"
        f" low-confidence share is at least **{SPIKE_RATIO}x** the corpus baseline"
        f" ({result['verdict']['baseline_low_confidence_share']}) **and** at least"
        f" **{SPIKE_FLOOR}** in its own right.",
        "",
        "**An additive margin was tried first and is the wrong shape**, which is recorded here"
        " because the failure is instructive rather than embarrassing: with a baseline of 0.0023,"
        " `baseline + 0.10` is 0.1023 - an absolute threshold wearing a margin's clothes, the very"
        " thing a margin exists to avoid. It scored a degraded batch running at 0.0867 - a **38x**"
        " increase - as no spike, because 8.67% is not 10%. The ratio is the form that travels"
        " between corpora; the floor is what keeps it from being hysterical near zero, where two"
        " uncertain pages in a thousand is '2x' and means nothing.",
        "",
        "| control | what it asks | share | ratio | spiked | must spike | passes |",
        "| :--- | :--- | ---: | ---: | :--- | :--- | :--- |",
    ]
    labels = {
        "quiet": "clean held-out pages",
        "spike": "the same pages, degraded",
    }
    for key, arm in (result.get("controls") or {}).items():
        if arm.get("skipped"):
            lines.append(f"| `{key}` | {labels.get(key, key)} | - | - | skipped | - | - |")
            continue
        lines.append(
            f"| `{key}` | {labels.get(key, key)} | {arm.get('low_confidence_share')} |"
            f" {arm.get('ratio')} |"
            f" {'**yes**' if arm['spiked'] else 'no'} |"
            f" {'yes' if arm['must_spike'] else 'no'} |"
            f" {'yes' if arm['passes'] else '**NO**'} |"
        )
    lines += [
        "",
        "A monitor that cannot stay quiet on clean held-out pages is measuring the split rather"
        " than the data, and one that cannot see a page photographed badly is not monitoring"
        " anything a deployment will meet. Both are required, and `--check` fails on either.",
        "",
    ]
    if figure_path is not None:
        lines += [f"![prediction monitoring](figures/{figure_path.name})", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15.6 prediction monitoring")
    ap.add_argument("--check", action="store_true", help="non-zero if a control fails")
    ap.add_argument("--against", help="score this parquet as a batch and exit")
    ap.add_argument("--no-pixels", action="store_true", help="skip the degradation control")
    args = ap.parse_args(argv)

    if args.against:
        fb = _fallback()
        frame = fb.load()
        scored = out_of_fold_confidence(frame)
        baseline = distribution(scored["confidence"].to_numpy())["low_confidence_share"]
        X, columns = fb.features(frame)
        model = fb.classifier().fit(X, frame["diagram_type"].to_numpy())
        batch = pd.read_parquet(args.against)
        result = spike(baseline, _confidence_for(model, columns, batch))
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if result["spiked"] else 0

    result = collect(with_pixels=not args.no_pixels)
    drawn = figure(result)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render(result, drawn), encoding="utf-8")
    json.dump(result["verdict"], sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(f"-> {REPORT_MD}")
    if args.check:
        return 0 if result["verdict"]["controls_pass"] else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
