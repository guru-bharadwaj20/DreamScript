"""Phase 15.5 - drift detection: has the input distribution moved away from what was trained on?

    python -m src.mlops.drift                 # reports/drift.md + .json
    python -m src.mlops.drift --check         # non-zero if the detector fails its own controls
    python -m src.mlops.drift --against <parquet>   # score a new batch against the reference

The plan asks for PSI / KS over input feature distributions, and names the three shifts it cares
about: new drawing styles, new diagram types, new capture devices. Those are not hypotheticals
here - this corpus contains all three, so the detector is calibrated against real shifts rather
than against Gaussian noise someone injected to make a threshold look principled.

## What is measured

The reference window is the **train** split of 4.2's handcrafted table; a candidate batch is
anything with the same columns. For each feature:

    PSI   sum over bins of (current_share - reference_share) * ln(current_share / reference_share)
          binned on the reference's own deciles, so the bins are a property of what was trained on
    KS    the two-sample Kolmogorov-Smirnov statistic and its p-value

PSI's conventional reading is the one used here and it is stated rather than tuned: **< 0.10 no
material shift, 0.10-0.25 moderate, > 0.25 significant**. The alarm fires on the significant band.

## The controls are the point

An alarm that fires is worthless until you know it does not fire on nothing. Three arms run every
time, and `--check` fails on the first two rather than on the headline:

    null        the reference split in half at random, scored against itself. **Must not alarm.**
                If it does, the detector is reporting sampling noise and every other number it
                produces is uninterpretable.
    self        the reference against itself, whole. PSI must be ~0 by construction; this catches
                a binning bug that would make even an identical distribution look shifted.
    shifted     the three real shifts this corpus contains, each of which **must** alarm:
                  new diagram type     train without state_machine, score fa_bresler
                  new capture device   train on rendered, score photographed (13.4's domains)
                  new drawing style    14.5's degradations, re-extracted from degraded pixels

A detector that passes `null` and fails to alarm on `shifted` is blind; one that alarms on `null`
is crying wolf. Both are reported, and neither is inferred from the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.features.build import IDENTITY
from src.utils.config import ROOT

TABLE = ROOT / "data" / "features" / "handcrafted.parquet"
REPORT_MD = ROOT / "reports" / "drift.md"
REPORT_JSON = ROOT / "reports" / "drift.json"

#: Columns that name a row rather than describe a page - 4.1.5's leak, and meaningless to drift.
#: Imported above, from the module that writes the table (4.2.2), rather than restated here -
#: it was one of four copies, and a column added to three of them is a silent leak in the fourth.

#: PSI's conventional bands. Stated, not tuned: a threshold chosen to make this corpus pass would
#: measure the corpus rather than the drift.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

#: A single feature moving is not a distribution moving. The alarm needs this many features in the
#: significant band, which is what keeps one noisy column from firing it on its own.
MIN_DRIFTED_FEATURES = 3

#: Deciles. Fewer bins hide a shift in the tail; more bins on a few thousand rows put single
#: pages in their own bin and make PSI a function of the sample size.
BINS = 10

#: PSI is undefined where a bin is empty on either side. The conventional repair is a floor, and
#: it has to be small enough not to invent a shift and large enough to keep the log finite.
EPSILON = 1e-6


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in IDENTITY and not c.startswith("_")]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = BINS) -> float:
    """Population stability index, binned on the reference's own quantiles.

    Binning on the reference matters: quantiles of the *combined* sample move when the candidate
    moves, which drags the bin edges toward the shift and understates it. The reference is what
    was trained on, so it is what defines the bins.
    """
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # A constant or near-constant feature has no distribution to shift; saying so is better
        # than returning a large number from a degenerate binning.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_share = np.histogram(current, bins=edges)[0] / len(current)
    ref_share = np.clip(ref_share, EPSILON, None)
    cur_share = np.clip(cur_share, EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ks_2samp

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 2 or len(current) < 2:
        return float("nan"), float("nan")
    result = ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def band(value: float) -> str:
    if not np.isfinite(value):
        return "undefined"
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "none"


def compare(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str] | None = None
) -> dict[str, Any]:
    """PSI and KS per feature, plus the alarm the whole thing exists to raise."""
    columns = columns or feature_columns(reference)
    rows = []
    for column in columns:
        if column not in current.columns:
            continue
        ref = reference[column].to_numpy(dtype=float)
        cur = current[column].to_numpy(dtype=float)
        value = psi(ref, cur)
        statistic, pvalue = ks(ref, cur)
        rows.append(
            {
                "feature": column,
                "psi": None if not np.isfinite(value) else round(value, 4),
                "band": band(value),
                "ks_statistic": None if not np.isfinite(statistic) else round(statistic, 4),
                "ks_pvalue": None if not np.isfinite(pvalue) else float(f"{pvalue:.3g}"),
            }
        )
    drifted = [r for r in rows if r["band"] == "significant"]
    moderate = [r for r in rows if r["band"] == "moderate"]
    rows.sort(key=lambda r: (r["psi"] is None, -(r["psi"] or 0)))
    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": len(rows),
        "drifted": len(drifted),
        "moderate": len(moderate),
        "alarm": len(drifted) >= MIN_DRIFTED_FEATURES,
        "worst": rows[:8],
        "criterion": (
            f"alarm when at least {MIN_DRIFTED_FEATURES} features reach PSI >= {PSI_SIGNIFICANT}"
        ),
    }


def load() -> pd.DataFrame:
    if not TABLE.is_file():
        raise FileNotFoundError(f"no feature table at {TABLE}; run python -m src.features.build")
    return pd.read_parquet(TABLE)


def reference_window(frame: pd.DataFrame) -> pd.DataFrame:
    """What the models were trained on: the train split, real pages only."""
    real = frame[~frame["synthetic"].astype(bool)] if "synthetic" in frame else frame
    train = real[real["split"] == "train"]
    return train if len(train) else real


def arm_null(frame: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """The reference split in half at random. Must not alarm."""
    reference = reference_window(frame)
    shuffled = reference.sample(frac=1.0, random_state=seed)
    half = len(shuffled) // 2
    result = compare(shuffled.iloc[:half], shuffled.iloc[half:])
    result["must_alarm"] = False
    result["passes"] = result["alarm"] is False
    result["reading"] = (
        "two random halves of the same distribution; an alarm here is sampling noise and would"
        " make every other number in this report uninterpretable"
    )
    return result


def arm_self(frame: pd.DataFrame) -> dict[str, Any]:
    """The reference against itself. PSI must be ~0 or the binning is wrong."""
    reference = reference_window(frame)
    result = compare(reference, reference)
    worst = max((r["psi"] or 0) for r in result["worst"]) if result["worst"] else 0.0
    result["must_alarm"] = False
    result["passes"] = result["alarm"] is False and worst < 1e-9
    result["max_psi"] = worst
    result["reading"] = "identical inputs; PSI is 0 by construction and anything else is a bug"
    return result


def arm_new_diagram_type(frame: pd.DataFrame) -> dict[str, Any]:
    """A type the reference has never seen - the plan's 'new diagram types'."""
    real = frame[~frame["synthetic"].astype(bool)] if "synthetic" in frame else frame
    reference = real[real["diagram_type"] != "state_machine"]
    current = real[real["diagram_type"] == "state_machine"]
    result = compare(reference, current)
    result["must_alarm"] = True
    result["passes"] = bool(result["alarm"])
    result["reading"] = "fa_bresler state machines against a reference that contains none"
    return result


def arm_new_capture_device(frame: pd.DataFrame) -> dict[str, Any]:
    """Rendered against photographed - the plan's 'new capture devices', and 13.4's domains."""
    from src.pipeline.fallback import DOMAIN

    real = frame[~frame["synthetic"].astype(bool)] if "synthetic" in frame else frame
    domain = real["source"].map(DOMAIN)
    reference = real[domain == "rendered"]
    current = real[domain == "photographed"]
    result = compare(reference, current)
    result["must_alarm"] = True
    result["passes"] = bool(result["alarm"])
    result["reading"] = "photographs of paper against a reference of rendered pages"
    return result


def arm_new_drawing_style(frame: pd.DataFrame, limit: int = 120) -> dict[str, Any]:
    """14.5's degradations, re-extracted from the degraded pixels - 'new drawing styles'.

    This is the only arm that needs pixels rather than the stored table, and it is worth the cost:
    the other two shifts are between corpora, and this one is the same pages photographed worse,
    which is what a deployment actually sees when someone's phone camera changes.
    """
    from src.pipeline.fallback import degraded_features

    reference = reference_window(frame)
    current = degraded_features(frame, limit=limit)
    if current is None or current.empty:
        return {"skipped": "no degraded features could be extracted", "passes": None}
    result = compare(reference, current, columns=feature_columns(reference))
    result["must_alarm"] = True
    result["passes"] = bool(result["alarm"])
    result["reading"] = "the same pages put through 14.5's degradations and re-extracted"
    return result


def collect(with_pixels: bool = True) -> dict[str, Any]:
    frame = load()
    arms: dict[str, Any] = {
        "null": arm_null(frame),
        "self": arm_self(frame),
        "new_diagram_type": arm_new_diagram_type(frame),
        "new_capture_device": arm_new_capture_device(frame),
    }
    if with_pixels:
        arms["new_drawing_style"] = arm_new_drawing_style(frame)
    controls = [arms["null"], arms["self"]]
    shifts = [
        a for k, a in arms.items() if k not in ("null", "self") and a.get("passes") is not None
    ]
    return {
        "what": "15.5 - PSI / KS drift detection, with the controls that make an alarm mean something",
        "table": {"path": "data/features/handcrafted.parquet", "rows": int(len(frame))},
        "thresholds": {
            "psi_moderate": PSI_MODERATE,
            "psi_significant": PSI_SIGNIFICANT,
            "min_drifted_features": MIN_DRIFTED_FEATURES,
            "bins": BINS,
        },
        "arms": arms,
        "verdict": {
            "controls_pass": all(a["passes"] for a in controls),
            "every_shift_detected": all(a["passes"] for a in shifts),
            "usable": all(a["passes"] for a in controls) and all(a["passes"] for a in shifts),
        },
    }


def render(result: dict) -> str:
    arms = result["arms"]
    lines = [
        "# Phase 15.5 - drift detection",
        "",
        "Generated by `python -m src.mlops.drift`.",
        "",
        f"PSI and two-sample KS over the {result['table']['rows']} rows of 4.2's handcrafted"
        f" feature table. PSI is binned on the **reference's own deciles**, because quantiles of"
        f" the combined sample move when the candidate moves and would understate the shift.",
        "",
        f"Bands are the conventional ones, stated rather than tuned: **< {PSI_MODERATE} none,"
        f" {PSI_MODERATE}-{PSI_SIGNIFICANT} moderate, >= {PSI_SIGNIFICANT} significant**. The"
        f" alarm needs **{MIN_DRIFTED_FEATURES} features** in the significant band, so one noisy"
        " column cannot fire it alone.",
        "",
        "| arm | what it asks | features drifted | alarm | must alarm | passes |",
        "| :--- | :--- | ---: | :--- | :--- | :--- |",
    ]
    labels = {
        "null": "two random halves of the reference",
        "self": "the reference against itself",
        "new_diagram_type": "state machines against a reference with none",
        "new_capture_device": "photographs against a rendered reference",
        "new_drawing_style": "14.5's degradations, re-extracted",
    }
    for key, arm in arms.items():
        if arm.get("skipped"):
            lines.append(f"| `{key}` | {labels.get(key, key)} | - | skipped | - | - |")
            continue
        lines.append(
            f"| `{key}` | {labels.get(key, key)} | {arm['drifted']} of {arm['features']} |"
            f" {'**yes**' if arm['alarm'] else 'no'} |"
            f" {'yes' if arm['must_alarm'] else 'no'} |"
            f" {'yes' if arm['passes'] else '**NO**'} |"
        )
    verdict = result["verdict"]
    lines += [
        "",
        "**The controls are what make the alarm mean anything.** `null` scores two random halves"
        " of the same distribution and must stay quiet - an alarm there is sampling noise, and it"
        " would make every other row in this table uninterpretable. `self` scores the reference"
        " against itself, where PSI is zero by construction, and catches a binning bug that would"
        " make even identical inputs look shifted.",
        "",
        f"Controls pass: **{verdict['controls_pass']}**. Every named shift detected:"
        f" **{verdict['every_shift_detected']}**.",
        "",
    ]
    for key, arm in arms.items():
        if arm.get("skipped") or not arm.get("worst"):
            continue
        lines += [
            f"### `{key}` - {labels.get(key, key)}",
            "",
            f"{arm.get('reading', '')}",
            "",
            "| feature | PSI | band | KS | p |",
            "| :--- | ---: | :--- | ---: | ---: |",
        ]
        for row in arm["worst"][:5]:
            lines.append(
                f"| `{row['feature']}` | {row['psi']} | {row['band']} |"
                f" {row['ks_statistic']} | {row['ks_pvalue']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15.5 drift detection")
    ap.add_argument("--check", action="store_true", help="non-zero if the detector fails a control")
    ap.add_argument("--against", help="score this parquet as a candidate batch and exit")
    ap.add_argument("--no-pixels", action="store_true", help="skip the degradation arm")
    args = ap.parse_args(argv)

    if args.against:
        frame = load()
        candidate = pd.read_parquet(args.against)
        result = compare(reference_window(frame), candidate)
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if result["alarm"] else 0

    result = collect(with_pixels=not args.no_pixels)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render(result), encoding="utf-8")
    json.dump(result["verdict"], sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(f"-> {REPORT_MD}")
    if args.check:
        return 0 if result["verdict"]["usable"] else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
