"""Phase 15.8 - the retraining trigger: when is retraining the right move, and for what.

    python -m src.mlops.retrain              # reports/retraining_trigger.md + .json
    python -m src.mlops.retrain --check      # non-zero if any component is due
    python -m src.mlops.retrain --explain <component>

The plan asks for "a threshold rule for when to retrain a component, documented and scripted".
The scripting is easy. The rule is where this row can be worthless, so the reasoning is stated.

## Retraining is not the answer to most alarms

A trigger that fires on any bad signal is a trigger that recommends retraining when retraining
cannot help, and this project has already measured three cases where exactly that is true:

    S3 (OCR)    three levers were tried on the same architecture. A larger backbone bought
                -0.0231 CER; **more epochs cost +0.0089 and stronger augmentation cost +0.0040**.
                Its own oracle says perfect *selection* over the crops already in hand scores
                0.0737 against the learned ranker's 0.1890 - so the distance to the bar is in
                which crop is chosen, not in reading it. Retraining the reader is the measured
                dead end here, and a trigger that recommends it would be recommending the thing
                that already failed twice.

    12.3.3      the model is at 0.7037 against a reference ceiling of 0.784, and the row's bar is
                *beat the reference*. Retraining against an unchanged emitter cannot clear a bar
                that rises with it. The trigger therefore refuses to fire for `generation` while
                the model sits below the ceiling, and names the emitter instead.

    14.3        a perfectly detected accepting state is dropped between two stages, worth 0.9167
                pass@1 on 48 pages. No amount of retraining any component fixes a field that is
                never written.

So the rule has two halves, and the second is what makes it worth having:

    fire    the input distribution moved (15.5), or predictions got less confident (15.6), or
            enough corrections accumulated to be worth learning from (15.7) - **and** retraining
            is a lever that has not already been measured as exhausted for that component.

    refuse  and say which lever to pull instead, naming the measurement that says so.

A trigger that only ever says "retrain" is a cron job. This one is allowed to say "do not".
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.utils.config import ROOT

REPORT_MD = ROOT / "reports" / "retraining_trigger.md"
REPORT_JSON = ROOT / "reports" / "retraining_trigger.json"

#: How many usable corrections are worth a retrain on their own. Set against the corpus rather
#: than picked: S3 trains on 8,896 label crops, so 500 corrected labels is ~5.6% new signal on the
#: exact examples a user cared enough to fix. Below that the run costs more than it can move.
CORRECTION_THRESHOLD = 500

#: Components where retraining has been *measured* not to help, with the measurement that says so.
#: A trigger is only useful if it can refuse, and a refusal has to cite something.
EXHAUSTED: dict[str, dict[str, str]] = {
    "ocr": {
        "lever": "crop selection, not the reader",
        "evidence": (
            "S3: more epochs +0.0089 CER, stronger augmentation +0.0040, retrain-on-reselection"
            " +0.0067 - three retrains, all worse. The oracle over the crops already in hand is"
            " 0.0737 against the learned ranker's 0.1890, so the remaining distance is selection"
        ),
    },
    "generation": {
        "lever": "12.1.6's emitter, not the model",
        "evidence": (
            "12.3.3 grades against the reference programs, so every point the emitter gains"
            " raises the bar too; the model is at 0.7037 under a 0.784 ceiling and cannot pass by"
            " imitating an emitter that is itself failing 35 of 162"
        ),
    },
}

#: Which signal belongs to which component. Drift and confidence are both measured on the routing
#: features, so they speak for the classifier; corrections speak for whatever they corrected.
SIGNAL_OWNER = {
    "drift": "classification",
    "confidence": "classification",
    "corrections": "ocr",
}


def _safe(callable_, default):
    try:
        return callable_()
    except Exception:  # noqa: BLE001 - a missing input must not take the whole report down
        return default


def drift_signal() -> dict[str, Any]:
    """15.5, read from its report rather than recomputed, so the two cannot disagree."""
    path = ROOT / "reports" / "drift.json"
    if not path.is_file():
        return {"available": False, "why": "no reports/drift.json; run python -m src.mlops.drift"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    arms = payload.get("arms", {})
    # The controls have to have passed, or the alarms in the same report mean nothing.
    controls = payload.get("verdict", {}).get("controls_pass")
    live = {k: a for k, a in arms.items() if k not in ("null", "self") and not a.get("skipped")}
    fired = [k for k, a in live.items() if a.get("alarm")]
    return {
        "available": True,
        "controls_pass": controls,
        "shifts_detected": fired,
        # A shift against a *held-out corpus* is the detector being validated, not production
        # drift. Firing a retrain on these would be firing on the calibration set.
        "fires": False,
        "why": (
            "the shifts in reports/drift.json are this row's own calibration arms - a new diagram"
            " type, a new capture device, 14.5's degradations - not observed production traffic."
            " They prove the detector works; they are not evidence that the input has moved."
            " `--against <batch>` is what scores real traffic"
        ),
    }


def confidence_signal() -> dict[str, Any]:
    """15.6, same reasoning: the controls are calibration, not traffic."""
    path = ROOT / "reports" / "prediction_monitoring.json"
    if not path.is_file():
        return {"available": False, "why": "no reports/prediction_monitoring.json"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = payload.get("verdict", {})
    return {
        "available": True,
        "controls_pass": verdict.get("controls_pass"),
        "baseline_low_confidence_share": verdict.get("baseline_low_confidence_share"),
        "fires": False,
        "why": (
            "the spike arm is the degradation control, not production traffic; a real batch is"
            " scored with `python -m src.mlops.monitor --against <batch>`"
        ),
    }


def corrections_signal() -> dict[str, Any]:
    """15.7: enough user corrections to be worth learning from?"""
    from src.mlops import corrections

    holdings = _safe(corrections.stats, {"records": 0, "after_last_wins": 0, "stale": 0})
    usable = holdings.get("after_last_wins", 0) - holdings.get("stale", 0)
    return {
        "available": True,
        "records": holdings.get("records", 0),
        "usable": usable,
        "threshold": CORRECTION_THRESHOLD,
        "fires": usable >= CORRECTION_THRESHOLD,
        "why": (
            f"{usable} usable corrections against a threshold of {CORRECTION_THRESHOLD}"
            + (
                "; 16.2.8 is Phase 16 and not built, so no user has yet had a way to make one"
                if holdings.get("records", 0) == 0
                else ""
            )
        ),
    }


def decide(signals: dict[str, dict]) -> list[dict[str, Any]]:
    """One verdict per component: retrain, do not retrain, or nothing is asking."""
    by_component: dict[str, list[str]] = {}
    for name, signal in signals.items():
        if signal.get("fires"):
            by_component.setdefault(SIGNAL_OWNER.get(name, "unknown"), []).append(name)

    out = []
    # Every component with a recorded refusal appears even when nothing is firing, so the report
    # carries "retraining this is a dead end, pull X instead" where someone will read it - rather
    # than only at the moment a signal happens to fire.
    for component in sorted(set(SIGNAL_OWNER.values()) | set(by_component) | set(EXHAUSTED)):
        firing = by_component.get(component, [])
        exhausted = EXHAUSTED.get(component)
        if not firing:
            note = EXHAUSTED.get(component)
            out.append(
                {
                    "component": component,
                    "verdict": "no",
                    "firing_signals": [],
                    "reason": "no signal is asking for a retrain"
                    + (f"; and if one did, the lever is {note['lever']}" if note else ""),
                    **({"evidence": note["evidence"]} if note else {}),
                }
            )
        elif exhausted:
            out.append(
                {
                    "component": component,
                    "verdict": "refuse",
                    "firing_signals": firing,
                    "reason": (
                        f"a signal fired, but retraining {component} is a measured dead end -"
                        f" pull {exhausted['lever']} instead"
                    ),
                    "evidence": exhausted["evidence"],
                }
            )
        else:
            out.append(
                {
                    "component": component,
                    "verdict": "retrain",
                    "firing_signals": firing,
                    "reason": f"{', '.join(firing)} fired and no measurement rules retraining out",
                }
            )
    return out


def collect() -> dict[str, Any]:
    signals = {
        "drift": drift_signal(),
        "confidence": confidence_signal(),
        "corrections": corrections_signal(),
    }
    decisions = decide(signals)
    return {
        "what": "15.8 - when to retrain a component, and when to refuse",
        "signals": signals,
        "decisions": decisions,
        "verdict": {
            "any_due": any(d["verdict"] == "retrain" for d in decisions),
            "refusals": [d["component"] for d in decisions if d["verdict"] == "refuse"],
        },
    }


def render(result: dict) -> str:
    lines = [
        "# Phase 15.8 - the retraining trigger",
        "",
        "Generated by `python -m src.mlops.retrain`.",
        "",
        "**A trigger that only ever says 'retrain' is a cron job.** This one is allowed to refuse,"
        " and the refusals are the part worth having: this project has measured three cases where"
        " retraining is the wrong move, and a rule that ignored them would recommend the thing"
        " that already failed.",
        "",
        "## The signals",
        "",
        "| signal | from | state | fires |",
        "| :--- | :--- | :--- | :--- |",
    ]
    origins = {
        "drift": "15.5 `reports/drift.json`",
        "confidence": "15.6 `reports/prediction_monitoring.json`",
        "corrections": "15.7 the correction store",
    }
    for name, signal in result["signals"].items():
        if not signal.get("available"):
            lines.append(f"| `{name}` | {origins.get(name, '')} | unavailable | no |")
            continue
        state = signal.get("why", "")
        lines.append(
            f"| `{name}` | {origins.get(name, '')} | {state} |"
            f" {'**yes**' if signal.get('fires') else 'no'} |"
        )
    lines += [
        "",
        "**Neither 15.5's nor 15.6's alarms fire this trigger from their reports**, and that is"
        " deliberate rather than an oversight: the shifts in those reports are each row's own"
        " calibration arms - a new diagram type, a new capture device, 14.5's degradations - which"
        " exist to prove the detector works. They are not observed traffic. Firing a retrain on"
        " them would be firing on the calibration set. Real traffic is scored with `--against`.",
        "",
        "## The decisions",
        "",
        "| component | verdict | firing | reason |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for decision in result["decisions"]:
        mark = {"retrain": "**retrain**", "refuse": "**refuse**", "no": "no"}[decision["verdict"]]
        lines.append(
            f"| {decision['component']} | {mark} |"
            f" {', '.join(decision['firing_signals']) or '-'} | {decision['reason']} |"
        )
    lines += [
        "",
        "## Where retraining is already known not to help",
        "",
        "| component | pull this instead | the measurement that says so |",
        "| :--- | :--- | :--- |",
    ]
    for component, entry in EXHAUSTED.items():
        lines.append(f"| {component} | {entry['lever']} | {entry['evidence']} |")
    lines += [
        "",
        f"The correction threshold is **{CORRECTION_THRESHOLD}** usable corrections. Set against"
        " the corpus rather than picked: S3 trains on 8,896 label crops, so 500 corrected labels"
        " is about 5.6% new signal on exactly the examples a user cared enough to fix. Below that"
        " the run costs more than it can move.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15.8 retraining trigger")
    ap.add_argument("--check", action="store_true", help="non-zero if any component is due")
    ap.add_argument("--explain", help="why this component would or would not be retrained")
    args = ap.parse_args(argv)

    result = collect()

    if args.explain:
        match = [d for d in result["decisions"] if d["component"] == args.explain]
        if not match:
            print(f"unknown component {args.explain!r}", file=sys.stderr)
            return 2
        json.dump(match[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render(result), encoding="utf-8")
    json.dump(result["verdict"], sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(f"-> {REPORT_MD}")
    if args.check:
        return 1 if result["verdict"]["any_due"] else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
