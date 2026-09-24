"""Phase 15.12 - the reproducibility check: does this machine produce the numbers on record?

    python -m src.mlops.reproduce              # reports/reproducibility.md + .json
    python -m src.mlops.reproduce --check      # non-zero if the environment or a metric diverges
    python -m src.mlops.reproduce --env-only   # just the environment audit, in seconds

The plan asks: fresh clone, `make all`, same headline metrics. That is two questions, and the
order matters more than it looks.

## The environment comes first, because it has already been the answer

A reproducibility check that goes straight to metrics answers "are the numbers different" without
answering "why", and on this project the why was the environment. **Three OpenCV distributions
were installed at once** - the pinned `opencv-contrib-python==4.10.0.84`, plus `opencv-python` and
`opencv-python-headless` at 5.0.0.93, neither of which appears in any requirements file. The
undeclared 5.0 shadowed the pin, and `cv2.HoughLinesP` and `cv2.MSER` changed behaviour under it:
eleven deskew tests raised `TypeError`, four text-layer tests failed their bars, and all of it was
briefly diagnosed as a code regression in Phase 3. It was not. It was `pip`.

So `audit_environment` runs first and is the cheap half: every pin in `requirements/*.txt`
compared against what is actually importable, plus a check for **distributions that provide the
same import and shadow each other**, which is the specific shape of that incident and is invisible
to a plain version comparison - every pin was satisfied while the wrong package was winning.

## The metrics half is honest about what it can reach

`make all` on a fresh clone needs the DVC payload and, for several rows, a GPU and hours. This
module does not pretend to run it. It re-derives the metrics that are **cheap and deterministic**
from artefacts already on disk and compares them to what `contributing.md` records, and it says
plainly which criteria it did not attempt and why. A check that silently omits the expensive half
and reports "reproducible" is worse than one that reports a partial pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import metadata
from typing import Any

from src.utils.config import ROOT

REPORT_MD = ROOT / "reports" / "reproducibility.md"
REPORT_JSON = ROOT / "reports" / "reproducibility.json"
REQUIREMENTS = ROOT / "requirements"

#: Distributions that install the same `cv2` import. Any two of these present at once means one is
#: shadowing the other and the version a pin names is not necessarily the version running. This is
#: the exact incident above, and it is listed rather than detected generically because the
#: import-to-distribution mapping is not something a pin file can express.
SHADOWING_GROUPS = {
    "cv2": ("opencv-python", "opencv-contrib-python", "opencv-python-headless"),
    "PIL": ("pillow", "pillow-simd"),
}

#: Metrics that can be re-derived from artefacts on disk in seconds, with what the record claims.
#: Anything needing a GPU, the LLM or a full pipeline run is deliberately absent and is listed in
#: `NOT_ATTEMPTED` instead, so the gap is stated rather than left to be noticed.
CHEAP_METRICS: tuple[dict[str, Any], ...] = (
    {
        "name": "S1 diagram-type accuracy",
        "artefact": "reports/s1_heldout_scribes.json",
        "key": "accuracy",
        "expected": 0.9871,
        "tolerance": 0.0005,
    },
    {
        "name": "S2 detection mAP@0.5",
        "artefact": "reports/s2_component_detection.json",
        "key": "map50",
        "expected": 0.9107,
        "tolerance": 0.0005,
    },
    {
        "name": "S4 role macro F1",
        "artefact": "reports/s4_role_labelling.json",
        "key": "macro_f1",
        "expected": 0.8003,
        "tolerance": 0.0005,
    },
    {
        "name": "S7 functional pass@1",
        "artefact": "experiments/llm/quality/done.json",
        "key": "lora.functional",
        "expected": 0.7037,
        "tolerance": 0.0005,
    },
    {
        "name": "12.3.3 reference ceiling",
        "artefact": "experiments/llm/quality/done.json",
        "key": "reference.functional",
        "expected": 0.784,
        "tolerance": 0.0005,
    },
)

#: What this check does not attempt, and why. Naming them is the point: a partial pass reported as
#: a pass is the failure mode this row exists to avoid.
NOT_ATTEMPTED = (
    ("S3 label OCR CER", "needs the TrOCR checkpoint and the label-crop corpus"),
    ("S5 graph edit distance", "needs the detector, TrOCR and the full assemble stage"),
    ("S6 / S8 pipeline runs", "needs a GPU and the golden page set"),
    ("every training row", "hours of GPU; 15.4's DVC DAG is the entry point, not this module"),
)


def _pins() -> dict[str, str]:
    """`{distribution: version}` for every `==` pin across requirements/*.txt."""
    pins: dict[str, str] = {}
    if not REQUIREMENTS.is_dir():
        return pins
    for path in sorted(REQUIREMENTS.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            match = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
            if match:
                pins[match.group(1).lower()] = match.group(2)
    return pins


def _public(version: str) -> str:
    """The version without its local segment.

    `torch==2.5.1` installed as `2.5.1+cu124` is the CUDA build of that exact version, not drift -
    PEP 440 calls `+cu124` a local version identifier and requirements files cannot name one
    without pinning a wheel index too. Reporting it as a mismatch would bury the real drift under
    a line that is always there.
    """
    return version.split("+", 1)[0]


def _installed() -> dict[str, str]:
    return {
        dist.metadata["Name"].lower(): dist.version
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    }


def audit_environment() -> dict[str, Any]:
    """Pins against reality, plus the shadowing check a version comparison cannot make."""
    pins, installed = _pins(), _installed()

    mismatched = [
        {"package": name, "pinned": version, "installed": installed[name]}
        for name, version in sorted(pins.items())
        if name in installed and _public(installed[name]) != _public(version)
    ]
    local_builds = [
        {"package": name, "pinned": version, "installed": installed[name]}
        for name, version in sorted(pins.items())
        if name in installed
        and installed[name] != version
        and _public(installed[name]) == _public(version)
    ]
    missing = [
        {"package": name, "pinned": version}
        for name, version in sorted(pins.items())
        if name not in installed
    ]

    shadowed = []
    for module, group in SHADOWING_GROUPS.items():
        present = [name for name in group if name in installed]
        if len(present) > 1:
            shadowed.append(
                {
                    "import": module,
                    "provided_by": [{"package": n, "version": installed[n]} for n in present],
                    "pinned": [n for n in present if n in pins],
                    "undeclared": [n for n in present if n not in pins],
                    "why_it_matters": (
                        "these install the same module, so which one wins is an install-order"
                        " accident and every pin can be satisfied while the wrong code runs"
                    ),
                }
            )

    return {
        "pins": len(pins),
        "mismatched": mismatched,
        "local_builds": local_builds,
        "missing": missing,
        "shadowed_imports": shadowed,
        "ok": not mismatched and not shadowed,
        "reading": (
            "a shadowed import is the failure this check exists for: it is invisible to a plain"
            " version comparison, because the pin is satisfied and a different distribution is"
            " the one actually imported"
        ),
    }


def _dig(payload: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(payload, dict) or part not in payload:
            return None
        payload = payload[part]
    return payload


def check_metrics() -> dict[str, Any]:
    """Re-read each cheap metric from its artefact and compare to the record."""
    rows = []
    for spec in CHEAP_METRICS:
        path = ROOT / spec["artefact"]
        if not path.is_file():
            rows.append({**spec, "found": None, "status": "artefact missing"})
            continue
        try:
            value = _dig(json.loads(path.read_text(encoding="utf-8")), spec["key"])
        except (OSError, ValueError) as exc:
            rows.append({**spec, "found": None, "status": f"unreadable: {exc}"})
            continue
        if not isinstance(value, int | float):
            rows.append({**spec, "found": value, "status": "not a number"})
            continue
        delta = abs(float(value) - spec["expected"])
        rows.append(
            {
                **spec,
                "found": round(float(value), 6),
                "delta": round(delta, 6),
                "status": "matches" if delta <= spec["tolerance"] else "DIVERGES",
            }
        )
    return {
        "checked": rows,
        "matched": sum(1 for r in rows if r["status"] == "matches"),
        "diverged": sum(1 for r in rows if r["status"] == "DIVERGES"),
        "unavailable": sum(1 for r in rows if r["status"] not in ("matches", "DIVERGES")),
        "ok": all(r["status"] == "matches" for r in rows),
    }


def collect(env_only: bool = False) -> dict[str, Any]:
    environment = audit_environment()
    result: dict[str, Any] = {
        "what": "15.12 - does this machine reproduce what the record claims",
        "environment": environment,
        "not_attempted": [{"criterion": c, "why": w} for c, w in NOT_ATTEMPTED],
    }
    if env_only:
        result["verdict"] = {"environment_ok": environment["ok"], "metrics_checked": False}
        return result
    metrics = check_metrics()
    result["metrics"] = metrics
    result["verdict"] = {
        "environment_ok": environment["ok"],
        "metrics_ok": metrics["ok"],
        "metrics_checked": True,
        "reproducible": environment["ok"] and metrics["ok"],
    }
    return result


def render(result: dict) -> str:
    environment = result["environment"]
    lines = [
        "# Phase 15.12 - reproducibility check",
        "",
        "Generated by `python -m src.mlops.reproduce`.",
        "",
        "Two questions in order, and the order is the finding: **the environment first**, because"
        " on this project the environment has already been the answer.",
        "",
        "## The environment",
        "",
        f"{environment['pins']} pinned distributions across `requirements/*.txt`.",
        "",
    ]
    if environment["shadowed_imports"]:
        lines += [
            "**Shadowed imports - the failure this check exists for:**",
            "",
            "| import | provided by | undeclared |",
            "| :--- | :--- | :--- |",
        ]
        for row in environment["shadowed_imports"]:
            provided = ", ".join(f"`{p['package']}` {p['version']}" for p in row["provided_by"])
            undeclared = ", ".join(f"`{n}`" for n in row["undeclared"]) or "-"
            lines.append(f"| `{row['import']}` | {provided} | {undeclared} |")
        lines += [
            "",
            "Two distributions providing one module means which one wins is an install-order"
            " accident, and **every pin can be satisfied while the wrong code runs** - which is"
            " exactly what happened here and is invisible to a version comparison.",
            "",
        ]
    else:
        lines += [
            "No shadowed imports: every checked module is provided by exactly one distribution.",
            "",
        ]

    if environment["mismatched"]:
        lines += [
            "**Installed does not match pinned:**",
            "",
            "| package | pinned | installed |",
            "| :--- | :--- | :--- |",
        ]
        for row in environment["mismatched"]:
            lines.append(f"| `{row['package']}` | {row['pinned']} | **{row['installed']}** |")
        lines.append("")
    else:
        lines += ["Every installed pin matches its requirement.", ""]

    if environment.get("local_builds"):
        builds = ", ".join(
            f"`{r['package']}` {r['installed']}" for r in environment["local_builds"]
        )
        lines += [
            f"Local builds of the pinned version, which are not drift: {builds}. PEP 440 calls"
            " `+cu124` a local version identifier; a requirements file cannot name one without"
            " pinning a wheel index too.",
            "",
        ]

    if "metrics" in result:
        metrics = result["metrics"]
        lines += [
            "## The headline metrics",
            "",
            f"{metrics['matched']} match, {metrics['diverged']} diverge,"
            f" {metrics['unavailable']} unavailable.",
            "",
            "| metric | recorded | found | status |",
            "| :--- | ---: | ---: | :--- |",
        ]
        for row in metrics["checked"]:
            found = "-" if row.get("found") is None else row["found"]
            lines.append(f"| {row['name']} | {row['expected']} | {found} | {row['status']} |")
        lines.append("")

    lines += [
        "## What this does not attempt",
        "",
        "A partial pass reported as a pass is the failure mode this row exists to avoid, so the"
        " gap is named rather than left to be noticed.",
        "",
        "| criterion | why not |",
        "| :--- | :--- |",
    ]
    for row in result["not_attempted"]:
        lines.append(f"| {row['criterion']} | {row['why']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15.12 reproducibility check")
    ap.add_argument("--check", action="store_true", help="non-zero if anything diverges")
    ap.add_argument("--env-only", action="store_true", help="skip the metric comparison")
    args = ap.parse_args(argv)

    result = collect(env_only=args.env_only)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render(result), encoding="utf-8")
    json.dump(result["verdict"], sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(f"-> {REPORT_MD}")
    if args.check:
        ok = result["verdict"].get("reproducible", result["verdict"]["environment_ok"])
        return 0 if ok else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
