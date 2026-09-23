"""Phase 15.3 - the model registry: one staged version per component, staged on evidence.

    python -m src.mlops.registry --populate   # register every component, stage it, write the report
    python -m src.mlops.registry --list       # what is registered, and at which stage
    python -m src.mlops.registry --check      # non-zero if any stage disagrees with its criterion

The plan asks for "staged versions (dev -> staging -> prod) per component". The mechanism is
MLflow's registry over 15.1's `file:mlruns` store - checked rather than assumed: the file backend
does support registered models and stage transitions on 2.17, so this row needs no tracking
server and no database.

## The stage is derived, never asserted

A registry whose stages are typed in by hand records an intention. This one records a
measurement: every component names the criterion it is held to (the S1-S9 bars in
`contributing.md`), and the stage is a function of whether the shipped artefact meets it.

    prod      the criterion is met, by the artefact that is actually wired into the pipeline
    staging   this is the best the project has and it is what ships, but the criterion is
              **not** met - so it is registered, and it is not called production
    dev       a candidate that lost: superseded by a later run, or never wired in

That middle rung is the point of the row. Components ship that miss their bar - OCR at 0.257 CER
against 0.15, assembly at 22.5 median GED against 3 - and a registry that promoted them to `prod`
because they are the only option would be recording a wish. Registering them at `staging` with
the gap in the tags says the true thing: this is what runs, and it is not good enough yet.

## What a version carries

Each version is tagged with the metric, the criterion, the comparison, the source artefact and
whether the component has a call site in `src/pipeline/*.py` - 14.4's distinction between being
built and being on the path, which a registry is exactly the wrong place to blur.

MLflow deprecated stages in 2.9 in favour of aliases, so both are set on every version: the
stage because "dev -> staging -> prod" is what this row is asked for by name, and the alias
because that is what will still resolve once stages are removed. Promotion is
idempotent: re-running `--populate` re-derives every stage from the current artefacts, so a
re-measured component moves stage on its own rather than needing someone to remember.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from typing import Any

from src.mlops.tracking import STORE_URI, _dig, available
from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "model_registry.json"
REPORT_MD = ROOT / "reports" / "model_registry.md"

#: The artefact each component actually ships, the metric it is judged on, and the phase bar it
#: answers to. `direction` is which way is better, so a criterion is checked rather than eyeballed.
SHIPPED: tuple[dict[str, Any], ...] = (
    {
        "name": "dreamscript-classification",
        "component": "classification",
        "run": "s1_clip_svm",
        "artefact": "reports/s1_heldout_scribes.json",
        "metric": "accuracy",
        "criterion": 0.92,
        "direction": "higher",
        "bar": "S1 - diagram-type accuracy on held-out scribes >= 92%",
    },
    {
        "name": "dreamscript-detection",
        "component": "detection",
        "run": "s2_component_detector",
        "artefact": "reports/s2_component_detection.json",
        "metric": "map50",
        "criterion": 0.80,
        "direction": "higher",
        "bar": "S2 - component detection mAP@0.5 >= 0.80",
    },
    {
        "name": "dreamscript-ocr",
        "component": "ocr",
        "run": "s3_trocr_large_selected",
        "artefact": "reports/s3_label_ocr.json",
        "metric": "cer",
        "criterion": 0.15,
        "direction": "lower",
        "bar": "S3 - label OCR character error rate <= 0.15",
    },
    {
        "name": "dreamscript-parsing",
        "component": "parsing",
        "run": "s4_role_hmm",
        "artefact": "reports/s4_role_labelling.json",
        "metric": "macro_f1",
        "criterion": 0.80,
        "direction": "higher",
        "bar": "S4 - HMM semantic-role macro F1 >= 0.80",
    },
    {
        "name": "dreamscript-assembly",
        "component": "assembly",
        "run": "s5_test_trocr_large",
        "artefact": "reports/s5_test_large.json",
        "metric": "overall.median_ged",
        "criterion": 3.0,
        "direction": "lower",
        "bar": "S5 - median graph edit distance to ground-truth IR <= 3",
    },
    {
        "name": "dreamscript-generation",
        "component": "generation",
        "run": "lora_r64_quality",
        "artefact": "experiments/llm/quality/done.json",
        "metric": "lora.functional",
        "criterion": 0.70,
        "direction": "higher",
        "bar": "S7 - functional correctness pass@1 >= 70%",
    },
)

#: Candidates that lost. They are registered so the registry records what was tried and beaten,
#: which is the thing a bare "here is the winner" registry throws away - but never above `dev`.
SUPERSEDED: tuple[dict[str, Any], ...] = (
    {
        "name": "dreamscript-ocr",
        "component": "ocr",
        "run": "s3_trocr_large_16_epochs",
        "artefact": "reports/s3_label_ocr_large.json",
        "metric": "cer",
        "why": "fewer epochs than the selected checkpoint",
    },
    {
        "name": "dreamscript-ocr",
        "component": "ocr",
        "run": "s3_trocr_large_28_epochs",
        "artefact": "reports/s3_label_ocr_large28.json",
        "metric": "cer",
        "why": "longer schedule, no better on the frozen split",
    },
    {
        "name": "dreamscript-ocr",
        "component": "ocr",
        "run": "s3_trocr_strong_augmentation",
        "artefact": "reports/s3_label_ocr_aug.json",
        "metric": "cer",
        "why": "stronger augmentation, not selected",
    },
    {
        "name": "dreamscript-ocr",
        "component": "ocr",
        "run": "s3_retrained_on_reselection",
        "artefact": "reports/s3_label_ocr_resel_retrain.json",
        "metric": "cer",
        "why": "retrained after crop reselection, not selected",
    },
)

#: 14.4's finding, reused rather than restated: a component can be built and still not be on the
#: path. The registry records which, because "what is in production" is otherwise ambiguous.
PIPELINE_PACKAGE = ROOT / "src" / "pipeline"

#: Which source package each component lives in, for the call-site check.
PACKAGES = {
    "classification": ("classify",),
    "detection": ("detect",),
    "ocr": ("ocr",),
    "parsing": ("parse",),
    "assembly": ("assemble",),
    "generation": ("llm", "codegen"),
}

#: The plan's stage names against MLflow's fixed vocabulary.
MLFLOW_STAGE = {"prod": "Production", "staging": "Staging", "dev": "None"}


def meets(value: float | None, criterion: float, direction: str) -> bool | None:
    if value is None:
        return None
    return value >= criterion if direction == "higher" else value <= criterion


def _metric(artefact: str, key: str) -> float | None:
    path = ROOT / artefact
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = _dig(payload, key)
    return float(value) if isinstance(value, (int, float)) else None


def _imports(path: Path) -> set[str]:
    """The `src.*` modules one file imports, top-level and function-local alike."""
    found: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return found
    for match in re.finditer(r"(?:from|import)\s+(src\.[A-Za-z0-9_.]*)", text):
        found.add(match.group(1))
    return found


def reachable_packages() -> set[str]:
    """Every `src.<package>` reachable from `src/pipeline/*.py`, following imports transitively.

    A direct-import check is not enough and reports the wrong thing. The pipeline never imports
    `src.ocr`; it imports `src.assemble`, and assembly is what runs TrOCR - so a direct check
    calls the OCR component "not on the path" while every golden page is decoding label crops
    through it. 14.4 asked its narrower question deliberately (which components does the
    pipeline *itself* call), and this row needs the wider one: is this component reached at all
    when a page is processed.
    """
    seen: set[str] = set()
    queue = [p for p in PIPELINE_PACKAGE.glob("*.py")]
    while queue:
        current = queue.pop()
        for module in _imports(current):
            package = module.split(".")[1] if module.count(".") >= 1 else None
            if not package or package in seen:
                continue
            seen.add(package)
            directory = ROOT / "src" / package
            if directory.is_dir():
                queue.extend(directory.glob("*.py"))
            elif (ROOT / "src" / f"{package}.py").is_file():
                queue.append(ROOT / "src" / f"{package}.py")
    return seen


def direct_call_site(component: str) -> bool:
    """Does `src/pipeline/*.py` import this component itself? 14.4's narrow question."""
    if not PIPELINE_PACKAGE.is_dir():
        return False
    wanted = {f"src.{n}" for n in PACKAGES.get(component, ())}
    for path in PIPELINE_PACKAGE.glob("*.py"):
        for module in _imports(path):
            # `src.ocr.s3` counts as `src.ocr`; a prefix match, not equality.
            if any(module == w or module.startswith(f"{w}.") for w in wanted):
                return True
    return False


def has_call_site(component: str) -> bool:
    """Is this component reached from the pipeline at all, following imports transitively?"""
    if not PIPELINE_PACKAGE.is_dir():
        return False
    return bool(set(PACKAGES.get(component, ())) & reachable_packages())


def resolve() -> list[dict[str, Any]]:
    """Every version the registry should hold, with its stage derived from the evidence."""
    rows: list[dict[str, Any]] = []
    for spec in SHIPPED:
        value = _metric(spec["artefact"], spec["metric"])
        passed = meets(value, spec["criterion"], spec["direction"])
        if passed is None:
            stage = "dev"
            reason = "the artefact is missing, so nothing can be claimed for it"
        elif passed:
            stage = "prod"
            reason = "meets its criterion and is the wired artefact"
        else:
            stage = "staging"
            reason = "this is what ships and it does not meet its criterion, so it is not prod"
        rows.append(
            {
                "name": spec["name"],
                "component": spec["component"],
                "run": spec["run"],
                "artefact": spec["artefact"],
                "metric": spec["metric"],
                "bar": spec["bar"],
                "value": value,
                "criterion": spec["criterion"],
                "direction": spec["direction"],
                "meets_criterion": passed,
                "stage": stage,
                "stage_reason": reason,
                "on_pipeline_path": has_call_site(spec["component"]),
                "direct_call_site": direct_call_site(spec["component"]),
                "role": "shipped",
            }
        )
    for spec in SUPERSEDED:
        rows.append(
            {
                "name": spec["name"],
                "component": spec["component"],
                "run": spec["run"],
                "artefact": spec["artefact"],
                "metric": spec["metric"],
                "bar": None,
                "value": _metric(spec["artefact"], spec["metric"]),
                "criterion": None,
                "direction": None,
                "meets_criterion": None,
                "stage": "dev",
                "stage_reason": spec["why"],
                "on_pipeline_path": False,
                "direct_call_site": False,
                "role": "superseded",
            }
        )
    return rows


def _run_id_for(client, run_name: str) -> str | None:
    """The 15.1 run this version came from, so the registry points back at its evidence."""
    for experiment in client.search_experiments():
        found = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"attributes.run_name = '{run_name}'",
            max_results=1,
        )
        if found:
            return found[0].info.run_id
    return None


def populate(store_uri: str = STORE_URI, rows: list[dict[str, Any]] | None = None) -> dict:
    """Register every resolved version and move it to its derived stage. Idempotent."""
    rows = resolve() if rows is None else rows
    result: dict[str, Any] = {
        "store": store_uri,
        "mlflow_available": available(),
        "registered": [],
        "errors": [],
    }
    if not available():
        result["note"] = "mlflow is not installed; the resolved table is still reported"
        result["versions"] = rows
        return result

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(store_uri)
    client = MlflowClient(tracking_uri=store_uri)

    for row in rows:
        try:
            try:
                client.create_registered_model(row["name"])
            except Exception:  # noqa: BLE001 - already there is the normal case on a re-run
                pass
            version = client.create_model_version(
                name=row["name"],
                source=str((ROOT / row["artefact"]).as_posix()),
                run_id=_run_id_for(client, row["run"]),
                tags={
                    "component": row["component"],
                    "run": row["run"],
                    "metric": row["metric"],
                    "value": "" if row["value"] is None else f"{row['value']:.4f}",
                    "criterion": "" if row["criterion"] is None else str(row["criterion"]),
                    "meets_criterion": str(row["meets_criterion"]),
                    "bar": row["bar"] or "",
                    "stage_reason": row["stage_reason"],
                    "on_pipeline_path": str(row["on_pipeline_path"]),
                    "direct_call_site": str(row.get("direct_call_site")),
                    "role": row["role"],
                    "registered_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            target = MLFLOW_STAGE[row["stage"]]
            if target != "None":
                # Stages are what the plan asks for by name, and MLflow deprecated them in 2.9
                # in favour of aliases. Both are set: the stage so this row means what it says,
                # the alias so the registry still resolves when stages are removed. The alias is
                # moved rather than added, because an alias points at one version by definition.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    client.transition_model_version_stage(
                        name=row["name"], version=version.version, stage=target
                    )
                client.set_registered_model_alias(
                    name=row["name"], alias=row["stage"], version=version.version
                )
            result["registered"].append(
                {**row, "version": int(version.version), "mlflow_stage": target}
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
            result["errors"].append({"name": row["name"], "run": row["run"], "error": str(exc)})
    result["versions"] = result["registered"]
    return result


def listing(store_uri: str = STORE_URI) -> list[dict]:
    """What the registry actually holds, read back from the store."""
    if not available():
        return []
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=store_uri)
    out = []
    for model in client.search_registered_models():
        for version in client.search_model_versions(f"name = '{model.name}'"):
            out.append(
                {
                    "name": model.name,
                    "version": int(version.version),
                    "stage": version.current_stage,
                    "run": version.tags.get("run"),
                    "metric": version.tags.get("metric"),
                    "value": version.tags.get("value"),
                    "meets_criterion": version.tags.get("meets_criterion"),
                    "role": version.tags.get("role"),
                }
            )
    return sorted(out, key=lambda r: (r["name"], r["version"]))


def disagreements(store_uri: str = STORE_URI) -> list[dict]:
    """Stages in the store that no longer match the artefacts. `--check` fails on these.

    Only the newest version of each (name, run) is compared: earlier versions are history, and a
    registry that demanded its own history be re-staged would fail every time a metric moved.
    """
    expected = {(r["name"], r["run"]): r for r in resolve()}
    newest: dict[tuple[str, str | None], dict] = {}
    for held in listing(store_uri):
        key = (held["name"], held["run"])
        if key not in newest or held["version"] > newest[key]["version"]:
            newest[key] = held
    out = []
    for key, held in newest.items():
        want = expected.get(key)
        if want is None:
            continue
        if held["stage"] != MLFLOW_STAGE[want["stage"]]:
            out.append(
                {
                    "name": held["name"],
                    "run": held["run"],
                    "version": held["version"],
                    "registered_stage": held["stage"],
                    "derived_stage": MLFLOW_STAGE[want["stage"]],
                    "reason": want["stage_reason"],
                }
            )
    return out


def render(result: dict) -> str:
    rows = result.get("versions") or []
    shipped = [r for r in rows if r.get("role") == "shipped"]
    superseded = [r for r in rows if r.get("role") == "superseded"]

    def fmt(value, digits=4):
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Phase 15.3 - the model registry",
        "",
        "Generated by `python -m src.mlops.registry --populate`.",
        "",
        f"MLflow's registry over 15.1's `{result['store']}`. The file backend supports registered"
        " models and stage transitions, so this row needs no tracking server and no database.",
        "",
        "**The stage is derived from the evidence, not typed in.** Each component names the"
        " criterion it answers to, and the stage is a function of whether the shipped artefact"
        " meets it. Re-running `--populate` re-derives every stage, so a re-measured component"
        " moves on its own.",
        "",
        "| component | run | metric | value | criterion | meets | stage | imported by"
        " pipeline | reached at all |",
        "| :--- | :--- | :--- | ---: | ---: | :--- | :--- | :--- | :--- |",
    ]
    for r in shipped:
        meets_text = {True: "yes", False: "**no**", None: "unknown"}[r["meets_criterion"]]
        lines.append(
            f"| {r['component']} | `{r['run']}` | {r['metric']} | {fmt(r['value'])} |"
            f" {fmt(r['criterion'], 2)} | {meets_text} | `{r['stage']}` |"
            f" {'yes' if r.get('direct_call_site') else 'no'} |"
            f" {'yes' if r['on_pipeline_path'] else 'no'} |"
        )
    missed = sorted(r["component"] for r in shipped if r["meets_criterion"] is False)
    lines += [
        "",
        (
            f"**{len(missed)} of {len(shipped)} shipped components miss their own bar**"
            f" ({', '.join(missed)}), and they are registered at `staging` rather than `prod`."
            " That distinction is the point of this row. Promoting them because they are the only"
            " option the project has would make the registry a statement of intent; leaving them"
            " at `staging` with the gap in their tags says the true thing - this is what runs,"
            " and it is not good enough yet."
            if missed
            else "**Every shipped component meets its criterion.**"
        ),
        "",
        "**Two path columns, because one of them alone misleads.** *Imported by pipeline* is"
        " 14.4's question - does `src/pipeline/*.py` call this component itself - and on its own"
        " it calls OCR unreached, while every golden page decodes label crops through it via"
        " `src.assemble`. *Reached at all* follows imports transitively and is the honest answer"
        " to \"does a page touch this\", but it is permissive enough to reach packages nothing"
        " on the hot path uses. Neither is the whole truth, so both are printed.",
        "",
        "## Superseded candidates",
        "",
        "Registered so the registry records what was tried and beaten, and never above `dev`.",
        "",
        "| component | run | metric | value | why it lost |",
        "| :--- | :--- | :--- | ---: | :--- |",
    ]
    for r in superseded:
        lines.append(
            f"| {r['component']} | `{r['run']}` | {r['metric']} | {fmt(r['value'])} |"
            f" {r['stage_reason']} |"
        )
    if result.get("errors"):
        lines += ["", "## Errors", ""]
        for e in result["errors"]:
            lines.append(f"- `{e['name']}` / `{e['run']}`: {e['error']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 15.3 model registry")
    ap.add_argument("--populate", action="store_true", help="register and stage every component")
    ap.add_argument("--list", action="store_true", dest="show", help="what is registered")
    ap.add_argument("--check", action="store_true", help="non-zero if a stage disagrees")
    ap.add_argument("--store", default=STORE_URI)
    args = ap.parse_args(argv)

    if args.check:
        bad = disagreements(args.store)
        json.dump({"disagreements": bad}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if bad else 0

    if args.show:
        json.dump(listing(args.store), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.populate:
        result = populate(args.store)
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        REPORT_MD.write_text(render(result), encoding="utf-8")
        json.dump(
            {
                "registered": len(result["registered"]),
                "errors": len(result["errors"]),
                "stages": {
                    stage: sum(1 for r in result["registered"] if r["stage"] == stage)
                    for stage in ("prod", "staging", "dev")
                },
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        print(f"-> {REPORT_MD}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
