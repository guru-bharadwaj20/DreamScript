# Phase 0.2 — Repository Scaffolding Acceptance Report

What was built, and the evidence each acceptance criterion was met.

| Task | Artifact | Acceptance evidence |
| :--- | :--- | :--- |
| 0.2.1 Directory tree | `docs/layout.md`, `.gitkeep` skeleton, `.gitignore` policy | Tree present; data and experiment contents excluded from git while the skeleton is tracked |
| 0.2.2 Package modules | 11 subpackages under `src/` | `pytest tests/test_package_imports.py` — 14 passed; each package declares its owning phase, and a test fails if `src/` and the list drift apart |
| 0.2.3 Config system | `src/utils/config.py`, `src/utils/cli.py`, 12 YAML configs | `pytest tests/test_config.py` — 31 passed, including `python -m src.<pkg> --config configs/<pkg>.yaml` for all 11 packages |
| 0.2.4 Logging | `src/utils/logging.py`, `scripts/dummy_run.py` | `dummy_run.py` → 6/6 PASS; run dir contains `config.yaml`, `env.json`, `metrics.json`, `run.jsonl`, `run.log`; `pytest tests/test_logging.py` — 7 passed |
| 0.2.5 Task runner | `Makefile`, `tasks.ps1` | All 24 tasks dispatch (`-DryRun`); `dummy-run` and `preprocess` executed for real; `pytest tests/test_task_runner.py` — 27 passed |
| 0.2.6 Pre-commit | `.pre-commit-config.yaml`, `requirements/dev.txt` | `pre-commit run --all-files` clean; a deliberately bad probe commit was **rejected with exit 1**, HEAD unchanged, 4 hooks failing (debug-statements, black, isort, ruff) |
| 0.2.7 Test scaffold | `tests/conftest.py`, `tests/fixtures/*.png`, `scripts/make_fixtures.py` | 5 fixtures, 15,155 bytes total; **full suite: 105 passed** |

## Design decisions worth carrying forward

**The entry-point contract exists before the stages do.** `python -m src.<pkg> --config
configs/<pkg>.yaml` works for all 11 packages today; unimplemented stages raise
`StageNotImplemented`, log the run, print the resolved config and exit 2. The interface is
therefore testable from Phase 0 rather than after each stage is written, and every stage
starts from a working skeleton instead of an empty file.

**Config composition is a small `defaults:` list, not full Hydra.** `configs/base.yaml` holds
paths, seed, logging and hardware; module configs inherit it and add one block. Dotted
`key=value` overrides mean a sweep never needs a new file. This keeps the Hydra idea that
matters without Hydra's working-directory rewriting, which would fight the run-directory
scheme in 0.2.4.

**Every run is immutable and self-describing.** `experiments/<timestamp>_<run_name>/` captures
the resolved config, the git commit and dirty flag, the GPU, and metrics as both
`metrics.json` and a `run.jsonl` stream. Two runs in the same second get distinct
directories. Nothing is ever overwritten, which is what makes the Phase 14 ablation table
trustworthy.

**`make` is not installed on the target machine.** GNU make is absent from this Windows box,
so `tasks.ps1` is what actually runs here and the `Makefile` stays canonical for Linux and
CI. `tests/test_task_runner.py` fails if the two drift apart, so the mirror cannot rot.

**Fixtures are generated, not photographed.** Five synthetic sketches (one per diagram type)
are committed at 15 KB total, so the suite runs with no dataset present. Sparse speckle noise
rather than per-pixel Gaussian keeps them realistic *and* small — full-frame noise defeated
PNG compression and made them 50 KB each. Regeneration is byte-identical, and a test asserts
it.

## Follow-ups deferred to later phases

- `pdf2image` needs the poppler binary on PATH before Phase 1 can ingest PDF scans.
- Real fixtures from the Phase 1.2 chaos corpus should be added alongside the synthetic ones
  once collection starts, to catch what synthetic geometry cannot.
- CI (Phase 15.9) will run `pre-commit run --all-files` and the pytest suite; the hooks are
  already pinned by revision so local and CI results match.
