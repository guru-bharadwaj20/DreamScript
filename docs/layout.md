# Repository Layout

Phase 0.2.1. Every artifact this project produces has exactly one place to live. If something
does not fit below, the tree is wrong, not the artifact.

```
DreamScript/
├── app/                    Phase 16 — the web demo
│   ├── backend/            FastAPI service (upload, predict, ir, code, feedback)
│   └── frontend/           camera capture, overlay, IR graph, code + execution panels
├── configs/                YAML configs; one per runnable module (Phase 0.2.3)
├── data/                   DVC-tracked, never committed raw (Phase 1)
│   ├── raw/                original photos and downloaded datasets, read-only
│   ├── interim/            cached preprocessing output and extracted primitives
│   ├── processed/          the unified corpus + manifest.parquet
│   └── features/           handcrafted feature tables and cached embeddings
├── docs/                   written deliverables
│   ├── data_cards/         one per dataset: source, license, size, bias notes
│   └── model_cards/        one per shipped model: intended use, metrics, limits
├── experiments/            per-run output: experiments/<timestamp>_<name>/ (Phase 0.2.4)
├── notebooks/              one clean notebook per syllabus unit (Phase 17.3)
├── reports/                figures, tables and analyses that go into the final report
├── requirements/           layered pins: base, torch, cv, classical, genai
├── scripts/                standalone checks and one-off utilities
├── src/                    the package — all importable pipeline code (Phase 0.2.2)
└── tests/                  pytest suite
    └── fixtures/           tiny sample images committed for fast tests
```

## Rules

1. **`src/` is importable, `scripts/` is not.** Anything a later phase needs to call lives in
   `src/`; `scripts/` holds entry points and environment checks that are run, never imported.
2. **Nothing writes outside `experiments/`, `data/interim/`, `data/processed/` or
   `data/features/`.** Reports and figures are copied into `reports/` deliberately, not
   dumped there by a training run.
3. **`data/raw/` is read-only.** Any transformation writes to `interim/` or `processed/`.
4. **Every run gets its own directory** under `experiments/`, named
   `<timestamp>_<phase>-<model>-<variant>-<seed>` per `docs/conventions.md`.
5. **The skeleton is committed, the contents are not.** `.gitkeep` files hold empty
   directories in git; `.gitignore` excludes everything else under the data and experiment
   trees.
