# Conventions

Phase 0.3.3. These are the rules that make 300 tasks across 18 phases stay navigable. Where a
convention can be checked by a machine it is (`src/utils/naming.py`, `tests/test_naming.py`);
the rest is enforced by review.

## 1. Experiment naming

```
<phase>-<model>-<variant>-<seed>
```

| Part | Rule | Examples |
| :--- | :--- | :--- |
| phase | `p` + phase number, sub-phase allowed | `p5`, `p7.3`, `p12` |
| model | lowercase model identifier, no spaces | `knn`, `svmpoly`, `yolov8`, `qwen7b` |
| variant | the one thing this run changes | `k7`, `deg3`, `1024px`, `lora32`, `baseline` |
| seed | `s` + integer | `s42` |

```
p5-knn-k7-s42            Phase 5, KNN with k=7, seed 42
p6-mlp-adamw-s42         Phase 6, MLP, AdamW optimizer
p6-svmpoly-deg3-s42      Phase 6, polynomial SVM, degree 3
p7-hmm-viterbi-s42       Phase 7, HMM with Viterbi decoding
p9-yolov8-1024px-s42     Phase 9, detector at 1024px
p11-qlearn-eps0.1-s42    Phase 11, Q-learning, epsilon 0.1
p12-qwen7b-lora32-s1     Phase 12, QLoRA rank 32
```

The variant field carries **the single dimension under test**. A sweep over k produces
`p5-knn-k1-s42`, `p5-knn-k3-s42`, … — never `p5-knn-run2-s42`, which tells a reader nothing.
Ablations use the removed component with a `no` prefix: `p14-full-nohmm-s42`.

Run names are set in config (`logging.run_name`) so they land in both the run directory and
the experiment tracker without being typed twice.

## 2. Run directories

`experiments/` holds two things, and it is worth saying which is which because this section
used to describe only the first while the tree contained only the second.

**Timestamped runs**, written by `src.utils.logging.start_run`:

```
experiments/<YYYYmmdd-HHMMSS>_<run_name>/
    config.yaml   the fully resolved config the run used
    env.json      python, torch, CUDA, GPU, git commit and dirty flag
    run.log       human-readable log
    run.jsonl     one JSON record per log line
    metrics.json  everything passed to log_metrics(), plus final status
```

Immutable. A rerun makes a new directory; nothing is overwritten. Never edit a run directory
by hand — if a number is wrong, rerun and note it in the report.

`start_run` has exactly two callers: `src.utils.cli.main` (the config-driven stage contract) and
`src.llm.run` (Phase 12.2). **That is the honest scope of this convention**, and it is smaller
than this document used to imply. A run that refuses before it starts is discarded rather than
recorded — see `Run.discard` — which is why the tree holds none of these directories at rest.

**Per-phase artefact directories**, written by the modules themselves:

```
experiments/assemble/    detections.json, direction.json  — caches the S5 stages read
experiments/detect/      train.json, report.json, arrows/, final/
experiments/llm/         runs/, benchmark/, compare/, commits/
experiments/ocr/         crnn.json, craft/, crnn/
experiments/pipeline/    router.json, handcrafted.joblib  — fitted models the pipeline loads
```

These are **not** runs and are not immutable: they are the derived artefacts a later stage
loads by name, which is why they are named rather than timestamped. Anything here is
reproducible from the module that wrote it; nothing here is a record of a particular execution.
A module that wants to record an execution uses `start_run`.

## 3. Git

- **Commit subject:** `Phase <n.n.n>: <what changed>`, imperative, no trailing period.
- **One task per commit.** A commit corresponds to one row of `contributing.md`, and flips that row's
  status in the same commit as the work.
- **Never commit** weights, datasets, run directories, or anything over 2 MB. Pre-commit
  blocks it (`check-added-large-files`).
- **Branch** for anything speculative; `main` stays green (suite passing, hooks clean).

## 4. Code

| Rule | Enforced by |
| :--- | :--- |
| Line length 100 | black, ruff |
| Import order: stdlib, third-party, first-party (`src`) | isort (`profile=black`) |
| Type hints on public functions | review |
| Docstring on every module, naming its owning phase | `tests/test_package_imports.py` |
| No hardcoded paths or hyperparameters — configs only | `tests/test_config.py` |
| Notebooks committed without output cells | nbstripout |

Comments explain *why*, not *what*. A comment that restates the code is deleted.

## 5. Configs

- `configs/base.yaml` holds anything shared; module configs inherit it under `defaults:`.
- One config per runnable module, named after it: `configs/<module>.yaml`.
- Sweeps use dotted overrides (`model=knn cv.n_splits=10`), not copies of the file.
- Paths are always repo-relative.

## 6. Data

| Directory | Rule |
| :--- | :--- |
| `data/raw/` | read-only; never written by code |
| `data/interim/` | caches; safe to delete and regenerate |
| `data/processed/` | the corpus and its manifest |
| `data/features/` | feature tables and cached embeddings |

Splits are **scribe-disjoint**: a person's drawings appear in exactly one split. Any script
that builds splits asserts this (Phase 1.3.3).

## 7. Metrics and reporting

- Every reported number names the run directory that produced it.
- Headline numbers are reported over **≥3 seeds** with a spread, never a single lucky run.
- The comparison baseline is stated explicitly next to any claimed improvement.
- Failures are reported alongside successes; `reports/gallery.md` carries the worst cases,
  not only the best.

## 8. Naming things

| Thing | Convention | Example |
| :--- | :--- | :--- |
| Python modules and functions | `snake_case` | `extract_primitives` |
| Classes | `PascalCase` | `DiagramTraversalEnv` |
| Config keys | `snake_case` | `binarization.window_size` |
| Diagram types | `snake_case` singular | `flowchart`, `state_machine`, `er_diagram` |
| Semantic roles | `kebab-case` | `decision-node`, `ui-button` |
| IR node ids | `<type>_<index>` | `node_0`, `edge_3` |
| Figures | `<phase>_<subject>.png` | `p5_knn_boundary.png` |
