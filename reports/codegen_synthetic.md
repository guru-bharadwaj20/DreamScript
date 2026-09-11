# Phase 12.1.4 — synthetic (diagram, code) pairs

Produced by `python -m src.codegen.synthetic build` (seed 1204, 2,400 per type, 30 worker
processes). Output is gitignored and rebuilt by that command:

- `data/processed/codegen/pairs/synthetic.jsonl` — 12,000 pairs, 12.1.1 schema, all `train`
- `data/processed/codegen/synthetic/images/<type>/<diagram_id>.png` — one rendered diagram per pair
- `data/processed/codegen/synthetic/report.json` — the numbers below

Load with `src.codegen.pairs.load_pairs("train", sources=["synthetic"])`.

## Why the inherited generator was not used

`src.synth.graphs` (shared with 11.2, left unchanged) was measured first, over its own 12,500-diagram
default run:

| defect | measurement |
| :--- | :--- |
| structural near-duplicates | circuits **5** distinct structures, ER **33**, flowcharts 1,015 of 2,500 each |
| looked unique only through the id | 2,500 / 2,500 distinct targets per type because the diagram id is in the code |
| target not a function of the IR text | ER columns and every circuit value / net lived only in `attrs`, which 12.1.2 drops |
| invalid IR | **0 of 500** pass `schemas/ir.schema.json` (`ir_version` float, type `er`, roles `source`/`resistor`) |
| unsolvable circuits | **926 of 2,500** fail ngspice's `.op` (12.1.7) |

## Pipeline

candidate (pure in `(type, seed)`) → IR schema validation → 12.1.1 pair with canonical id →
exact / near-duplicate / structure-cap dedup → 12.1.7 filter (compile, esbuild + render, ngspice) →
render → `write_shard` (filter and postcondition again) → merge.

| type | candidates | exact dup | near dup | structure cap | IR invalid | filter rejects | kept |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flowchart | 3,904 | 22 | 0 | 1,338 | 0 | 0 | 2,400 |
| state_machine | 3,904 | 143 | 6 | 136 | 0 | 0 | 2,400 |
| er_diagram | 3,904 | 0 | 0 | 107 | 0 | 0 | 2,400 |
| wireframe | 3,904 | 641 | 0 | 353 | 0 | 0 | 2,400 |
| circuit | 3,904 | 0 | 96 | 817 | 0 | 0 | 2,400 |

Candidates are generated in batches, so more are produced than kept; the dedup columns count
every candidate seen. 0 filter rejects is a property of the construction (every net has a DC path
to ground, no inductor/source-only loop, no invisible information), and the filter still ran on
all of them.

## Distribution and diversity of the 12,000 kept pairs

Structure signature = the IR text with every node text and edge label removed (shape, role,
wiring, traversal). Effective structures = exp(Shannon entropy) over signatures.

| type | distinct structures | max per structure | effective structures | nodes p10 / p50 / p90 / max | label vocabulary |
| :--- | ---: | ---: | ---: | :--- | ---: |
| flowchart | 1,565 | 12 | 1,107.2 | 5 / 8 / 13 / 22 | 89 |
| state_machine | 2,142 | 12 | 1,870.0 | 3 / 5 / 8 / 9 | 43 |
| er_diagram | 1,810 | 12 | 1,438.1 | 10 / 19 / 31 / 44 | 74 |
| wireframe | 2,162 | 12 | 1,898.6 | 4 / 10 / 27 / 87 | 65 |
| circuit | 1,740 | 12 | 1,342.1 | 9 / 14 / 19 / 27 | 68 |

Structure families (the 5 of `graphs.STRUCTURES`, requested round-robin):

| type | linear | branching | looping | nested | disconnected |
| :--- | ---: | ---: | ---: | ---: | ---: |
| flowchart | 184 | 496 | 483 | 519 | 718 |
| state_machine | 378 | 505 | 506 | 505 | 506 |
| er_diagram | 474 | 482 | 482 | 481 | 481 |
| wireframe | 134 | 533 | 550 | 551 | 632 |
| circuit | 93 | 533 | 553 | 615 | 606 |

`linear` is under-filled on purpose: there are few distinct linear shapes, so the structure cap
(12 per signature) stops them repeating, and the quota is filled from families that still have new
shapes. Flowchart targets are 2,320 structured / 80 dispatch (3.3%).

## Rendering

All 12,000 PNGs rendered in 64.5 s (489 MB), grayscale, with 1.3.7's wobbled-stroke primitives.
A sample of each type was opened and inspected. Two rendering bugs were found that way and fixed:
`ingest.synthetic._text` silently skips a 240 px label patch that overflows the canvas, so every
label on a narrow flowchart was missing; and BFS layering put every circuit part on one row and
every net on the next, so each wire crossed the drawing. Known remaining limit: in dense wireframe
rows, leaf labels can overlap.

## Limits

- Synthetic pairs are train-only (12.1.8) and say nothing about fidelity to human drawings; ER and
  circuit have no real diagrams at all (12.1.6).
- The vocabulary is finite (43–89 label words per type). Diversity is in structure, not language.
