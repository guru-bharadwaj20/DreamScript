# Phase 3.2.8 — shape and text layers

40 hdBPMN photographs, split by `src/preprocess/layers.py`. Both layers are
written per page to `data/interim/layers` as PNG, ink black on white.

| | |
| :--- | ---: |
| pages | 40 |
| median text share of ink | 33.0% |
| text components | 2882 |
| shape components | 2370 |
| pages where both files were written | 40 |
| pages where the split is a partition | 40 |

The text layer holds more than half the *components* and a third of the *pixels*:
handwriting is many small marks, drawing is a few large ones, and a split that got
that ratio backwards would be visibly wrong. The absolute share is an over-claim —
Phase 3.2.7 measured 16% of annotated connector ink landing in the text proposal — so
read the text layer as *enriched for writing*, not as writing. Where the share is far
below the median, the page's writing touches its shapes and stayed with them; the
module docstring says why that direction was chosen.

| page | text share | text components | shape components |
| :--- | ---: | ---: | ---: |
| `ex00_writer0031.ir` | 69.8% | 117 | 21 |
| `ex00_writer0030.ir` | 64.2% | 103 | 13 |
| `ex00_writer0059.ir` | 63.5% | 125 | 17 |
| `ex00_writer0057.ir` | 62.4% | 88 | 11 |
| `ex00_writer0041.ir` | 60.7% | 83 | 12 |
| `ex00_writer0038.ir` | 59.8% | 104 | 10 |
| `ex00_writer0008.ir` | 57.6% | 83 | 21 |
| `ex00_writer0025.ir` | 53.5% | 72 | 60 |
| `ex00_writer0045.ir` | 52.8% | 97 | 41 |
| `ex00_writer0058.ir` | 49.6% | 56 | 42 |
| `ex00_writer0051.ir` | 48.9% | 115 | 51 |
| `ex00_writer0035.ir` | 48.6% | 101 | 32 |
| `ex00_writer0054.ir` | 47.8% | 85 | 37 |
| `ex00_writer0042.ir` | 47.7% | 74 | 18 |
| `ex00_writer0053.ir` | 46.6% | 92 | 15 |
| `ex00_writer0002.ir` | 45.5% | 142 | 142 |
| `ex00_writer0006.ir` | 44.8% | 80 | 13 |
| `ex00_writer0013.ir` | 43.7% | 99 | 15 |
| `ex00_writer0047.ir` | 38.5% | 55 | 6 |
| `ex00_writer0034.ir` | 38.3% | 190 | 151 |
