# Phase 3.3.1 — stroke IoU

30 items with exact stroke ground truth, put through the Phase 3.1
photometric pipeline. The ground truth is the rasterised pen trajectory from the FA
database rather than a hand tracing — see `src/preprocess/evalset.py` for why, and for
why this is an **upper bound** on real photographs.

| | |
| :--- | ---: |
| mean IoU | **0.821** |
| median IoU | 0.868 |
| worst IoU | 0.558 |
| items below the 0.80 bar | 9 |
| mean precision | 0.823 |
| mean recall | 0.998 |

## By degradation

Each item carries one to three of the Phase 1.3.6 transforms, so an item counts under
each of them. Worst first.

| damage | items | mean IoU |
| :--- | ---: | ---: |
| jpeg | 7 | 0.726 |
| blur | 12 | 0.737 |
| glare | 6 | 0.790 |
| paper_texture | 9 | 0.806 |
| stain | 7 | 0.836 |
| shadow | 7 | 0.848 |
| brightness | 9 | 0.850 |

## Worst ten items

| item | damage | IoU | precision | recall |
| :--- | :--- | ---: | ---: | ---: |
| `writer002_fa_007` | blur, jpeg, paper_texture | 0.558 | 0.558 | 1.000 |
| `writer008_fa_005` | glare, blur, jpeg | 0.593 | 0.593 | 1.000 |
| `writer019_fa_003` | glare, blur, jpeg | 0.633 | 0.633 | 1.000 |
| `writer012_fa_007` | blur | 0.645 | 0.645 | 1.000 |
| `writer010_fa_001` | blur, stain, paper_texture | 0.687 | 0.687 | 1.000 |
| `writer018_fa_005` | shadow, jpeg, paper_texture | 0.741 | 0.741 | 1.000 |
| `writer003_fa_005` | stain, blur, brightness | 0.758 | 0.759 | 1.000 |
| `writer007_fa_007` | brightness, blur, shadow | 0.769 | 0.769 | 1.000 |
| `writer009_fa_003` | blur | 0.785 | 0.785 | 1.000 |
| `writer015_fa_011` | brightness, paper_texture, jpeg | 0.824 | 0.825 | 1.000 |
