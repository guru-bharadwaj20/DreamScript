# Phase 14.8 - human baseline and the speedup that survives the pass rate

**No human was timed for this row.** none was run; every human-side number below comes from the stated model. The machine side is measured: 13.6's cold-cache page-to-code latency and 12.1.6's reference programs.

## The comparison at the median test diagram

* the median held-out diagram has **14.5 nodes, 17.0 edges** and a reference program of **2355.0 characters** (162 diagrams)
* modelled human time: **1017.2 s** (read the drawing, type the program, run it and fix the typo)
* measured pipeline latency: **2.71 s** median over 25 pages, cold cache
* end-to-end functional pass rate: **0.1914** (from a correct IR it is 0.784)

| | seconds | speedup |
| :--- | ---: | ---: |
| human, modelled | 1017.2 | 1.00x |
| DreamScript, when the output is correct | 47.7 | **21.32x** |
| DreamScript, expected (a wrong program costs a repair) | 241.8 | **4.21x** |

The pitch number is the second one. The first assumes away the 81% of pages whose program does not pass its own functional test, and a person has to find that out by reading the program - which is why the review cost is charged in both rows and the repair cost only in the second.

**Break-even pass rate: -3.0397.** A value at or below zero means the pipeline still wins at a pass rate of *zero*, and that is what this corpus reports - because under these parameters repairing a wrong program costs 240 s while writing one from the drawing costs 1,017 s. So the number this row actually turns on is **the cost of a repair, not the pass rate**: the `repair_equals_rewrite` arm below charges a repair the full rewrite, and that is the arm to argue about.

## Sensitivity

| arm | human s | expected machine s | expected speedup |
| :--- | ---: | ---: | ---: |
| default | 1017.2 | 241.8 | 4.21x |
| fast_typist_300cpm | 711.1 | 241.8 | 2.94x |
| slow_typist_120cpm | 1629.5 | 241.8 | 6.74x |
| quick_reader | 967.9 | 241.8 | 4.0x |
| careful_reader | 1116.0 | 241.8 | 4.62x |
| cheap_repair_60s | 1017.2 | 96.2 | 10.57x |
| expensive_repair_600s | 1017.2 | 532.9 | 1.91x |
| repair_equals_rewrite | 1017.2 | 873.4 | 1.16x |

## The study this row did not run

1. 10 participants who can write Python, none of them the author
2. 8 diagrams each from the held-out test split, stratified by node count, order counterbalanced so fatigue does not land on the same pages
3. timed from first sight of the photograph to a program that passes 12.3.3's functional test for that diagram - the same test the pipeline is scored by
4. record think-aloud only after the timed run, so it does not slow the timing
5. report per-participant medians and the spread, never a single mean
