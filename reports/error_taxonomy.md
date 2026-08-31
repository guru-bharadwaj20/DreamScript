# Phase 5.3.4 — error taxonomy

All three models' out-of-fold predictions over the 1340 real photographs, from 5.2.1's partition at seed 42 - so every model is judged on identical rows.

## The prediction `plan.md` made

The task line reads *expect flowchart <-> state_machine*. **It does not hold.** The largest confusion is `circuit -> flowchart`, and the predicted pair appears at no rank at all inside the top ten.

## Confusions, pooled over the three models

| # | true | predicted | pages | share of the true class | logreg / knn / tree |
| ---: | :--- | :--- | ---: | ---: | :--- |
| 1 | circuit | flowchart | 64 | 0.533 | 24 / 20 / 20 |
| 2 | flowchart | circuit | 50 | 0.028 | 15 / 12 / 23 |
| 3 | flowchart | er_diagram | 32 | 0.018 | 4 / 11 / 17 |
| 4 | wireframe | flowchart | 30 | 0.017 | 5 / 4 / 21 |
| 5 | er_diagram | flowchart | 29 | 0.193 | 9 / 12 / 8 |
| 6 | flowchart | wireframe | 24 | 0.013 | 5 / 11 / 8 |
| 7 | circuit | wireframe | 19 | 0.158 | 5 / 10 / 4 |
| 8 | wireframe | er_diagram | 19 | 0.011 | 6 / 3 / 10 |
| 9 | wireframe | circuit | 13 | 0.007 | 1 / 5 / 7 |
| 10 | er_diagram | wireframe | 10 | 0.067 | 2 / 5 / 3 |

## Systematic or model-specific

| missed by | pages |
| :--- | ---: |
| no model | 1135 |
| 1 of 3 | 133 |
| 2 of 3 | 42 |
| all three models | 30 |

**30 pages (2.2%) are missed by all three models** - a hard core that is a property of the corpus rather than of any estimator, and the population Phase 9's detector has to attack.

| class | rows | missed by all three | share |
| :--- | ---: | ---: | ---: |
| circuit | 40 | 19 | 0.475 |
| er_diagram | 50 | 7 | 0.140 |
| flowchart | 600 | 2 | 0.003 |
| state_machine | 50 | 0 | 0.000 |
| wireframe | 600 | 2 | 0.003 |

## Confidence when wrong

| model | mean confidence when right | when wrong | errors | confident (≥ 0.90) errors |
| :--- | ---: | ---: | ---: | ---: |
| `logreg` | 0.973 | 0.825 | 79 | 31 (39%) |
| `knn` | 1.000 | 1.000 | 97 | 97 (100%) |
| `tree` | 0.992 | 0.970 | 131 | 117 (89%) |

## What the misread pages look like — `circuit -> flowchart`

31 misread circuit pages against 8 read correctly, same true class, features standardised so the gap is in standard deviations.

| feature | gap (SD) | the misread pages are |
| :--- | ---: | :--- |
| `contain_nested_count` | -3.766 | lower |
| `conn_components` | -3.066 | lower |
| `dir_flow_axis_missing` | -1.623 | lower |
| `dir_axis_aligned_missing` | -1.623 | lower |
| `dir_angle_entropy_missing` | -1.623 | lower |
| `arrows_per_node` | +1.126 | higher |

## Error rate by source corpus

| source | rows | mean error rate | classes present |
| :--- | ---: | ---: | :--- |
| chaos | 140 | 0.319 | circuit, er_diagram, state_machine |
| hdbpmn | 600 | 0.060 | flowchart |
| sketch2code | 600 | 0.036 | wireframe |
