# Phase 5.3.3 — logistic regression coefficients

5.1.1's selected model (`{'penalty': 'l1', 'solver': 'saga', 'C': 10.0, 'class_weight': None}`) fitted on all 1340 real photographs, 58 standardised columns. A coefficient is the change in the log-odds of that class per **one standard deviation** of the feature, holding the others fixed; `exp(coefficient)` is the same thing as a multiplier on the odds.

## What L1 kept

| class | features kept | zeroed | largest \|coefficient\| | intercept |
| :--- | ---: | ---: | ---: | ---: |
| circuit | 53 | 5 (9%) | 1.603 | +0.995 |
| er_diagram | 54 | 4 (7%) | 2.384 | -1.560 |
| flowchart | 56 | 2 (3%) | 3.497 | +2.741 |
| state_machine | 23 | 35 (60%) | 1.440 | -3.252 |
| wireframe | 54 | 4 (7%) | 4.359 | +1.076 |

**17% of all (class, feature) weights are exactly zero.** No feature is dropped for every class.

## The strongest weights per class

### circuit

| feature | coefficient | odds ratio per SD | pushes |
| :--- | ---: | ---: | :--- |
| `line_to_curve` | -1.603 | 0.20 | away |
| `text_label_length` | -1.495 | 0.22 | away |
| `text_per_node` | -1.237 | 0.29 | away |
| `text_inside_share` | -1.224 | 0.29 | away |
| `conn_cycle_count` | -1.117 | 0.33 | away |
| `global_aspect` | +0.894 | 2.44 | towards |
| `dir_flow_axis` | +0.893 | 2.44 | towards |
| `conn_self_loops` | -0.887 | 0.41 | away |

### er_diagram

| feature | coefficient | odds ratio per SD | pushes |
| :--- | ---: | ---: | :--- |
| `text_label_length` | +2.384 | 10.85 | towards |
| `text_inside_share` | +2.276 | 9.74 | towards |
| `edge_count` | +1.723 | 5.60 | towards |
| `dir_axis_aligned` | +1.705 | 5.50 | towards |
| `layout_col_regularity_missing` | +1.173 | 3.23 | towards |
| `shape_frac_circle` | -1.169 | 0.31 | away |
| `dir_flow_axis` | +1.147 | 3.15 | towards |
| `contain_max_depth` | -1.103 | 0.33 | away |

### flowchart

| feature | coefficient | odds ratio per SD | pushes |
| :--- | ---: | ---: | :--- |
| `text_label_length` | -3.497 | 0.03 | away |
| `text_area_frac` | +2.234 | 9.34 | towards |
| `arrowhead_count` | +1.674 | 5.33 | towards |
| `global_aspect` | +1.254 | 3.51 | towards |
| `dir_axis_aligned` | +1.086 | 2.96 | towards |
| `dir_flow_axis` | +0.833 | 2.30 | towards |
| `conn_mean_degree` | -0.822 | 0.44 | away |
| `dir_angle_entropy` | -0.797 | 0.45 | away |

### state_machine

| feature | coefficient | odds ratio per SD | pushes |
| :--- | ---: | ---: | :--- |
| `text_area_frac` | -1.440 | 0.24 | away |
| `dir_axis_aligned` | -1.420 | 0.24 | away |
| `text_block_count` | -1.234 | 0.29 | away |
| `shape_frac_ellipse` | +0.951 | 2.59 | towards |
| `dir_flow_axis` | +0.901 | 2.46 | towards |
| `dir_angle_entropy` | +0.815 | 2.26 | towards |
| `conn_self_loops` | +0.654 | 1.92 | towards |
| `shape_frac_circle` | +0.636 | 1.89 | towards |

### wireframe

| feature | coefficient | odds ratio per SD | pushes |
| :--- | ---: | ---: | :--- |
| `dir_flow_axis` | -4.359 | 0.01 | away |
| `text_label_length` | +2.661 | 14.31 | towards |
| `dir_axis_aligned` | -2.080 | 0.12 | away |
| `global_aspect` | -1.986 | 0.14 | away |
| `dir_angle_entropy` | -1.343 | 0.26 | away |
| `line_to_curve` | +1.212 | 3.36 | towards |
| `arrows_per_node` | -1.114 | 0.33 | away |
| `global_bbox_fill` | -0.910 | 0.40 | away |

## Against 4.2.6's univariate ranking

Spearman correlation between each feature's largest absolute coefficient and its mutual information: **+0.484**.

| feature | max \|coefficient\| | rank by mutual information |
| :--- | ---: | ---: |
| `dir_flow_axis` | 4.359 | 2 |
| `text_label_length` | 3.497 | 5 |
| `text_inside_share` | 2.276 | 8 |
| `text_area_frac` | 2.234 | 47 |
| `dir_axis_aligned` | 2.080 | 17 |
