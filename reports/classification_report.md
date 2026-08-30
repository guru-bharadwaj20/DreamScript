# Phase 5.2.3 — classification report

Out-of-fold predictions from 3 repeats of stratified 5-fold CV over 1340 real photographs. Every number is an average over the repeats.

## Headline

| model | macro F1 | weighted F1 | balanced accuracy | macro − weighted |
| :--- | ---: | ---: | ---: | ---: |
| `logreg` | **0.7917** | 0.9379 | 0.7891 | -0.1461 |
| `knn` | **0.7614** | 0.9224 | 0.7523 | -0.1611 |
| `tree` | **0.7434** | 0.9001 | 0.7596 | -0.1568 |
| `majority` | **0.1237** | 0.2770 | 0.2000 | -0.1533 |

## `logreg`

| class | precision | recall | F1 | ± | support |
| :--- | ---: | ---: | ---: | ---: | ---: |
| circuit | 0.343 | 0.233 | 0.277 | 0.053 | 40 |
| er_diagram | 0.814 | 0.780 | 0.796 | 0.006 | 50 |
| flowchart | 0.941 | 0.958 | 0.950 | 0.004 | 600 |
| state_machine | 0.920 | 0.993 | 0.955 | 0.016 | 50 |
| wireframe | 0.981 | 0.981 | 0.981 | 0.001 | 600 |

## `knn`

| class | precision | recall | F1 | ± | support |
| :--- | ---: | ---: | ---: | ---: | ---: |
| circuit | 0.359 | 0.208 | 0.263 | 0.004 | 40 |
| er_diagram | 0.663 | 0.653 | 0.658 | 0.017 | 50 |
| flowchart | 0.938 | 0.942 | 0.940 | 0.002 | 600 |
| state_machine | 0.980 | 0.980 | 0.980 | 0.008 | 50 |
| wireframe | 0.955 | 0.978 | 0.967 | 0.002 | 600 |

## `tree`

| class | precision | recall | F1 | ± | support |
| :--- | ---: | ---: | ---: | ---: | ---: |
| circuit | 0.273 | 0.325 | 0.296 | 0.031 | 40 |
| er_diagram | 0.575 | 0.673 | 0.620 | 0.019 | 50 |
| flowchart | 0.913 | 0.909 | 0.911 | 0.007 | 600 |
| state_machine | 0.929 | 0.953 | 0.941 | 0.001 | 50 |
| wireframe | 0.962 | 0.937 | 0.949 | 0.005 | 600 |

## `majority`

| class | precision | recall | F1 | ± | support |
| :--- | ---: | ---: | ---: | ---: | ---: |
| circuit | 0.000 | 0.000 | 0.000 | 0.000 | 40 |
| er_diagram | 0.000 | 0.000 | 0.000 | 0.000 | 50 |
| flowchart | 0.448 | 1.000 | 0.619 | 0.000 | 600 |
| state_machine | 0.000 | 0.000 | 0.000 | 0.000 | 50 |
| wireframe | 0.000 | 0.000 | 0.000 | 0.000 | 600 |
