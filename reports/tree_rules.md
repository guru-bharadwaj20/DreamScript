# Phase 5.3.2 — decision tree rules

5.1.3's selected tree (`{'criterion': 'entropy', 'max_depth': 12, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'ccp_alpha': 0.003977}`) refitted on all 1340 real photographs: **54 leaves**, mean depth 7.02, maximum 11. Thresholds are in each feature's own units, mapped back through the fitted scaler, and 4.2.3's missingness indicators are printed as *present* / *missing* rather than as a threshold on a 0/1 column.

## Shape of the tree

| | |
| :--- | ---: |
| leaves | 54 |
| median rows per leaf | 5 |
| leaves holding a single row | 3 (5.6%) |
| mean leaf purity | 0.919 |
| rows covered by the 15 rules below | 86.2% |

Leaves per predicted class: circuit 11, er_diagram 7, flowchart 23, state_machine 1, wireframe 12.

## The 15 rules by coverage

| # | predicts | rows | purity | rule |
| ---: | :--- | ---: | ---: | :--- |
| 1 | **wireframe** | 454 | 1.000 | `dir_angle_entropy <= 0.6715 and text_label_length > 0.068 and dir_flow_axis <= -0.4369 and arrows_per_node <= 4.1 and text_block_count <= 21.5` |
| 2 | **flowchart** | 215 | 1.000 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.068 and text_block_count > 18.5 and dir_axis_aligned > 0.9646` |
| 3 | **flowchart** | 75 | 0.987 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.0507 and text_block_count > 18.5 and dir_axis_aligned <= 0.9646 and line_to_curve > 0.5386 and conn_self_loops <= 11 and global_aspect > 0.6387` |
| 4 | **flowchart** | 74 | 1.000 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.068 and text_block_count > 18.5 and dir_axis_aligned <= 0.9646 and line_to_curve <= 0.5386 and text_area_frac > 0.1542 and edge_count > 15` |
| 5 | **flowchart** | 51 | 1.000 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.068 and text_block_count <= 18.5 and conn_self_loops > 5.5` |
| 6 | **state_machine** | 50 | 1.000 | `dir_angle_entropy > 0.6715 and text_area_frac <= 0.2775 and arrowhead_count <= 7.5 and global_bbox_fill > 0.6598` |
| 7 | **wireframe** | 38 | 0.974 | `dir_angle_entropy <= 0.6715 and text_label_length > 0.068 and dir_flow_axis <= -0.4369 and arrows_per_node <= 4.1 and text_block_count > 21.5 and global_aspect <= 1.1658` |
| 8 | **flowchart** | 33 | 1.000 | `dir_angle_entropy > 0.6715 and text_area_frac > 0.2775 and text_label_length <= 0.141 and global_ink_coverage <= 0.0776 and dir_axis_aligned > 0.5653` |
| 9 | **er_diagram** | 32 | 0.969 | `dir_angle_entropy <= 0.6715 and text_label_length > 0.0841 and dir_flow_axis > -0.4369 and global_ink_coverage > 0.0312 and arrowhead_count <= 4.5 and line_to_curve <= 2.1456 and text_per_node > 0.8651 and text_area_frac <= 0.6893 and global_aspect <= 1.3669` |
| 10 | **flowchart** | 28 | 0.929 | `dir_angle_entropy <= 0.6715 and text_label_length > 0.068 and dir_flow_axis > -0.4369 and global_ink_coverage > 0.0312 and arrowhead_count > 4.5 and global_aspect > 0.6871` |
| 11 | **flowchart** | 26 | 1.000 | `dir_angle_entropy <= 0.6715 and 0.0507 < text_label_length <= 0.068 and text_block_count > 18.5 and dir_axis_aligned <= 0.9646 and line_to_curve > 0.5386 and global_aspect > 0.7251 and layout_col_regularity is present` |
| 12 | **wireframe** | 25 | 1.000 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.068 and text_block_count <= 18.5 and conn_self_loops <= 5.5 and dir_flow_axis <= -0.6761` |
| 13 | **wireframe** | 25 | 0.960 | `dir_angle_entropy <= 0.6715 and text_label_length > 0.068 and dir_flow_axis > -0.4369 and global_ink_coverage <= 0.0312 and dir_axis_aligned <= 0.9242 and layout_nn_distance <= 0.2579` |
| 14 | **flowchart** | 15 | 1.000 | `dir_angle_entropy <= 0.6715 and 0.068 < text_label_length <= 0.0841 and dir_flow_axis > -0.4369 and global_ink_coverage > 0.0312 and arrowhead_count <= 4.5 and line_to_curve <= 1.2294` |
| 15 | **circuit** | 14 | 0.929 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.068 and text_block_count <= 18.5 and conn_self_loops <= 5.5 and dir_flow_axis > -0.6761 and conn_cycle_count <= 264 and 0.5524 < global_aspect <= 1.7778 and layout_nn_distance is present` |

## Rules for the minority classes

The three small classes are where the tree earns its class weighting, and where its rules are least trustworthy - a leaf covering four rows is a memorised page.

| predicts | rows | purity | rule |
| :--- | ---: | ---: | :--- |
| circuit | 14 | 0.929 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.068 and text_block_count <= 18.5 and conn_self_loops <= 5.5 and dir_flow_axis > -0.6761 and conn_cycle_count <= 264 and 0.5524 < global_aspect <= 1.7778 and layout_nn_distance is present` |
| circuit | 8 | 1.000 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.05 and text_block_count > 18.5 and dir_axis_aligned <= 0.9646 and line_to_curve <= 0.5386 and text_area_frac <= 0.1542 and conn_self_loops <= 10 and conn_cycle_count <= 53 and text_inside_share <= 0.232` |
| circuit | 5 | 1.000 | `dir_angle_entropy <= 0.6715 and text_label_length <= 0.068 and text_block_count <= 18.5 and conn_self_loops <= 5.5 and dir_flow_axis > -0.3648 and conn_cycle_count <= 264 and global_aspect > 0.5524 and layout_nn_distance is missing` |
| er_diagram | 32 | 0.969 | `dir_angle_entropy <= 0.6715 and text_label_length > 0.0841 and dir_flow_axis > -0.4369 and global_ink_coverage > 0.0312 and arrowhead_count <= 4.5 and line_to_curve <= 2.1456 and text_per_node > 0.8651 and text_area_frac <= 0.6893 and global_aspect <= 1.3669` |
| er_diagram | 10 | 0.900 | `dir_angle_entropy <= 0.6715 and text_label_length > 0.0841 and dir_flow_axis > -0.4369 and global_ink_coverage > 0.0312 and arrowhead_count <= 4.5 and line_to_curve <= 2.1456 and text_per_node > 0.8651 and text_area_frac <= 0.6893 and global_aspect > 1.3669 and shape_frac_diamond <= 0.2083` |
| er_diagram | 3 | 0.667 | `dir_angle_entropy <= 0.6715 and 0.0507 < text_label_length <= 0.068 and text_block_count > 18.5 and dir_axis_aligned <= 0.9646 and line_to_curve > 0.5386 and global_aspect > 0.7251 and layout_col_regularity is missing and text_area_frac <= 0.4344 and arrowhead_count > 10` |
| state_machine | 50 | 1.000 | `dir_angle_entropy > 0.6715 and text_area_frac <= 0.2775 and arrowhead_count <= 7.5 and global_bbox_fill > 0.6598` |
