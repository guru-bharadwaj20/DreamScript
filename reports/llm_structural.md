# Phase 12.3.4 - structural fidelity of LoRA test outputs

`src.eval.structural.fidelity` over 162 test generations: an AST-vs-graph comparison per diagram type, where everything expected is derived from the `ir_text` the model was shown and never from the IR file, so the metric cannot reward reproducing information the model could not see.

| diagram type | n | LoRA structural | LoRA parsed | reference structural |
| :--- | :--- | :--- | :--- | :--- |
| flowchart | 114 | 0.9627 | 99.1% | 0.8873 |
| state_machine | 48 | 0.9891 | 100.0% | 0.683 |

**Overall structural fidelity 0.9705** against a reference ceiling of 0.8267.
