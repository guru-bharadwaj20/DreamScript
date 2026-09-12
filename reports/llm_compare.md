# Phase 12.2.7 - zero-shot vs few-shot vs LoRA (test)

`Qwen/Qwen2.5-Coder-7B-Instruct`, NF4, frozen 12.2.4 template (`dreamscript-codegen/v1`), greedy, all 162 test pairs. Few-shot: k=2 same-type train exemplars nearest in IR size, 96 train pairs excluded because their IR text also occurs in validation/test. `novel IR` = the 131 test pairs whose IR text never occurs in train (fa_bresler exercises repeat across writers).

| arm | contract | syntax | executes | functional pass@1 | functional (novel IR) | dropped nodes | invented/program | LoRA - arm, 95% CI | batch | ms/sample |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| zero-shot | 93.8% | 100.0% | 40.7% | 3.1% | 3.8% | 18.9% | 1.198 | (0.0, 0.0) | 8 | 8119.8 |
| few-shot (k=2) | 95.1% | 94.4% | 93.8% | 24.7% | 25.9% | 22.8% | 0.877 | (-0.2901, -0.1481) | 8 | 20588.9 |
| LoRA | 93.8% | 100.0% | 40.7% | 3.1% | 3.8% | 18.9% | 1.198 |  | 8 | 8124.1 |

**LoRA wins on functional pass@1: False**

![three-way](figures/p12_three_way.png)
