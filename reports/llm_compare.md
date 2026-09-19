# Phase 12.2.7 - zero-shot vs few-shot vs LoRA (test)

`Qwen/Qwen2.5-Coder-7B-Instruct`, NF4, frozen 12.2.4 template (`dreamscript-codegen/v1`), greedy, all 162 test pairs. Few-shot: k=2 same-type train exemplars nearest in IR size, 96 train pairs excluded because their IR text also occurs in validation/test. `novel IR` = the 131 test pairs whose IR text never occurs in train (fa_bresler exercises repeat across writers).

| arm | contract | syntax | executes | functional pass@1 | functional (novel IR) | dropped nodes | invented/program | LoRA - arm, 95% CI | batch | ms/sample |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| zero-shot | 93.8% | 100.0% | 40.7% | 3.1% | 3.8% | 18.9% | 1.198 | (0.5988, 0.7469) | 8 | 8134.6 |
| few-shot (k=2) | 93.2% | 96.3% | 95.7% | 39.5% | 32.8% | 23.3% | 0.988 | (0.216, 0.3951) | 8 | 21962.3 |
| LoRA | 99.4% | 99.4% | 98.8% | 70.4% | 66.4% | 16.8% | 0.864 |  | 16 | 10740.1 |

**LoRA wins on functional pass@1: True**

![three-way](figures/p12_three_way.png)
