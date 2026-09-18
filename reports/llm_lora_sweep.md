# Phase 12.2.3 - LoRA configuration sweep

Base `Qwen/Qwen2.5-Coder-7B-Instruct`, NF4, LR 2e-4, cosine, 1,200,000 training tokens per arm (same seed, so the same packed rows in the same order), validation loss over the 176 validation pairs' supervised tokens. alpha = 2r throughout.

| arm | overrides | val loss @0 | best val loss | best step | steps | tokens | wall min | peak alloc GB | peak reserved GB | tokens/s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| lora8 | lora.r=8 lora.alpha=16 | 0.35647 | 0.08617 | 37 | 37 | 1206048 | 22.0 | 10.294 | 14.607 | 914.0 |
| lora16 | lora.r=16 lora.alpha=32 | 0.35647 | 0.06814 | 37 | 37 | 1206048 | 21.9 | 10.64 | 14.628 | 917.0 |
| lora32 | lora.r=32 lora.alpha=64 | 0.35647 | 0.05269 | 37 | 37 | 1206048 | 22.0 | 11.269 | 14.995 | 912.0 |
| lora64 | lora.r=64 lora.alpha=128 | 0.35647 | 0.04293 | 37 | 37 | 1206048 | 22.1 | 12.562 | 16.664 | 908.0 |
| lora16attn | lora.r=16 lora.alpha=32 lora.targets=[q_proj,k_proj,v_proj,o_proj] | 0.35647 | 0.13089 | 37 | 37 | 1206048 | 16.7 | 10.133 | 14.324 | 1206.0 |

**Selected: lora64** (lowest best validation loss).

![curves](figures/p12_lora_sweep.png)
