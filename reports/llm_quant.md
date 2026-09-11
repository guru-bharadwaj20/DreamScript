# Phase 12.2.2 - NF4 QLoRA on the RTX 4500 Ada (24 GiB)

Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, 4-bit NF4 + double quantisation, bf16 compute, SDPA,
LoRA r=16 / alpha=32 on all seven linear projections (40,370,176 trainable parameters).
Runs: `experiments/llm/quant_grid.json` (`python -m src.llm.quant`), packed-row probe
(`src.llm.quant.packed_probe`, logs `experiments/llm/probe_iso_{nf4,bf16}.log`). Every cell is a
real training step: forward, backward, fused AdamW step; first step dropped as warm-up.
Process VRAM ceiling 20 GiB (`quant.VRAM_CAP_GIB`), so crossing the cliff raises OOM instead of
spilling into WDDM shared memory.

## Flags read back from the loaded objects

| check | value |
| :--- | :--- |
| `Linear4bit` modules / quant type | 196 / nf4 |
| weights on device after load | 5.60 GB |
| attention implementation | sdpa |
| SDPA kernels that execute on this torch 2.5.1 Windows build | flash **no** (compiled out), mem-efficient yes, cuDNN yes, math yes |
| `flash-attn` / `xformers` installed | no / no |
| gradient checkpointing flag | true |
| gradient checkpointing *effective* (flag and `model.training`) | true only after `model.train()` |

**The eval-mode trap, measured**: `from_pretrained` returns an eval-mode model and HF applies
checkpointing only when `self.training`, so the flag reads `True` while nothing is checkpointed.
At 512 tokens: batch 1 **12.48 GB** in eval mode vs **7.16 GB** in train mode; batch 2 **19.08 GB**
vs **8.54 GB**. The profile now calls `model.train()` and records `gradient_checkpointing_effective`.

## Fit grid (random ids, full-sequence logits; tokens/s, peak allocated GB)

| arm | 1024 x1 | 1024 x2 | 1024 x4 | 1024 x8 | 2048 x1 | 2048 x2 | 2048 x4 | 4096 x1 | 4096 x2 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 base, ckpt | 968 / 8.21 | 989 / 10.96 | 938 / 15.80 | OOM | 967 / 10.30 | 918 / 15.80 | OOM | 888 / 14.47 | OOM |
| bf16 base, no ckpt | 1366 / 18.71 | OOM | | | OOM | | | OOM | |
| peft `prepare_model_for_kbit_training` (fp32 upcast), ckpt | 336 / 10.59 | 347 / 12.91 | 344 / 17.53 | OOM | 335 / 12.89 | 254 / 17.53 | OOM | 246 / 17.48 | OOM |

* **Throughput saturates at micro-batch 1** at every length: batch 2 at 1024 is +2.2%, batch 4 is
  -3.1%. The step is compute-bound (4-bit dequantisation per matmul), so a larger micro-batch buys
  VRAM and nothing else; effective batch comes from gradient accumulation.
* **peft's k-bit helper is 2.9x slower** (336 vs 968 tokens/s at 1024). It casts every non-4-bit
  parameter - including the 1.09 B-parameter embedding and `lm_head` - and the layer norms to
  float32, so hidden states leave each norm in fp32 and attention runs in fp32. Rejected; the loop
  keeps the base in bf16 and trains fp32 LoRA master weights under bf16 autocast.
* **No checkpointing is 1.41x faster but only fits 1024 x1** (18.71 GB, 1.3 GB under the cap).
  Real pairs reach 3,696 tokens, so it cannot train the corpus. Rejected.
* The memory that grows with length is the logits: 152,064 x L, upcast to fp32 in the loss.

## The configuration 12.2.5 trains with (real packed rows)

`train.selected_loss` applies `lm_head` only at supervised (completion) positions and packs
examples first-fit-decreasing with a block-diagonal causal mask and per-segment position ids.

| base | bin | rows (655 train pairs, hdbpmn+fa) | fill | tokens/s | peak alloc / reserved GB |
| :--- | ---: | ---: | ---: | ---: | ---: |
| NF4 | 1536 | 646 | 0.851 | 1,178 | 8.67 / 9.74 |
| NF4 | **2048** | 512 | 0.871 | 916 | **8.41 / 10.83** |
| NF4 | 3072 | 319 | 0.958 | 881 | 9.27 / 12.42 |
| bf16 (unquantised) LoRA | 1024 | 655 | 0.981 | 1,048 | 18.36 / 21.01 |
| bf16 (unquantised) LoRA | 2048 | 512 | 0.871 | 981 | 18.10 / 20.61 |

* **Isolation verified**: three examples packed in one row score **0.7206** against **0.7191**
  scored apart (0.2%, bf16 noise); the same row under a plain causal mask scores **2.4969**, so the
  mask is load-bearing. A CPU test pins the equality exactly on a tiny Qwen2.
* **Fits with headroom**: at the chosen 2,048-token bin peak reserved is **10.83 GB of 24 GiB**,
  i.e. ~13 GB free with the desktop (0.6 GB) and a concurrent 3 GB job accounted for. Selected-
  positions loss cut peak from 10.30 GB (full logits, 2048 x1) to 8.41 GB.
* **Unquantised bf16 LoRA was measured and rejected**: at 2048 it is 981 vs 916 tokens/s (+7%)
  for 18.10 GB allocated / 20.61 GB reserved - past the 20 GiB cap, no room for the 3 GB job.
* Throughput per token falls with row length on the mem-efficient kernel with an explicit mask
  (1536: 1,178; 3072: 881), but a 1536 bin barely packs (646 rows for 655 examples). 2048 is kept
  as the bin; examples longer than the bin get their own row uncut (97 of 655), none truncated.
