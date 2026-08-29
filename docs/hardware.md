# Hardware Profile & VRAM Budget

Phase 0.1.7. Every training decision in this project is sized against one GPU. This document
records what that GPU is, how much of it each phase may use, and what to do when a phase does
not fit.

## Target machine

| Property | Value |
| :--- | :--- |
| GPU | NVIDIA RTX 4500 Ada Generation |
| VRAM | 24 GB GDDR6 ECC (24,570 MiB reported) |
| Architecture | Ada Lovelace, compute capability **sm_89** |
| Driver | 595.95 |
| CUDA (torch build) | 12.4 |
| cuDNN | 9.1.0 |
| bf16 | Supported (verified by `scripts/check_cuda.py`) |
| TF32 | Available; left **off** by default for determinism (Phase 0.1.6) |
| OS | Windows 11 Pro 26200 |
| Python | 3.11.15 (`.venv`, created by `uv`) |
| PyTorch | 2.5.1+cu124 |

Reproduce these numbers at any time:

```
.venv/Scripts/python.exe scripts/check_cuda.py
```

## Per-phase VRAM budget

Budgets are the working set a phase is allowed to occupy, leaving ~2 GB headroom for the
display, allocator fragmentation and CUDA context. Anything above 22 GB must be redesigned,
not attempted.

| Phase | Workload | Precision | Batch | Budget | Headroom notes |
| :--- | :--- | :--- | :---: | ---: | :--- |
| 3 | Preprocessing, primitive extraction | CPU only | — | 0 GB | OpenCV/skimage are CPU-bound; parallelize over cores |
| 4 | Handcrafted features | CPU only | — | 0 GB | Pandas/NumPy |
| 5 | DT / KNN / LogReg | CPU only | — | 0 GB | scikit-learn, dataset fits in RAM |
| 6.1 | Frozen ResNet/CLIP embedding extraction | fp16 inference | 64 | ~3 GB | One pass, cached to disk |
| 6.2 | MLP on embeddings | fp32 | 128 | ~1 GB | Trivially small |
| 6.3 | SVM (kernel) | CPU only | — | 0 GB | O(n²) memory in RAM, not VRAM |
| 7 | RF / XGBoost / HMM / GMM | CPU only | — | 0 GB | XGBoost may use `device="cuda"` (~1 GB) if needed |
| 8 | K-means / hierarchical clustering | CPU only | — | 0 GB | scikit-learn |
| 9.1 | YOLOv8/RT-DETR detector fine-tune | AMP bf16 | 16 @ 1024px | **~10 GB** | Largest non-LLM job; drop to batch 8 if tiling at 1280px |
| 9.2 | Scratch CNN on shape crops | fp32 | 128 @ 64px | ~2 GB | Teaching/ablation model |
| 9.3 | CRNN+CTC OCR | AMP bf16 | 64 | ~4 GB | TrOCR baseline instead: ~8 GB |
| 11 | Q-learning / DQN traversal | fp32 | 64 | ~1 GB | Tabular variant is CPU-only |
| 12 | **QLoRA fine-tune, 7B base** | NF4 + bf16 compute | 1 × grad-accum 16 | **~18 GB** | Peak phase; see breakdown below |
| 12 | 7B inference (serving) | NF4 | 1 | ~6 GB | Leaves room for the detector to stay resident |
| 16 | Serving detector + OCR + LLM together | mixed | 1 | ~12 GB | All three models resident for the live demo |

### Phase 12 QLoRA breakdown (the binding constraint)

| Component | Estimate |
| :--- | ---: |
| 7B base weights, NF4 double-quantized | ~4.2 GB |
| LoRA adapters (r=32, attention + MLP projections) | ~0.3 GB |
| Adam optimizer state for adapters only (fp32 m, v) | ~0.7 GB |
| Activations, seq len 2048, batch 1, grad checkpointing on | ~6–9 GB |
| Gradients + fragmentation + CUDA context | ~2–3 GB |
| **Total** | **~15–18 GB** |

Levers, in the order to pull them if it OOMs:

1. Gradient checkpointing on (assumed above; costs ~30% throughput).
2. Sequence length 2048 → 1024. The serialized IR for a typical diagram is well under
   1024 tokens, so this is nearly free.
3. `paged_adamw_8bit` instead of AdamW (saves ~0.5 GB and survives momentary spikes).
4. LoRA rank 32 → 16, attention-only target modules.
5. Base model 7B → 3B (Qwen2.5-Coder-3B) as the documented fallback.

## Disk and RAM

| Resource | Requirement |
| :--- | :--- |
| System RAM | 32 GB recommended; kernel SVM and hierarchical clustering are the RAM peaks |
| Disk — datasets | ~40 GB (hdBPMN, FC-A/B, DIDI, IAM, Sketch2Code + self-collected photos) |
| Disk — model weights | ~25 GB (4-bit 7B, detector, OCR, embedding backbones) |
| Disk — experiments | ~20 GB (checkpoints, MLflow artifacts, augmented caches) |

## Determinism vs. throughput

`src/utils/seed.py` sets `cudnn.deterministic=True`, `cudnn.benchmark=False` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. This costs roughly 5–15% throughput on convolution-heavy
work (Phase 9) and is kept on for every reported number. Exploratory sweeps may pass
`deterministic=False`, but any result that reaches the report must be reproduced with it on.

## Escalation path

If a phase exceeds this budget, in order: reduce batch and accumulate gradients → shorten
sequences → reduce model size → move the job to CPU (classical models only) → drop the
variant and record why in the phase's ablation table. Nothing in this project is allowed to
depend on hardware the project does not have.
