# Model Cards

One card per model that reaches the pipeline or the report. `TEMPLATE.md` is the form.

Phase 17.4 filled one card for each model the served pipeline runs, plus the S1 classifier that
the headline classification number refers to. The stage in each card comes from the registry
(`reports/model_registry.md`), which derives it from whether the model meets its criterion.

| Model | Card | Stage | Phase | Registry |
| :--- | :--- | :--- | :--- | :---: |
| CLIP ViT-B/32 + RBF SVM (S1) | [`clip_svm.md`](clip_svm.md) | classification | 5–6 | prod |
| Naive Bayes router | [`nb_router.md`](nb_router.md) | classification (served) | 7.2, 13.3 | — |
| YOLOv8n component detector (S2) | [`detector.md`](detector.md) | detection | 9.1 | prod |
| YOLOv8m-pose arrow model | [`arrow_pose.md`](arrow_pose.md) | assembly | 10.1.6 | staging (with S5) |
| TrOCR-large label OCR (S3) | [`ocr_trocr.md`](ocr_trocr.md) | OCR | 9.3 | staging |
| HMM role decoder (S4) | [`hmm_roles.md`](hmm_roles.md) | parsing | 7.3 | prod |
| Qwen2.5-Coder-7B QLoRA (S6, S7) | [`synth_lora.md`](synth_lora.md) | synthesis | 12 | prod |

### Planned in Phase 0, not shipped

These were trained and evaluated in their phases. The pipeline does not run them, so they get no
card, and the evidence for each is in its `contributing.md` row:
the classical DT/KNN/LogReg (5), the MLP (6.2), the polynomial SVM (6.3), the RF/boosting ensemble
(7.1), the GMM shape vocabulary (7.4) and the Q-learning traversal agent (11). The pipeline orders
nodes with a reading-order DFS because 11.2.10 measured that RL adds nothing to executability.
The CRNN+CTC OCR (9.3) was replaced by TrOCR.

## Rule

A model without a completed card does not go into `src/serve`, and its numbers do not go into
the final report. The card must name the exact `experiments/<timestamp>_<run_name>/`
directory that produced the reported metrics, so any number in the report can be traced back
to the run that generated it.
