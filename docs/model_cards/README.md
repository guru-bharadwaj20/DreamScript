# Model Cards

One card per model that reaches the pipeline or the report. `TEMPLATE.md` is the form.

| Model | Card | Stage | Owning phase | Status |
| :--- | :--- | :--- | :--- | :---: |
| Classical diagram-type classifier (DT / KNN / LogReg) | `classical_clf.md` | classification | 5 | ❌ |
| MLP on image embeddings | `mlp_embed.md` | classification | 6.2 | ❌ |
| SVM (polynomial kernel) | `svm_poly.md` | classification | 6.3 | ❌ |
| Ensemble (RF / gradient boosting) | `ensemble_clf.md` | classification | 7.1 | ❌ |
| Naive Bayes text-stat prior | `nb_prior.md` | classification | 7.2 | ❌ |
| HMM role decoder | `hmm_roles.md` | parsing | 7.3 | ❌ |
| GMM shape vocabulary | `gmm_shapes.md` | parsing | 7.4 | ❌ |
| Component detector (CNN) | `detector.md` | detection | 9.1 | ❌ |
| Handwriting OCR (CRNN+CTC) | `ocr_crnn.md` | OCR | 9.3 | ❌ |
| Q-learning traversal agent | `rl_traversal.md` | traversal | 11 | ❌ |
| QLoRA code synthesizer (7B) | `synth_lora.md` | synthesis | 12 | ❌ |

## Rule

A model without a completed card does not go into `src/serve`, and its numbers do not go into
the final report. The card must name the exact `experiments/<timestamp>_<run_name>/`
directory that produced the reported metrics, so any number in the report can be traced back
to the run that generated it.
