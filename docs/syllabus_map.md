# Syllabus map

Phase 17.2. For each unit requirement: the phase that covers it, the code, the figure, and the
number to quote. Every figure is in `reports/figures/`. Every number is copied from the
`contributing.md` row named in the Phase column, where the full measurement and its caveats are
recorded. The notebook for each unit (17.3) is in `notebooks/` and shows the same figures and
results inline.

## Unit 1 — supervised learning basics · [`notebooks/unit1_classical.ipynb`](../notebooks/unit1_classical.ipynb)

| Requirement | Phase | Code | Figure | Result to quote |
| :--- | :--- | :--- | :--- | :--- |
| Decision tree, KNN, logistic regression | 5.1 | `src/classify/tree.py`, `knn.py`, `linear.py`, `baselines.py` | `p5_tree.png` | macro F1 on 33 handcrafted features: logreg 0.7917, KNN 0.7614, tree 0.7434, against a majority baseline of 0.1237 (`reports/classification_report.md`) |
| Precision / recall / AUC, cross-validation | 5.2 | `src/classify/metrics.py`, `pr.py`, `roc.py`, `cv.py`, `grouped.py`, `calibration.py` | `p5_pr.png`, `p5_roc.png`, `p5_calibration.png`, `p5_confusion.png`, `p5_learning_curves.png` | cross-validation is grouped by scribe, so no writer is in both train and test folds |
| KNN decision boundaries | 5.3.1 | `src/classify/boundaries.py` | `p5_knn_boundaries.png` | 2-D projections cost half the model: best projected macro F1 0.583 against 0.764 on all 33 features |

## Unit 2 — neural networks and kernels · [`notebooks/unit2_ann_svm.ipynb`](../notebooks/unit2_ann_svm.ipynb)

| Requirement | Phase | Code | Figure | Result to quote |
| :--- | :--- | :--- | :--- | :--- |
| ANN / MLP on image embeddings | 6.2 | `src/classify/mlp.py`, `backprop.py`, `activations.py`, `torchnet.py` | `p6_schedules.png` | backprop is derived by hand in `docs/backprop_derivation.md` |
| Optimizer convergence comparison | 6.2.4 | `src/classify/optimizers.py` | `p6_optimizers.png` | the learning rate matters 20x more than the optimizer: 0.0173 spread across optimizers, 0.3390 across rates within SGD |
| SVM with polynomial kernel | 6.3.2 | `src/classify/svm.py`, `poly.py`, `rbf.py`, `supportvectors.py`, `surface.py` | `p6_kernel_surfaces.png`, `p6_support_vectors.png` | flowchart vs state machine is linearly separable (every cell 1.0000). On circuit vs flowchart the kernel gains +0.0438 |
| Shipped classifier (S1) | 5–6 | `src/classify/s1.py` | — | CLIP ViT-B/32 + one-vs-one RBF SVM: 0.9871 accuracy on held-out scribes, target 0.92 |

## Unit 3 — ensembles and probabilistic models · [`notebooks/unit3_probabilistic.ipynb`](../notebooks/unit3_probabilistic.ipynb)

| Requirement | Phase | Code | Figure | Result to quote |
| :--- | :--- | :--- | :--- | :--- |
| Random forest / gradient boosting | 7.1 | `src/classify/forest.py`, `boosting.py`, `adaboost.py`, `bagging.py`, `stacking.py`, `shapley.py` | `p7_shap_summary.png` | Gini and SHAP give the same ranking (Spearman 0.984). Permutation importance disagrees, and 20 of 58 columns have negative permutation importance |
| Naive Bayes | 7.2 | `src/classify/bayes.py`, `multinomial.py`, `prior.py`, `independence.py` | — | the NB prior over detected classes is the pipeline's router (`src/pipeline/routing.py`) |
| HMM + Viterbi | 7.3 | `src/parse/viterbi.py`, `baumwelch.py`, `transitions.py`, `emissions.py`, `s4.py` | `p7_hmm_transitions.png`, `p7_hmm_emissions.png`, `p7_baumwelch.png` | S4 role macro F1 0.8003 against a target of 0.80. The margin (0.0003) is smaller than the seed spread (0.0010) |
| GMM + EM | 7.4 | `src/parse/emfit.py`, `softshapes.py`, `continuous.py` | `p7_gmm_em.png`, `p7_gmm_selection.png`, `p7_gmm_montage.png`, `p7_scribe_covariance.png` | the mixture's ARI of 0.1499 is twice K-means' 0.0717 on the same rows (8.1) |

## Unit 4 — unsupervised, deep, reinforcement, LLMs, MLOps · [`notebooks/unit4_deep_rl_llm.ipynb`](../notebooks/unit4_deep_rl_llm.ipynb)

| Requirement | Phase | Code | Figure | Result to quote |
| :--- | :--- | :--- | :--- | :--- |
| K-means / hierarchical clustering | 8 | `src/cluster/kmeans.py`, `hierarchy.py`, `linkage.py`, `choosek.py`, `styles.py` | `p8_choose_k.png`, `p8_dendrogram.png`, `p8_style_clusters.png` | elbow and silhouette both choose K = 3, and at K = 3 the ARI against the shape labels is 0.0028. The clusters found are not shape types |
| CNN + parameter calculation | 9 | `src/detect/scratchcnn.py`, `cnnmath.py`, `train.py`, `s2.py` | `p9_filters.png`, `p9_featuremaps.png`, `p9_gradcam.png`, `p9_detection.png` | from-scratch CNN: 372,183 parameters, macro F1 0.9595. YOLO detector (S2): mAP@0.5 0.9107, target 0.80. The parameter arithmetic is in `docs/cnn_math.md` |
| Reinforcement learning (Q-learning) | 11 | `src/rl/qlearning.py`, `sarsa.py`, `dqn.py`, `env.py`, `reward.py`, `shaping.py` | `p11_convergence.png`, `p11_qvalues.png`, `p11_sarsa.png`, `p11_dqn.png`, `p11_ablation.png` | implemented and evaluated, not load-bearing. Removing RL costs nothing in executability (11.2.10) |
| LLM fine-tuning (LoRA) | 12 | `src/llm/train.py`, `sweep.py`, `quant.py`, `score.py`, `functional.py` | `p12_train_loss.png`, `p12_lora_sweep.png`, `p12_hparam_sweep.png`, `p12_three_way.png` | S6: 99.4% of generated programs execute. S7: 70.37% pass@1, measured from the IR, not from a photograph |
| MLOps | 15 | `src/mlops/tracking.py`, `registry.py`, `drift.py`, `monitor.py`, `retrain.py` | `p15_prediction_monitoring.png` | MLflow tracking on a file backend, a registry whose stage is derived from the S-criteria, and PSI/KS drift checks (`reports/drift.md`) |

## What is covered, and how strongly

All four units are implemented and evaluated. Two of the checkmarks carry a caveat, and it should
be given when the result is presented:

- **RL is not load-bearing.** A plain reading-order DFS generates code that runs just as often.
  The learned traversal adds +0.026 semantic accuracy from loop marks and costs 0.024–0.050 edge F1.
- **The LoRA numbers are IR → code.** From a photograph, the vision half fails first. S3 and S5
  miss their targets in `reports/master_results.md`.
