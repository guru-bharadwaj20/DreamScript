"""Phase 17.3 - write the four per-unit notebooks.

    python scripts/make_notebooks.py            # write notebooks/unit*.ipynb
    python scripts/make_notebooks.py --execute  # and run them, so the outputs are stored

The notebooks read results. They do not reproduce them: every number comes from a committed
file under `reports/`, and every plot from `reports/figures/`. They open on a fresh clone with no
corpus, no GPU and no weights, which is the point of a notebook a reader opens first. The command
that regenerates each artefact is in `reports/README.md`.

Generated rather than hand-edited so that the four stay the same shape, and so that a figure
renamed in `reports/figures/` fails here rather than as a broken image in a notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = """\
import json
from pathlib import Path
from IPython.display import Markdown, display

ROOT = Path.cwd()
while not (ROOT / "reports").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
CELLS = json.loads((ROOT / "reports" / "master_results.json").read_text(encoding="utf-8"))["cells"]


def results(*stages):
    \"\"\"The master table's rows for these stages, as a Markdown table.\"\"\"
    rows = ["| stage | model | metric | value | target | verdict |", "| :-- | :-- | :-- | --: | --: | :-: |"]
    for c in CELLS:
        if c["stage"] in stages and c.get("available", True):
            verdict = "" if c.get("passes") is None else ("pass" if c["passes"] else "**fail**")
            target = "" if c.get("target") is None else c["target"]
            rows.append(f"| {c['stage']} | {c['model']} | {c['metric']} | {c['value']} | {target} | {verdict} |")
    display(Markdown("\\n".join(rows)))


def report(name, *keys):
    \"\"\"Selected top-level fields of one reports/*.json.\"\"\"
    data = json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))
    return {k: data[k] for k in keys if k in data}
"""

# unit -> (file stem, title, intro, [(heading, text, figures, code)])
UNITS: dict[int, tuple[str, str, str, list[tuple[str, str, list[str], str]]]] = {
    1: (
        "unit1_classical",
        "Unit 1 — decision tree, KNN, logistic regression",
        "Phase 5. Diagram-type classification from 33 handcrafted features (Phase 4), under "
        "cross-validation grouped by scribe so that no writer appears in both train and test.",
        [
            (
                "The three classifiers (5.1)",
                "Macro F1 against a majority baseline. Source: `reports/classification_report.md`.",
                ["p5_tree.png"],
                'results("classify")',
            ),
            (
                "Precision, recall, ROC and calibration (5.2)",
                "",
                [
                    "p5_pr.png",
                    "p5_roc.png",
                    "p5_calibration.png",
                    "p5_confusion.png",
                    "p5_learning_curves.png",
                ],
                "",
            ),
            (
                "KNN decision boundaries (5.3.1)",
                "Projecting to two dimensions costs half the model: the best projected macro F1 is "
                "0.583, against 0.764 on all 33 features. The figure shows where the classes sit, "
                "not how the model separates them.",
                ["p5_knn_boundaries.png"],
                "",
            ),
        ],
    ),
    2: (
        "unit2_ann_svm",
        "Unit 2 — MLP, optimizers, SVM kernels",
        "Phase 6. A hand-derived MLP (`docs/backprop_derivation.md`), an optimizer study and a "
        "polynomial-kernel SVM. The shipped S1 classifier is an RBF SVM on CLIP embeddings.",
        [
            (
                "S1 — held-out-scribe accuracy",
                "",
                [],
                'report("s1_heldout_scribes.json", "model", "accuracy", "accuracy_std", "macro_f1", "minimum_repeat_accuracy", "passes")',
            ),
            (
                "Optimizers and schedules (6.2.4)",
                "The learning rate matters about 20x more than the optimizer: the spread across five "
                "optimizers is 0.0173, and the spread across learning rates within SGD alone is 0.3390.",
                ["p6_optimizers.png", "p6_schedules.png"],
                "",
            ),
            (
                "Polynomial kernel and support vectors (6.3.2)",
                "Flowchart vs state machine is linearly separable: every cell scores 1.0000. On "
                "circuit vs flowchart, the kernel gains +0.0438.",
                ["p6_kernel_surfaces.png", "p6_support_vectors.png"],
                "",
            ),
        ],
    ),
    3: (
        "unit3_probabilistic",
        "Unit 3 — ensembles, Naive Bayes, HMM, GMM",
        "Phase 7. Ensembles and feature importance, the Naive Bayes prior that routes the "
        "pipeline, the HMM that labels node roles (S4), and the GMM shape vocabulary.",
        [
            (
                "Feature importance (7.1)",
                "Gini and SHAP give the same ranking (Spearman 0.984). Permutation importance, the one "
                "measured on held-out rows, disagrees with both.",
                ["p7_shap_summary.png", "p7_occlusion.png"],
                "",
            ),
            (
                "HMM + Viterbi — S4 (7.3)",
                "The margin over the target is 0.0003 and the seed spread is 0.0010. The target is "
                "met, not beaten.",
                ["p7_hmm_transitions.png", "p7_hmm_emissions.png", "p7_baumwelch.png"],
                'results("parse")\nreport("s4_role_labelling.json", "sequences", "nodes", "macro_f1", "macro_f1_std", "seeds_clearing_target", "seeds")',
            ),
            (
                "GMM + EM (7.4)",
                "",
                [
                    "p7_gmm_em.png",
                    "p7_gmm_selection.png",
                    "p7_gmm_montage.png",
                    "p7_scribe_covariance.png",
                ],
                "",
            ),
        ],
    ),
    4: (
        "unit4_deep_rl_llm",
        "Unit 4 — clustering, CNNs, reinforcement learning, LoRA, MLOps",
        "Phases 8, 9, 11, 12 and 15.",
        [
            (
                "Clustering (8)",
                "Elbow and silhouette both choose K = 3. At K = 3 the ARI against the shape labels is "
                "0.0028, so the structure they find is not shape type.",
                ["p8_choose_k.png", "p8_dendrogram.png", "p8_style_clusters.png"],
                "",
            ),
            (
                "CNN — filters, feature maps, detection (9)",
                "The from-scratch CNN has 372,183 parameters (the arithmetic is in "
                "`docs/cnn_math.md`). The YOLO detector is S2.",
                ["p9_filters.png", "p9_featuremaps.png", "p9_gradcam.png", "p9_detection.png"],
                'results("detect", "ocr")',
            ),
            (
                "Q-learning traversal (11)",
                "Implemented and evaluated, but not load-bearing: removing RL costs nothing in "
                "executability (11.2.10).",
                [
                    "p11_convergence.png",
                    "p11_qvalues.png",
                    "p11_sarsa.png",
                    "p11_dqn.png",
                    "p11_ablation.png",
                ],
                "",
            ),
            (
                "QLoRA code synthesis (12)",
                "S6 and S7 are measured from the IR to code, not from a photograph.",
                ["p12_train_loss.png", "p12_lora_sweep.png", "p12_three_way.png"],
                'results("codegen", "assemble")',
            ),
            (
                "MLOps (15)",
                "",
                ["p15_prediction_monitoring.png"],
                'report("drift.json", "what", "thresholds", "verdict")',
            ),
        ],
    ),
}


def build(unit: int) -> tuple[Path, nbformat.NotebookNode]:
    stem, title, intro, sections = UNITS[unit]
    cells = [
        new_markdown_cell(
            f"# {title}\n\n{intro}\n\nThis notebook reads committed results from `reports/`. "
            "It reproduces nothing and needs no corpus or GPU. `reports/README.md` names the "
            "command that regenerates each file. The full syllabus mapping is in "
            "`docs/syllabus_map.md`."
        ),
        new_code_cell(SETUP),
    ]
    for heading, text, figures, code in sections:
        for name in figures:
            if not (ROOT / "reports" / "figures" / name).exists():
                raise SystemExit(f"{stem}: reports/figures/{name} does not exist")
        cells.append(new_markdown_cell(f"## {heading}" + (f"\n\n{text}" if text else "")))
        if code:
            cells.append(new_code_cell(code))
        # Figures as Markdown image links, not executed outputs: they render on GitHub and in
        # Jupyter alike, and the notebooks do not carry 7 MB of base64 copies of files already
        # in the repository.
        for name in figures:
            cells.append(
                new_markdown_cell(
                    f"![{name}](../reports/figures/{name})\n\n`reports/figures/{name}`"
                )
            )
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    return OUT / f"{stem}.ipynb", nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute", action="store_true", help="run each notebook and store its outputs"
    )
    args = parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    for unit in sorted(UNITS):
        path, nb = build(unit)
        if args.execute:
            from nbclient import NotebookClient

            NotebookClient(nb, timeout=120, resources={"metadata": {"path": str(OUT)}}).execute()
        nbformat.write(nb, path)
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
