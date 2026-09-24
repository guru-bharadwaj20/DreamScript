"""Phase 13.4 - the classifier fallback chain itself: handcrafted -> embedding -> CNN.

    python -m src.pipeline.chain --build     # train the rungs, write reports/p13_chain.*
    python -m src.pipeline.chain --check     # non-zero if the chain is worse than its first rung

`src.pipeline.fallback` establishes that the classifier rung is real - that a three-class model
on the rebuilt corpus is reading diagram type and not the renderer. This module builds the chain
that rung sits at the top of.

**The served pipeline does not call this, and that is a decision rather than an omission.**
`core._classify` uses `pipeline.routing`: a naive-Bayes prior over the detector's class
histogram, which costs nothing because the boxes are already in hand. Every rung here needs a
second pass over the page - handcrafted features, a CLIP embedding, a CNN forward - which is
three more model loads for a decision the histogram makes at 0.98. This module answers the
research question (do independent rungs recover pages the first one loses), and 15.5's drift
detector reuses its feature layout; the answer is reported, not served.

## Why three rungs and not one model three times

A fallback chain is only worth its complexity if the rungs fail *independently*. Three rungs
that read the same input fail together, and the second and third never fire on anything the
first got wrong. So the rungs here are separated by their input, not by their hyperparameters:

    1. handcrafted    38 shape, layout, connectivity and text statistics from `src.features`,
                      computed on the page's detected primitives. Fails when detection fails.
    2. embedding      CLIP ViT-B/32 over the raw page, projected to 128 components. Reads the
                      picture, never the primitives, so a detection failure does not reach it.
    3. CNN            a small convolutional net trained here on 128x128 grayscale pages. No
                      handcrafted feature, no pretrained backbone - the rung that still answers
                      when both the primitive extractor and the pretrained embedding are out of
                      their depth.

## Abstention, and where the thresholds come from

A rung hands down only when it is not confident. `tau` is the minimum top-class probability a
rung needs to answer, and it is **chosen on the validation split and reported on test**, which
is the only arrangement under which the reported numbers are not the thresholds' training score.
The search is a coarse grid; the chosen values and the grid are both in the report, because a
threshold picked from a grid and quoted without the grid is a number with no error bars.

The chain's claim is modest and checkable: it must be **at least as accurate as its first rung
alone**, on the test split, or it is complexity that buys nothing and `--check` fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.embed.cache import aligned
from src.pipeline.fallback import IDENTITY, classifier, load
from src.utils.config import ROOT

PCA128 = ROOT / "data" / "features" / "embeddings_pca128.npy"
REPORT_MD = ROOT / "reports" / "p13_chain.md"
REPORT_JSON = ROOT / "reports" / "p13_chain.json"
WEIGHTS = ROOT / "experiments" / "p13_chain_cnn.pt"

#: The grid each rung's abstention threshold is searched over, on validation only.
TAU_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

#: Page side in pixels for the CNN rung. Large enough that a box is still a box, small enough
#: that 3,054 pages fit in memory as float32 (3054 * 128 * 128 * 4 = 200 MB).
SIDE = 128


# -------------------------------------------------------------------------------------------
# rung 3: a small CNN over the raw page
# -------------------------------------------------------------------------------------------


def _page_tensor(paths: list[str]) -> np.ndarray:
    """Grayscale pages at SIDE x SIDE, letterboxed so the aspect ratio is not destroyed."""
    import cv2

    out = np.zeros((len(paths), SIDE, SIDE), dtype=np.float32)
    for i, path in enumerate(paths):
        image = cv2.imread(str(ROOT / path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        h, w = image.shape
        scale = SIDE / max(h, w)
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        # Letterboxed onto white, because the pages are ink on paper and padding with black
        # would put a hard edge where the page simply ends.
        canvas = np.full((SIDE, SIDE), 255, dtype=np.uint8)
        y0, x0 = (SIDE - resized.shape[0]) // 2, (SIDE - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        out[i] = canvas.astype(np.float32) / 255.0
    return out


class PageCNN:
    """Four convolutional blocks and a linear head. Deliberately small: this is the rung that
    runs when the other two have abstained, so it must be cheap and it must not be a second
    copy of the embedding rung."""

    def __init__(self, classes: list[str], epochs: int = 30, seed: int = 0):
        self.classes = classes
        self.epochs = epochs
        self.seed = seed
        self.net = None
        self.device = "cpu"

    def _build(self):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        blocks = []
        channels = [1, 16, 32, 64, 128]
        for a, b in zip(channels[:-1], channels[1:], strict=True):
            blocks += [
                nn.Conv2d(a, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        return nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, len(self.classes)),
        )

    def fit(self, images: np.ndarray, y: np.ndarray, batch: int = 64):
        import torch
        from torch import nn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = self._build().to(self.device)
        index = {c: i for i, c in enumerate(self.classes)}
        targets = torch.tensor([index[v] for v in y], dtype=torch.long)
        data = torch.from_numpy(images).unsqueeze(1)

        # Class weights, because flowchart is two thirds of the corpus and an unweighted loss
        # would find "always flowchart" a good local minimum on the rung that matters least.
        counts = np.array([(y == c).sum() for c in self.classes], dtype=np.float64)
        weight = torch.tensor(
            counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32
        )
        loss_fn = nn.CrossEntropyLoss(weight=weight.to(self.device))
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=3e-4, weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)

        generator = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=generator)
            for start in range(0, len(order), batch):
                pick = order[start : start + batch]
                xb = data[pick].to(self.device)
                yb = targets[pick].to(self.device)
                optimiser.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                optimiser.step()
            schedule.step()
        return self

    def predict_proba(self, images: np.ndarray, batch: int = 256) -> np.ndarray:
        import torch

        self.net.eval()
        data = torch.from_numpy(images).unsqueeze(1)
        out = []
        with torch.no_grad():
            for start in range(0, len(data), batch):
                logits = self.net(data[start : start + batch].to(self.device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)))


# -------------------------------------------------------------------------------------------
# the rungs, trained on train and applied to validation and test
# -------------------------------------------------------------------------------------------


def _features(frame: pd.DataFrame) -> np.ndarray:
    columns = [c for c in frame.columns if c not in IDENTITY and c not in ("domain", "group")]
    return frame[columns].to_numpy(dtype=float)


def _split_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: (frame["split"] == name).to_numpy() for name in ("train", "validation", "test")}


def rung_probabilities(frame: pd.DataFrame, cnn_epochs: int = 30) -> dict[str, Any]:
    """Each rung fitted on `train`, predicting probabilities for every row."""
    classes = sorted(frame["diagram_type"].unique())
    masks = _split_masks(frame)
    y = frame["diagram_type"].to_numpy()
    train = masks["train"]

    handcrafted = _features(frame)
    first = classifier().fit(handcrafted[train], y[train])

    embeddings, found = aligned(frame["id"].tolist(), drop_missing=False)
    if not found.all():  # pragma: no cover - aligned raises first
        raise KeyError("the embedding cache does not cover the feature table")
    pca = np.load(PCA128)[: len(frame)] if PCA128.is_file() else embeddings
    if len(pca) != len(frame):
        pca = embeddings
    # An SVC with a probability head, because the chain needs a calibrated-ish confidence to
    # decide whether to hand down, and a bare decision function is not one.
    second = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=10.0, probability=True, random_state=0)
    ).fit(pca[train], y[train])

    images = _page_tensor(frame["path"].tolist())
    third = PageCNN(classes, epochs=cnn_epochs).fit(images[train], y[train])

    def ordered(model_classes, proba):
        """Columns in `classes` order - a fold can leave a model with a different class order."""
        position = {c: i for i, c in enumerate(list(model_classes))}
        return np.stack(
            [proba[:, position[c]] if c in position else np.zeros(len(proba)) for c in classes],
            axis=1,
        )

    return {
        "classes": classes,
        "masks": masks,
        "y": y,
        "device": third.device,
        "probabilities": {
            "handcrafted": ordered(first.classes_, first.predict_proba(handcrafted)),
            "embedding": ordered(second.classes_, second.predict_proba(pca)),
            "cnn": ordered(third.classes, third.predict_proba(images)),
        },
    }


def apply_chain(fitted: dict, taus: dict[str, float], mask: np.ndarray) -> dict[str, Any]:
    """Run the chain over the rows in `mask`, recording which rung answered each one."""
    classes = np.array(fitted["classes"])
    order = ["handcrafted", "embedding", "cnn"]
    n = int(mask.sum())
    answered_by = np.array(["none"] * n, dtype=object)
    predicted = np.array([None] * n, dtype=object)

    pending = np.ones(n, dtype=bool)
    for rung in order:
        proba = fitted["probabilities"][rung][mask]
        confidence = proba.max(axis=1)
        choice = classes[proba.argmax(axis=1)]
        # The last rung never abstains: something has to answer, and "no label" is not a
        # fallback, it is a dropped page.
        accepts = pending & (confidence >= taus[rung] if rung != order[-1] else pending)
        predicted[accepts] = choice[accepts]
        answered_by[accepts] = rung
        pending &= ~accepts

    truth = fitted["y"][mask]
    per_rung = {}
    for rung in order:
        picked = answered_by == rung
        per_rung[rung] = {
            "answered": int(picked.sum()),
            "share": round(float(picked.mean()), 4) if n else 0.0,
            "accuracy": (
                round(float(accuracy_score(truth[picked], predicted[picked])), 4)
                if picked.any()
                else None
            ),
        }
    return {
        "n": n,
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "per_rung": per_rung,
        "taus": dict(taus),
    }


def choose_taus(fitted: dict) -> dict[str, Any]:
    """Grid-search the two thresholds on validation. Test is never consulted here."""
    validation = fitted["masks"]["validation"]
    best, best_score, searched = None, -1.0, 0
    for tau1 in TAU_GRID:
        for tau2 in TAU_GRID:
            taus = {"handcrafted": tau1, "embedding": tau2, "cnn": 0.0}
            score = apply_chain(fitted, taus, validation)["accuracy"]
            searched += 1
            # Ties go to the *lower* threshold pair: if handing down more often does not help,
            # the simpler chain that answers early is the one to keep.
            if score > best_score:
                best, best_score = taus, score
    return {
        "taus": best,
        "validation_accuracy": round(float(best_score), 4),
        "grid": list(TAU_GRID),
        "combinations_searched": searched,
        "note": "chosen on validation; every number reported on test is out of this search",
    }


def collect(cnn_epochs: int = 30) -> dict[str, Any]:
    frame = load()
    manifest = pd.read_parquet(ROOT / "data" / "processed" / "manifest.parquet")
    frame = frame.merge(manifest[["id", "path"]], on="id", how="left")
    frame = frame[frame["path"].notna()].reset_index(drop=True)

    fitted = rung_probabilities(frame, cnn_epochs=cnn_epochs)
    chosen = choose_taus(fitted)
    test = fitted["masks"]["test"]

    chain = apply_chain(fitted, chosen["taus"], test)
    # The bar the chain has to clear: its own first rung, answering everything alone.
    alone = apply_chain(fitted, {"handcrafted": 0.0, "embedding": 0.0, "cnn": 0.0}, test)
    classes = np.array(fitted["classes"])
    solo = {
        rung: round(
            float(
                accuracy_score(
                    fitted["y"][test], classes[fitted["probabilities"][rung][test].argmax(axis=1)]
                )
            ),
            4,
        )
        for rung in ("handcrafted", "embedding", "cnn")
    }
    return {
        "what": "13.4 - the handcrafted -> embedding -> CNN chain, measured on the test split",
        "corpus": {
            "rows": int(len(frame)),
            "classes": {k: int(v) for k, v in frame["diagram_type"].value_counts().items()},
            "splits": {k: int(v) for k, v in frame["split"].value_counts().items()},
        },
        "rungs": {
            "handcrafted": "38 shape/layout/connectivity/text statistics from the primitives",
            "embedding": "CLIP ViT-B/32 over the raw page, 128 PCA components, RBF SVC",
            "cnn": f"a 4-block CNN trained here on {SIDE}x{SIDE} grayscale pages"
            f" ({cnn_epochs} epochs, {fitted['device']})",
        },
        "each_rung_alone_on_test": solo,
        "thresholds": chosen,
        "chain_on_test": chain,
        "first_rung_alone_on_test": {
            "accuracy": alone["per_rung"]["handcrafted"]["accuracy"],
            "n": alone["n"],
        },
        "verdict": {
            "chain_at_least_as_good_as_its_first_rung": bool(
                chain["accuracy"] >= solo["handcrafted"] - 1e-9
            ),
            "chain_accuracy": chain["accuracy"],
            "first_rung_accuracy": solo["handcrafted"],
        },
    }


def render(result: dict) -> str:
    chain, solo = result["chain_on_test"], result["each_rung_alone_on_test"]
    lines = [
        "# Phase 13.4 - the classifier fallback chain",
        "",
        "Generated by `python -m src.pipeline.chain --build`.",
        "",
        "The three rungs are separated by **what they read**, not by their hyperparameters - a"
        " chain of three models over one input fails three times at once.",
        "",
        "| rung | input | alone on test |",
        "| :--- | :--- | ---: |",
    ]
    for rung, description in result["rungs"].items():
        lines.append(f"| {rung} | {description} | {solo[rung]:.4f} |")
    lines += [
        "",
        "## Abstention",
        "",
        f"Thresholds are searched over {result['thresholds']['grid']}"
        f" ({result['thresholds']['combinations_searched']} combinations) **on validation**,"
        f" reaching {result['thresholds']['validation_accuracy']:.4f} there. Test is never"
        " consulted by the search, so the test numbers below are out of sample for the"
        " thresholds as well as for the models.",
        "",
        f"Chosen: handcrafted hands down below **{chain['taus']['handcrafted']:.2f}**, embedding"
        f" below **{chain['taus']['embedding']:.2f}**. The CNN never abstains - something has to"
        " answer, and an unlabelled page is a dropped page, not a fallback.",
        "",
        "## The chain on test",
        "",
        "| rung | pages it answered | share | accuracy on those |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for rung, entry in chain["per_rung"].items():
        accuracy = f"{entry['accuracy']:.4f}" if entry["accuracy"] is not None else "-"
        lines.append(f"| {rung} | {entry['answered']} | {entry['share']:.4f} | {accuracy} |")
    verdict = result["verdict"]
    lines += [
        "",
        f"**Chain: {chain['accuracy']:.4f}** over {chain['n']} test pages, against"
        f" **{verdict['first_rung_accuracy']:.4f}** for the handcrafted rung answering everything"
        " alone."
        + (
            " The chain earns its complexity."
            if verdict["chain_at_least_as_good_as_its_first_rung"]
            else " **The chain does not earn its complexity and is not wired.**"
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = collect(cnn_epochs=args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["verdict"], indent=2))
    print(f"-> {args.out}")
    if args.check and not result["verdict"]["chain_at_least_as_good_as_its_first_rung"]:
        print("the chain is no better than its first rung", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
