"""Phase 9.3.5 - the per-style hook, built the way 8.7 said it should have been.

    python -m src.ocr.adapt
    python -m src.ocr.adapt --epochs 6

8.6 clustered 105 hdbpmn writers into three style groups and proved they are about the hand
rather than the exercise. 8.7 spent them on per-style OCR heads and the answer was **no**: style
routing cost 31.3% relative CER against a single global head, and it lost to a random partition
of the same shape by a further 6.0%.

That result came with a diagnosis, and this task is that diagnosis tested rather than repeated.
8.7's closing paragraph named three things that would have to be tried before concluding style
conditioning cannot work, and said which one its own numbers most directly recommended:

    "per-cluster *adaptation* from the global head rather than independent fine-tuning from the
     pretrained one, so each head inherits all 7,599 crops of training before specialising"

**That removes the 23.9% arithmetic penalty** 8.7 measured with its `random` arm - the cost of
training each head on a third of the data - and leaves only the question 8.6 actually asked. So:

    global        9.3.2's `finetune` CRNN, unchanged. Every crop routes to it.
    style_adapt   three copies of that model, each given a few more epochs on **one 8.6
                  cluster's** writers. A val crop routes by its page's writer's cluster.
    random_adapt  **the control.** Three copies, each adapted on a random writer group of the
                  *same size*, routed the same way.

Every arm starts from the same weights and every adapted head has already seen all 16,974
training crops, so `style_adapt - global` is now the cost of specialisation alone and
`style_adapt - random_adapt` is, as in 8.7, what the clustering itself contributed.

## Why the answer can now be different from 8.7's, and why it might not be

Different, because the arithmetic penalty is gone: a head that has already been trained on
everything and is then nudged toward one style is not a head trained on a third of the data.

Possibly not different, because the other half of 8.7's mechanism survives intact: **the split is
by writer**, so every test hand is unseen, and a head narrowed onto twelve writers' habits is
narrowed away from the unseen writer it will actually meet. Adaptation reduces how far it can
narrow; it does not change the direction.

This is run on the CRNN rather than on TrOCR because adaptation from a shared checkpoint needs
several full copies of the model, and 3.8M parameters makes that cheap where 62M does not - and
because 9.3.3 establishes which of the two is the recommendation in the first place.

## What it measured

4,530 validation crops, three clusters, 86.09% of crops routable (the rest are fa_bresler's
writers, who have no 8.6 style and fall back to the global head so that every arm is scored on
identical crops). Six epochs of adaptation per head at lr 1e-4 from the shared checkpoint.

    arm              CER      WER    exact
    style_adapt   0.6816   0.9503   0.0656
    random_adapt  0.6818   0.9520   0.0669
    global        0.6864   0.9496   0.0660

    style_adapt - global         -0.0048 CER   (-0.70% relative)
    random_adapt - style_adapt   +0.0002 CER

**8.7's diagnosis was half right, and this is the half that was.** 8.7 measured per-style OCR
heads costing **+31.3% relative CER** and named adaptation-from-the-global-head as the thing to
try. Doing it removes the catastrophe entirely: **-31.3% becomes -0.70%**, a small improvement
instead of a large regression. So the 23.9% that 8.7's `random` arm attributed to *dividing the
training set* was correctly attributed - it disappears exactly when the division does.

**The other half of 8.7's finding survives untouched, and it is the one that decides the task.**
`random_adapt` - three heads adapted on random writer groups of the same sizes - scores
**0.6818 against style_adapt's 0.6816. The clustering contributes 0.0002 CER**, which is two ten-
thousandths and is not a result. 8.7 found the clusters *worse* than random by 6.0%; with the
arithmetic penalty removed they are neither better nor worse. **Grouping writers by how they
write is worth nothing here**, and the whole of the 0.0048 improvement is explained by "a few
more epochs of training on a subset", which an arbitrary subset delivers just as well.

**So the hook is built, integrated and measured, and the recommendation is not to route.** The
plan's row asks for integration and it exists - `routing()` reads 8.6's assignment, a crop is
routed by its page's writer, and an unrouted writer falls back - but nothing should be conditioned
on the cluster, because the same gain is available from any partition and from simply training
longer.

**The mechanism 8.7 identified is still the explanation.** The split is by writer, so every test
hand is unseen. Adaptation narrows a head toward its cluster's habits, which is narrowing away
from the unseen writer it will meet; adaptation narrows less far than independent fine-tuning did,
which is why the penalty shrank from 31.3% to nothing, and it narrows in the same direction, which
is why the clustering still earns nothing. **Three phases have now asked this corpus to support a
specialised model - 7.4.7's per-scribe densities, 8.7's per-style heads, and this - and the answer
has been the same every time, for the same reason: there is not enough of it.**
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from src.ocr.metrics import RUNS, score

SEED = 42
EPOCHS = 6
LR = 1e-4

#: Writers 8.6 has no style for - fa_bresler's 25, and any hdbpmn writer missing from the style
#: table - route to the global head. Falling back rather than dropping them keeps every arm
#: scored on exactly the same crops, which is what makes the CERs comparable at all.
UNROUTED = -1


def routing() -> tuple[dict[str, int], dict[str, int]]:
    """8.6's `scribe -> cluster`, and a random partition of the same group sizes."""
    from src.ocr.styled import random_groups, style_groups

    style = {str(k): int(v) for k, v in style_groups().items()}
    return style, {str(k): int(v) for k, v in random_groups(style, SEED).items()}


def adapt_heads(
    groups: dict[str, int],
    base_state,
    chars: str,
    device,
    epochs: int,
    lr: float,
    tag: str,
) -> dict[int, object]:
    """One copy of the global model per group, given `epochs` more on that group's crops only."""
    from src.ocr.crnn import Batches, build_model, diagram_split, train

    files, texts, frame = diagram_split("train")
    scribes = frame["scribe"].astype(str).to_numpy()
    table = {c: i + 1 for i, c in enumerate(chars)}

    heads: dict[int, object] = {}
    for cluster in sorted(set(groups.values())):
        mask = np.array([groups.get(s, UNROUTED) == cluster for s in scribes])
        if mask.sum() < 32:
            continue
        model = build_model(len(chars) + 1).to(device)
        model.load_state_dict(copy.deepcopy(base_state))
        subset_files = [f for f, keep in zip(files, mask, strict=True) if keep]
        subset_texts = [t for t, keep in zip(texts, mask, strict=True) if keep]
        train(
            model,
            Batches(subset_files, subset_texts, table),
            epochs,
            device,
            lr=lr,
            tag=f"{tag}{cluster}",
        )
        heads[cluster] = model
        print(f"[{tag}] cluster {cluster}: {int(mask.sum())} crops", flush=True)
    return heads


def route_and_predict(heads, global_model, groups, files, scribes, chars, device) -> list[str]:
    """Each crop decoded by its writer's head, or by the global head when it has none."""
    import torch

    from src.ocr.crnn import Batches, greedy_decode

    assignment = np.array([groups.get(str(s), UNROUTED) for s in scribes])
    predictions: list[str] = [""] * len(files)
    for cluster in sorted(set(assignment.tolist())):
        model = heads.get(cluster, global_model)
        index = np.flatnonzero(assignment == cluster)
        subset = [files[i] for i in index]
        batches = Batches(subset, [""] * len(subset), {}, shuffle=False)
        model.eval()
        out: list[str] = []
        with torch.no_grad():
            for x, _, _, _ in batches:
                logits = model(x.to(device)).log_softmax(-1).cpu().numpy()
                out.extend(greedy_decode(logits[i], chars) for i in range(logits.shape[0]))
        for position, prediction in zip(index, out, strict=True):
            predictions[position] = prediction
    return predictions


def run(epochs: int = EPOCHS, lr: float = LR, model_name: str = "finetune") -> dict:
    import copy as _copy

    from src.ocr.crnn import diagram_split, load

    global_model, chars, device = load(model_name)
    base_state = _copy.deepcopy(global_model.state_dict())

    style, random = routing()
    files, texts, frame = diagram_split("val", provenance=("annotated", "derived", "detected"))
    scribes = frame["scribe"].astype(str).to_numpy()

    arms: dict[str, list[str]] = {}
    arms["global"] = route_and_predict({}, global_model, {}, files, scribes, chars, device)
    for name, groups in (("style_adapt", style), ("random_adapt", random)):
        heads = adapt_heads(groups, base_state, chars, device, epochs, lr, name)
        arms[name] = route_and_predict(heads, global_model, groups, files, scribes, chars, device)
        del heads

    RUNS.mkdir(parents=True, exist_ok=True)
    scores = {}
    for name, predictions in arms.items():
        scores[name] = score(texts, predictions)
        (RUNS / f"adapt_{name}.json").write_text(
            json.dumps(
                {
                    "model": f"adapt_{name}",
                    "split": "val",
                    "files": frame["file"].tolist(),
                    "predictions": predictions,
                }
            ),
            encoding="utf-8",
        )

    routed = float(np.mean([str(s) in style for s in scribes]))
    return {
        "epochs": epochs,
        "lr": lr,
        "base_model": model_name,
        "clusters": len(set(style.values())),
        "routable_share": round(routed, 4),
        "arms": scores,
        # 8.7's two differences, recomputed with the arithmetic penalty removed.
        "specialisation_cost": round(scores["style_adapt"]["cer"] - scores["global"]["cer"], 4),
        "clustering_contribution": round(
            scores["random_adapt"]["cer"] - scores["style_adapt"]["cer"], 4
        ),
        "relative_vs_global": round(
            (scores["style_adapt"]["cer"] - scores["global"]["cer"])
            / max(scores["global"]["cer"], 1e-9),
            4,
        ),
        "best": min(scores, key=lambda k: scores[k]["cer"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--model", default="finetune")
    ap.add_argument("--out", type=Path, default=RUNS / "adapt.json")
    args = ap.parse_args(argv)

    result = run(args.epochs, args.lr, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
