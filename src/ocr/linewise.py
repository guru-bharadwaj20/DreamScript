"""Line-wise OCR for wrapped diagram labels.

    python -m src.ocr.linewise --checkpoint finetune

The original CRNN emits one left-to-right CTC sequence per crop.  A BPMN node often contains two
or three stacked lines, so pooling its height destroys the reading order.  This decoder detects
the horizontal bands before recognition, reads each with the shared CRNN, then joins them in
top-to-bottom order.  It evaluates the exact unchanged validation rows used by Phase 9.3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.ocr.metrics import RUNS, score
from src.utils.config import ROOT

CHECKPOINTS = ROOT / "experiments" / "ocr" / "crnn"


def load_model(checkpoint: str = "finetune"):
    """Load a Phase 9.3 CRNN checkpoint and its frozen alphabet."""
    import torch

    from src.ocr.crnn import build_model

    weights = CHECKPOINTS / f"{checkpoint}.pt"
    alphabet = CHECKPOINTS / "alphabet.json"
    if not weights.is_file() or not alphabet.is_file():
        raise FileNotFoundError(f"missing CRNN checkpoint or alphabet for {checkpoint!r}")
    chars = json.loads(alphabet.read_text(encoding="utf-8"))["chars"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(chars) + 1).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model, chars, device


def segment(path: Path) -> list[np.ndarray]:
    """Return writing bands in reading order; a single-line crop stays a single item."""
    import cv2

    from src.ocr.crnn import text_lines

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    return text_lines(image)


def predict_files(
    files: list[str], checkpoint: str = "finetune", batch_size: int = 64
) -> list[str]:
    """Recognize every segmented line, then compose each crop's reading-order transcript."""
    import torch

    from src.ocr.crnn import fit, greedy_decode

    model, chars, device = load_model(checkpoint)
    owners, inputs = [], []
    outputs = [[] for _ in files]
    for index, name in enumerate(files):
        for band in segment(Path(name)):
            owners.append(index)
            inputs.append(fit(band))

    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = np.stack(inputs[start : start + batch_size])
            tensor = torch.from_numpy(batch).float().div_(255.0).sub_(0.5).unsqueeze(1).to(device)
            logits = model(tensor).log_softmax(-1).cpu().numpy()
            for owner, row in zip(owners[start : start + batch_size], logits, strict=True):
                outputs[owner].append(greedy_decode(row, chars))
    return [" ".join(part for part in lines if part).strip() for lines in outputs]


def run(checkpoint: str = "finetune") -> dict:
    from src.ocr.crnn import diagram_split

    files, truths, frame = diagram_split("val", provenance=("annotated", "derived", "detected"))
    predicted = predict_files(files, checkpoint)
    result = score(truths, predicted)
    RUNS.mkdir(parents=True, exist_ok=True)
    name = f"linewise_{checkpoint}"
    (RUNS / f"{name}.json").write_text(
        json.dumps(
            {
                "model": name,
                "split": "val",
                "files": frame["file"].tolist(),
                "predictions": predicted,
            }
        ),
        encoding="utf-8",
    )
    return {"model": name, "checkpoint": checkpoint, **result}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="finetune", choices=("iam", "finetune"))
    args = ap.parse_args(argv)
    try:
        result = run(args.checkpoint)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
