"""Phase 13 - running Phase 9 and 10 on a page that is not in the corpus.

Every assembly stage takes an `src.assemble.corpus.Page`, which knows its split, its source and
where its ground truth lives - all true of a corpus page and none of it true of a photograph
somebody just took. `LoosePage` is the same shape with none of the provenance: the pixels, the
size, and a name. The tracer only ever reads `image`, `diagonal` and `name`, so it does not
notice the difference.

Detection is run here rather than read from `corpus.detections`, which is a cache built over the
corpus's own held-out pages and has nothing to say about a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.assemble.corpus import WEIGHTS
from src.detect.classes import CLASSES

#: Boxes below this detector score are not nodes. 10.1.1's floor, kept so the pipeline and the
#: measured assembly see the same page.
MIN_SCORE = 0.25


@dataclass(slots=True)
class LoosePage:
    """A page with pixels and no provenance."""

    name: str
    image: Path
    native_size: tuple[int, int]
    scale: float = 1.0
    source: str = "loose"
    split: str = "none"
    id: str = ""

    @property
    def size(self) -> tuple[float, float]:
        w, h = self.native_size
        return (w * self.scale, h * self.scale)

    @property
    def diagonal(self) -> float:
        w, h = self.size
        return float((w * w + h * h) ** 0.5)


@lru_cache(maxsize=1)
def _detector(weights: str = str(WEIGHTS)):
    from ultralytics import YOLO

    return YOLO(weights)


def page_for(path: str | Path) -> LoosePage:
    import cv2

    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read an image from {path}")
    height, width = image.shape[:2]
    return LoosePage(name=path.stem, image=path, native_size=(width, height))


def detect_boxes(path: str | Path, min_score: float = MIN_SCORE) -> list[dict]:
    """9.1's detector on one page, in the `{id, bbox, cls, score}` shape the tracer wants."""
    prediction = _detector().predict(str(path), verbose=False, imgsz=1280)[0]
    out: list[dict] = []
    for index, row in enumerate(prediction.boxes):
        score = float(row.conf.item())
        name = CLASSES[int(row.cls.item())]
        if score < min_score or name == "arrowhead":
            continue
        x1, y1, x2, y2 = (float(v) for v in row.xyxy[0].tolist())
        out.append(
            {
                "id": f"d{index:03d}",
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "cls": name,
                "score": score,
            }
        )
    return out


def assemble_diagram(
    path: str | Path, boxes: list[dict], *, read_text: bool = False
) -> dict[str, Any]:
    """10.1's tracer over those boxes, as a plain IR dict."""
    from src.assemble import tracing

    page = page_for(path)
    diagram = tracing.to_diagram(page, boxes, tracing.trace(page, boxes))
    if read_text:
        # Off by default: `nodetext` measures this as break-even at best on S5.
        from src.assemble.nodetext import assign, read_page

        labels = assign(read_page(page, boxes), boxes)
        for node in diagram.nodes:
            if labels.get(node.id):
                node.text = labels[node.id]
    return diagram.to_dict() if hasattr(diagram, "to_dict") else diagram
