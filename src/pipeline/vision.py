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
from typing import Any, Protocol, cast

from src.assemble.corpus import WEIGHTS
from src.detect.classes import CLASSES

#: Boxes below this detector score are not nodes. 10.1.1's floor, kept so the pipeline and the
#: measured assembly see the same page.
MIN_SCORE = 0.25


class PageLike(Protocol):
    """What Phase 10 actually reads off a page.

    `corpus.Page` derives `image` from its split and name, so a page with pixels and no
    provenance cannot subclass it. This states the real requirement instead: the tracer touches
    `name`, `image` and `diagonal` and nothing else, which is why `LoosePage` works at all. The
    casts below are that fact made checkable rather than asserted in a docstring.
    """

    name: str

    @property
    def image(self) -> Path: ...

    @property
    def diagonal(self) -> float: ...


@dataclass(slots=True)
class LoosePage:
    """A page with pixels and no provenance. Satisfies `PageLike`."""

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


def detect_boxes(
    path: str | Path, min_score: float = MIN_SCORE, *, keep_arrowheads: bool = False
) -> list[dict]:
    """9.1's detector on one page, in the `{id, bbox, cls, score}` shape the tracer wants.

    **`keep_arrowheads` exists because dropping them silently broke routing.** The tracer does
    not want arrowheads, so they were filtered here - but 13.3's prior is *fitted* on the
    detector's full class histogram, in which the arrowhead count is the single most
    discriminative feature: a flowchart page averages 69.97 of them and a state machine
    essentially none. Routing the filtered list is therefore a train/serve skew, and it is not
    theoretical - golden page `hdbpmn__ex08_writer0096` has 92 boxes including 67 arrowheads and
    routes `flowchart` at 1.0, while the same page filtered has 24 boxes and routes
    `state_machine` at **0.9909**, straight past 13.5's gate and into a Python state machine
    emitted for a BPMN diagram.

    So the classifier is handed what the fit saw, and assembly is handed what the tracer wants.
    """
    prediction = _detector().predict(str(path), verbose=False, imgsz=1280)[0]
    out: list[dict] = []
    for index, row in enumerate(prediction.boxes):
        score = float(row.conf.item())
        name = CLASSES[int(row.cls.item())]
        if score < min_score or (name == "arrowhead" and not keep_arrowheads):
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


#: Which corpus each routed type stands in for, when the S5 stages ask a page where it came from.
#: A live page has no provenance, but `_orient`'s direction weights and `statelabels`' recogniser
#: are both fitted per corpus, so the routed type is the honest proxy: 13.3 has already decided
#: this is a state machine, and the only state-machine corpus is fa_bresler.
SOURCE_FOR_TYPE = {"state_machine": "fa_bresler", "flowchart": "hdbpmn"}


def without_arrowheads(boxes: list[dict]) -> list[dict]:
    """The tracer's view of a detection list. See `detect_boxes` on why routing keeps them."""
    return [b for b in boxes if str(b.get("cls")) != "arrowhead"]


def assemble_diagram(
    path: str | Path, boxes: list[dict], *, read_text: bool = False, kind: str = ""
) -> dict[str, Any]:
    """10.1's tracer over those boxes, plus the S5 stages, as a plain IR dict.

    **The three stages after the tracer are the ones S5 is actually scored with**, and the
    pipeline was not running any of them: `statelabels` for node text, `_prune_short_loops`, and
    `_orient` for edge direction. Composed, they are what moved S5's val median from 9.0 to 3.0,
    and without them this stage emitted nodes named `d000`-`d011` with no text at all - 0 of 295
    node labels on the 25 golden pages. They are gated on `kind` because both the direction
    weights and the recogniser are fitted per corpus and 13.3 has already decided the type.
    """
    from src.assemble import s5, tracing

    page = page_for(path)
    page.source = SOURCE_FOR_TYPE.get(kind, "loose")
    # Deliberate: `tracing` is annotated for `corpus.Page` but only reads `PageLike`'s three
    # members. The cast keeps that narrow claim visible instead of widening tracing's signature
    # for a caller Phase 10 was not written for.
    known = cast(Any, page)
    # Arrowheads reach this function now, because routing needs them; the tracer never did.
    boxes = without_arrowheads(boxes)
    diagram = tracing.to_diagram(known, boxes, tracing.trace(known, boxes))

    if page.source in s5.TEXT_SOURCES:
        # The label is inside the shape for a state machine, so the node box is the crop and no
        # text detector is involved. `nodetext` reads nothing here; see `statelabels`.
        from src.assemble import statelabels

        labels = statelabels.read_page(known, boxes)
        for node in diagram.nodes:
            if labels.get(node.id):
                node.text = labels[node.id]
    if page.source in s5.EDGE_TEXT_SOURCES:
        # The trigger is written *beside* the connector - the one place neither `statelabels`
        # (inside the shape) nor `nodetext` (owned by a node) looks. **S5 cannot see this
        # stage**: `irdiff` compares edges as endpoint-key sets and has no `edge_substitute`,
        # so a machine with every trigger correct scores identically to one with none. It is
        # wired here rather than there because what it changes is the emitted program - with no
        # triggers every transition is an epsilon transition and epsilon-closure collapses the
        # whole machine to one state.
        from src.assemble import edgelabels

        edgelabels.apply(page, diagram)

    if page.source != "loose":
        s5._prune_short_loops(known, diagram)
        s5._orient(known, diagram)

    if read_text:
        # Off by default: `nodetext` measures this as break-even at best on S5.
        from src.assemble.nodetext import assign, read_page

        labels = assign(read_page(known, boxes), boxes)
        for node in diagram.nodes:
            if labels.get(node.id):
                node.text = labels[node.id]
    # `to_diagram` returns an `ir.model.Diagram`; the pipeline caches and serialises plain
    # dicts, so the boundary converts once here rather than in every consumer.
    if hasattr(diagram, "to_dict"):
        return cast(dict[str, Any], diagram.to_dict())
    return cast(dict[str, Any], diagram)
