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

from src.assemble.corpus import CONF_FLOOR, IMGSZ, WEIGHTS
from src.detect.classes import CLASSES

#: Boxes below this detector score are not nodes. 10.1.1's floor, kept so the pipeline and the
#: measured assembly see the same page. It is `tracing.MIN_SCORE`, and it applies to the *tracer's*
#: view only - see `detect_boxes` on why the router is handed everything above `CONF_FLOOR`.
MIN_SCORE = 0.25

#: The detector's inference settings, imported from `assemble.corpus` rather than restated.
#: They were restated, and drifted: this file ran the detector at `imgsz=1280`, which is
#: `src.detect.arrows.IMGSZ` - the *arrow* model's size - while 13.3's prior is fitted on
#: `corpus.detections`, produced at `corpus.IMGSZ = 896` with `conf=CONF_FLOOR`. A prior fitted on
#: one frame and applied to another is a train/serve skew, and it is the same class of defect
#: `keep_arrowheads` below exists to describe. Now there is one definition of the frame.
DETECT_IMGSZ = IMGSZ
DETECT_IOU = 0.7


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
    path: str | Path, min_score: float = CONF_FLOOR, *, keep_arrowheads: bool = False
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

    **`min_score` defaults to `CONF_FLOOR`, not to `MIN_SCORE`, for the same reason.**
    `routing.fit` histograms `corpus.detections(page)`, which is the unfiltered list down to
    0.05; filtering to 0.25 here handed the fitted prior a different distribution from the one it
    was fitted on. The 0.25 floor is the *tracer's*, and `tracer_boxes` applies it there - exactly
    where `tracing.node_boxes` applies it on the corpus path.
    """
    prediction = _detector().predict(
        str(path), verbose=False, imgsz=DETECT_IMGSZ, conf=CONF_FLOOR, iou=DETECT_IOU
    )[0]
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


def tracer_boxes(boxes: list[dict]) -> list[dict]:
    """`tracing.node_boxes`' rule, applied to a live page's detections.

    Arrowheads are not nodes, and neither is anything below `MIN_SCORE` - the corpus path drops
    both in `node_boxes`, and this is the same two lines against a list that came from the
    detector directly rather than from the cache.
    """
    return [b for b in without_arrowheads(boxes) if float(b.get("score", 0.0)) >= MIN_SCORE]


def arrow_detector_available() -> bool:
    """Whether 10.1.6's pose checkpoint is on this machine.

    `arrows.apply` raises `FileNotFoundError` without it. On the corpus path that is correct -
    a measurement with the arrow stage silently off is a wrong measurement - but a served page
    should answer with traced edges rather than a 502, so the stage is gated here and the IR
    records which one ran.
    """
    from src.detect.arrows import BEST

    return Path(BEST).is_file()


def assemble_diagram(
    path: str | Path, boxes: list[dict], *, read_text: bool = False, kind: str = ""
) -> dict[str, Any]:
    """`s5.assemble` over a live page's detections, as a plain IR dict.

    **This used to be a hand-rolled subset of `s5.assemble` and it left out the two stages that
    move the number most.** It ran `statelabels`, `edgelabels`, `_prune_short_loops` and
    `_orient`, and not `arrows.apply` or `pagetext.apply` - both of which `s5.score_page` has on
    by default, and which are worth median GED 20.5 -> 16.0 with pass share 0.2284 -> 0.2654
    (arrow edges) and hdbpmn median 27.0 -> 25.0 (page text). Every published S5 figure therefore
    described a composition the pipeline did not run. It calls the same function now.

    Two deliberate differences from `score_page`, both stated rather than left to be inferred:

        edge_text   on here, off there. S5 compares edges as endpoint-key sets and cannot see an
                    edge label at all, so the measurement is indifferent to it - but the emitted
                    *program* is not: with no triggers every transition is an epsilon transition
                    and epsilon-closure collapses the whole machine to one state.
        arrow_edges gated on the checkpoint being present, because a served page must degrade to
                    traced edges rather than fail. See `arrow_detector_available`.
    """
    from src.assemble import s5

    page = page_for(path)
    page.source = SOURCE_FOR_TYPE.get(kind, "loose")
    # Deliberate: the S5 stages are annotated for `corpus.Page` but only read `PageLike`'s three
    # members plus `source`/`id`. The cast keeps that narrow claim visible instead of widening
    # their signatures for a caller Phase 10 was not written for.
    known = cast(Any, page)
    arrow_edges = arrow_detector_available()
    diagram = s5.assemble(
        known,
        # Arrowheads and weak boxes both reach this function, because routing needs the whole
        # histogram down to `CONF_FLOOR`; the tracer wants neither.
        boxes=tracer_boxes(boxes),
        state_text=True,
        arrow_edges=arrow_edges,
        page_text=True,
        edge_text=True,
        direct=True,
        prune_loops=True,
        text=read_text,
    )
    # `to_diagram` returns an `ir.model.Diagram`; the pipeline caches and serialises plain
    # dicts, so the boundary converts once here rather than in every consumer.
    plain = cast(dict[str, Any], diagram.to_dict() if hasattr(diagram, "to_dict") else diagram)
    meta = plain.setdefault("meta", {})
    if isinstance(meta, dict):
        # Which composition answered, on the document itself. Without this an IR assembled with
        # traced edges and one assembled with arrow edges are indistinguishable after the fact.
        meta["assembled_with"] = "arrow_edges" if arrow_edges else "traced_edges"
    return plain
