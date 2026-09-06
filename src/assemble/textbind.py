"""Phase 10.1.2 - deciding which node a text box belongs to, in three measurable stages.

    python -m src.assemble.textbind

A page arrives from 9.1 as a bag of boxes and from 3.2 as a bag of text regions, and nothing
connects them. `Diagram.nodes[i].text` is the field this task has to fill, and the plan's line is
*containment, then nearest centroid, then Hungarian assignment* with a target of 0.90 binding
accuracy. Writing that as one function would produce one number; writing it as three stages that
each report what they resolved and how well is what makes the cascade's shape visible - and the
shape turns out to be the result.

## Ground truth, which is the hard half

There is no annotation saying "this ink region is that node's label". What there is, on every
held-out page, is a ground-truth node box. So a text box's true owner is *defined geometrically
from the ground-truth boxes*: the smallest ground-truth node whose box contains at least
`CONTAIN` of the text box's area, and **`None` - "no owner" - when no ground-truth node does**.
The no-owner class is not padding - it is the class an edge label falls into, so it is scored as a
class rather than discarded - but on this corpus it is a minority, 21.7% (below), because 3.2's
proposer finds far more fragments than it once did and most of them sit inside *some* node's box.

The binder never sees any of that. It is handed 9.1's *detected* boxes and 3.2's text boxes and
nothing else; its answer is a detected box, which is mapped back to a ground-truth node by IoU
>= `MATCH_IOU` only at scoring time. A binder that names a detection with no ground-truth match
is wrong, which is the honest treatment - a label bound to a hallucinated node is a wrong label.

Two limitations follow from the construction and both are real:

* **A label drawn outside its node is scored as no-owner.** hdBPMN writers routinely overflow a
  task box; the geometric truth calls that ink unowned, so a binder that correctly attaches it is
  penalised. This makes every number here a lower bound on semantic correctness.
* **The truth inherits 3.2's proposals.** `text_boxes` is a *proposer*, and on a 3-state
  automaton it proposes 38 regions. Most are stroke fragments, not text. They are kept, because
  the proposer is what the binder will actually be fed in Phase 10, but it means the corpus is
  dominated by regions that no node should claim.

## The three stages

    containment      a text box with >= CONTAIN of its area inside a node box belongs to it.
                     Nested boxes (a BPMN task inside a pool inside a lane) are resolved to the
                     **smallest** container, since the innermost box is the one whose label it is.
    nearest centroid the leftovers, to the nearest node centre within NEAR_TOL of the page
                     diagonal. Many-to-one: a node has as many text boxes as it has lines.
    hungarian        what neither stage claimed, as a global one-to-one optimum over
                     distance/diagonal with pairs beyond HUNG_TOL forbidden, via
                     `scipy.optimize.linear_sum_assignment`.

**Stage 3 is one-to-one and stage 2 is not, deliberately.** A node owns as many text boxes as its
label has lines, so a one-to-one rule applied to the whole page is wrong by construction - and
the `hungarian_only` control below measures exactly how wrong. One-to-one is defensible only on
the residual, where the leftovers are far from everything and the question is which of several
distant nodes gets the one remaining claim.

## An earlier run of this file measured a different corpus, and was wrong

A first pass through this docstring reported 22,586 text boxes over these same 470 pages - 48 a
page. The number below is **149,547 - 318 a page, 591 on hdbpmn** - and the two are not a tuning
difference, they are 6.6x apart. Both numbers came from `text_boxes`, which does nothing but call
3.2.7's `propose` through `load_or_compute`; the cache never returns a stale answer silently
(3.2.9's fingerprint check sees to that), so the earlier run was not reading corrupted primitives.
It was reading 3.2.7's proposer *before it was finished being tuned*. The per-page cache under
`data/interim/primitives_detect/` is now warm and fingerprint-current for all 470 pages - checked
here directly, not assumed - and recomputing any of them from the image gives back exactly what
the cache holds. **149,547 is the honest count of what 3.2.7 proposes today; 22,586 is what an
earlier, stricter version of that proposer found, scored under this task's name.** The number
below is the current one, all 470 pages, one clean run, nothing partial.

## What it measured

470 held-out pages, 149,547 text boxes proposed by 3.2, 7,811 node detections at score >= 0.25.

    stage             resolved    share    accuracy
    containment        120,394   0.8051      0.9223
    nearest centroid     8,823   0.0590      0.0339
    hungarian             1,466   0.0098     0.0280
    unbound (no owner)  18,864   0.1261      0.9260

    overall accuracy                         0.8616   <- target 0.90, NOT met

    control                       accuracy
    reject everything               0.2166
    always nearest centroid         0.4445
    largest containing box          0.4958
    hungarian only (one-to-one)     0.2438

**The target is missed at 0.8616, and every control that was supposed to be a weak, degenerate
baseline is now weaker than the cascade by a wide margin - rejecting everything scores 0.2166,
not the 0.94 an earlier run found.** That flip is not the cascade improving; it is the corpus.
78.3% of 3.2's (now much larger) pool of proposed text regions have a ground-truth owner, so
"reject everything" throws away the majority class instead of guessing it, and "accuracy over all
proposals" has stopped being a rejection metric in disguise - it is close to an honest binding
number already, which is why restricting to owned boxes barely moves it:

    stage                   owned boxes   correct    accuracy
    containment                 114,802     106,916     0.9313
    nearest centroid                845         140      0.1657
    hungarian                       111           6      0.0541
    missed entirely (unbound)     1,396           0      0.0000

    owner-only accuracy                                  0.9139
    largest containing box                               0.3677

**Owner-only accuracy (0.9139) is now higher than overall accuracy (0.8616), the reverse of the
earlier run, because the boxes with no owner - 21.7% of the corpus, mostly stroke fragments in
open space - are exactly the ones the cascade is worst at rejecting cleanly.** `unbound`'s own
stage accuracy (0.926) looks strong only because refusing an unowned box is trivially "correct";
the 1,396 owned boxes the cascade leaves unbound score 0 by definition, and that recall gap - not
containment's precision, which sits at 0.9313 - is what keeps overall accuracy under target.
Containment still resolves 80.5% of the corpus and is right 92.2% of the time it fires; the miss
is not in what fires, it is in what 3.2 proposes and 9.1 detects around it.

## Does the global optimum beat greedy? Now, yes, and by more than a rounding error

Stage 3 is where the plan's Hungarian assignment lives:

    stage-3 rule            resolved   stage accuracy   overall
    hungarian                  1,466           0.0280     0.8616
    greedy nearest (same tol) 10,036           0.0139     0.8093

**Greedy resolves 6.8x more of the residual than Hungarian does, and loses on both counts.** With
149,547 candidates the residual left to stage 3 is no longer 1% of the corpus - greedy's laxer
one-box-per-node-slot behaviour lets it grab 10,036 pairs where Hungarian's true one-to-one
constraint accepts only 1,466 - and even so Hungarian both resolves fewer wrong answers and costs
less overall accuracy (0.8616 against greedy's 0.8093, a 0.052 gap that used to be a rounding
error and is not one now). `STAGE3 = "hungarian"` stays the default because it is what the plan
specifies and it is now measurably the better choice, not just the cheaper one to defend.

The `hungarian_only` control shows what one-to-one costs applied to the *whole* page rather than
just the residual: 0.2438 against the cascade's 0.8616. Global optimality was never the missing
ingredient in text binding, before or after this corpus grew 6.6x; the missing ingredient is
knowing which regions are text at all, and 10.1.2 does not know that - 3.2's proposer does not
classify, it proposes.

## Per source

    source          pages     boxes   overall   owner-only   owned share
    hdbpmn            242   142,983    0.8603       0.9117        0.7989
    fa_bresler         96     4,974    0.9554       0.9968        0.3721
    flowchartseg      132     1,590    0.6849       0.9972        0.6767

hdbpmn supplies 95.6% of every text box in the corpus - the other two sources are rounding errors
next to it - so the headline numbers above are, in practice, hdbpmn's numbers. fa_bresler and
flowchartseg both post owner-only accuracy above 0.99: their proposers see far fewer boxes per
page (52 and 12 respectively, against hdbpmn's 591) and most of what they do propose sits cleanly
inside one detected node. flowchartseg's overall accuracy (0.6849) is the corpus's worst *despite*
that near-perfect owner-only number, because 32.3% of its proposals are unowned and unbound
scores every one of those wrong when a wrong control would call it right - a page-count artefact
of a small, sparse split, not a cascade weakness.

## The pool-contains-everything artefact: real, and now separated out

`true_owners` gives a text box's owner as the *smallest* ground-truth node containing it, and a
BPMN pool or lane is drawn once per page and covers nearly everything on it. A stroke fragment
sitting in open space inside a pool - not inside any task, not text at all - still gets an owner
for free if no smaller node claims it first, so a proposer that over-generates fragments could be
inflating `owned_share` for a reason that has nothing to do with binding. **It is checked, not
assumed: nodes covering more than `CONTAINER_AREA_FRAC` (30%) of the page are pulled out of owner
candidacy and the same run is scored again.** `owned_share` drops from 0.7834 to
**`owned_share_excl_containers` = 0.64** - containers really do manufacture roughly 14 points of
owned-share out of nothing, confirming the effect is real. But it is a minority contributor: only
18.3% of *all* owned boxes trace their ownership to a container, so most of the swing from the
earlier run's 29.1% owned share to this run's 78.3% is not pools, it is 3.2's now-larger proposer
finding far more fragments that sit inside ordinary, small node interiors - any shape's own box
trivially "owns" any box drawn inside it, container or not. Binding accuracy on the
container-excluded population is, if anything, slightly *better*
(`owner_only_accuracy_excl_containers` = 0.9344 against 0.9139 including them), because a
container's own interior is larger and messier than a single node's, so both numbers are reported
rather than picking the more flattering one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from src.assemble.corpus import (
    RUNS,
    Page,
    centre,
    contains_fraction,
    detections,
    distance,
    iou,
    pages,
    truth,
)
from src.preprocess.cache import load_or_compute
from src.utils.config import ROOT
from src.utils.parallel import pmap

PRIMITIVES = ROOT / "data" / "interim" / "primitives_detect"
OUT = RUNS / "textbind.json"

#: 9.1.2's class list minus the one class that is not a node region.
NOT_A_NODE = "arrowhead"

#: Detections below this score are dropped before binding. 9.1's corpus cache is floored at 0.05
#: so that assembly can make this choice itself; 0.25 is the detector's own default operating
#: point, kept rather than tuned so that binding is measured against the boxes Phase 10 ships.
SCORE_FLOOR = 0.25

#: Share of a text box's area that must fall inside a node box for stage 1 to claim it, and the
#: same threshold that defines the ground-truth owner. A label sits wholly inside its node when
#: the writer stayed in the lines, and 0.8 leaves room for the ones who did not quite.
CONTAIN = 0.8

#: Stage 2's distance tolerance, as a share of the page diagonal. Beyond this the "nearest" node
#: centre is not evidence of anything - on a 20-node page some node is always nearest.
NEAR_TOL = 0.05

#: Stage 3's, wider because the residual is by definition what stage 2's tolerance refused.
HUNG_TOL = 0.10

#: A predicted detection is credited with a ground-truth node's identity only above this IoU.
#: Scoring only; the binder never sees it.
MATCH_IOU = 0.5

#: A ground-truth node whose box covers more than this share of the page is a *container* - a
#: BPMN pool or lane, drawn once per page and containing almost every other node and every
#: stroke on it. Diagnostic only: `true_owners` still gives it credit (a label really is inside
#: its pool), but a container's near-universal reach means it hands out "ownership" to strokes
#: that are not labels at all, so `owned_share_excl_containers` reports what the share looks
#: like with containers removed from owner candidacy, and the gap between the two numbers is
#: measured rather than assumed away.
CONTAINER_AREA_FRAC = 0.3

#: The stage-3 rule actually used. "greedy" is supported so the claim that the global optimum
#: buys nothing stays checkable rather than asserted.
STAGE3 = "hungarian"

STAGES = ("containment", "nearest", "hungarian")


@dataclass(frozen=True)
class Binding:
    """One text box's answer: which node box, and which stage decided it."""

    text: int
    node: int | None
    stage: str


# -- the inputs, both of them as the binder sees them ---------------------------------------


def text_boxes(page: Page, directory: Path = PRIMITIVES) -> list[list[float]]:
    """3.2's proposed text regions for this page, in detector pixels."""
    primitives, _ = load_or_compute(page.name, page.image, directory)
    return [[float(v) for v in box] for box in primitives.text_boxes]


def node_boxes(page: Page, score_floor: float = SCORE_FLOOR) -> list[list[float]]:
    """The detector's node regions - every class but `arrowhead`, above the score floor."""
    return [
        list(row["bbox"])
        for row in detections(page)
        if row["cls"] != NOT_A_NODE and row["score"] >= score_floor
    ]


# -- the cascade ----------------------------------------------------------------------------


def containment(
    texts: list[list[float]], nodes: list[list[float]], threshold: float = CONTAIN
) -> dict[int, int]:
    """Stage 1. Text box -> the *smallest* node box containing `threshold` of its area.

    Smallest, because BPMN nests a task inside a lane inside a pool and all three contain the
    label. The innermost box is the one whose label it is; taking the largest instead is the
    `largest_containing` control, which is measured rather than dismissed.
    """
    out: dict[int, int] = {}
    for i, text in enumerate(texts):
        best, best_area = None, float("inf")
        for j, node in enumerate(nodes):
            area = max(node[2] * node[3], 0.0)
            if contains_fraction(text, node) >= threshold and area < best_area:
                best, best_area = j, area
        if best is not None:
            out[i] = best
    return out


def nearest(
    texts: list[list[float]],
    nodes: list[list[float]],
    indices: list[int],
    diagonal: float,
    tolerance: float,
) -> dict[int, int]:
    """Nearest node centre within `tolerance * diagonal`. Many-to-one, and unresolved beyond it."""
    out: dict[int, int] = {}
    limit = tolerance * diagonal
    for i in indices:
        here = centre(texts[i])
        best, best_distance = None, limit
        for j, node in enumerate(nodes):
            d = distance(here, centre(node))
            if d <= best_distance:
                best, best_distance = j, d
        if best is not None:
            out[i] = best
    return out


def hungarian(
    texts: list[list[float]],
    nodes: list[list[float]],
    indices: list[int],
    diagonal: float,
    tolerance: float,
) -> dict[int, int]:
    """Stage 3. A global one-to-one optimum over normalised centre distance.

    Pairs beyond `tolerance` are given a cost the solver will only pay if it must, and are then
    dropped from the answer - `linear_sum_assignment` has to return a complete matching on the
    smaller side, so forbidding a pair means filtering the result rather than the matrix.
    """
    if not indices or not nodes:
        return {}
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    cost = np.array(
        [[distance(centre(texts[i]), centre(n)) / diagonal for n in nodes] for i in indices]
    )
    rows, columns = linear_sum_assignment(cost)
    return {
        indices[r]: int(c) for r, c in zip(rows, columns, strict=True) if cost[r, c] <= tolerance
    }


def bind(
    texts: list[list[float]],
    nodes: list[list[float]],
    diagonal: float,
    *,
    threshold: float = CONTAIN,
    near_tol: float = NEAR_TOL,
    hung_tol: float = HUNG_TOL,
    stage3: str = STAGE3,
) -> list[Binding]:
    """The whole cascade, one `Binding` per text box, in text-box order.

    The public entry point. Nothing here reads ground truth; `nodes` are detections.
    """
    first = containment(texts, nodes, threshold)
    left = [i for i in range(len(texts)) if i not in first]
    second = nearest(texts, nodes, left, diagonal, near_tol)
    left = [i for i in left if i not in second]
    if stage3 == "greedy":
        third = nearest(texts, nodes, left, diagonal, hung_tol)
    else:
        third = hungarian(texts, nodes, left, diagonal, hung_tol)

    out: list[Binding] = []
    for i in range(len(texts)):
        if i in first:
            out.append(Binding(i, first[i], "containment"))
        elif i in second:
            out.append(Binding(i, second[i], "nearest"))
        elif i in third:
            out.append(Binding(i, third[i], "hungarian"))
        else:
            out.append(Binding(i, None, "unbound"))
    return out


# -- ground truth, and the map from a detection back to it ----------------------------------


def true_owners(
    texts: list[list[float]], gt_nodes: list[tuple[str, list[float]]]
) -> list[str | None]:
    """Each text box's true owner id, or `None` for the no-owner class.

    Defined by the same geometry stage 1 uses, but over **ground-truth** boxes. That symmetry is
    deliberate: it means a wrong answer is a wrong *box*, not a disagreement about what
    containment means.
    """
    out: list[str | None] = []
    for text in texts:
        best, best_area = None, float("inf")
        for node_id, box in gt_nodes:
            area = max(box[2] * box[3], 0.0)
            if contains_fraction(text, box) >= CONTAIN and area < best_area:
                best, best_area = node_id, area
        out.append(best)
    return out


def detection_identity(
    nodes: list[list[float]], gt_nodes: list[tuple[str, list[float]]], threshold: float = MATCH_IOU
) -> list[str | None]:
    """For each detected box, the ground-truth node it *is*, or `None` if it matches nothing.

    Greedy by IoU, one ground-truth node claimed once, so two overlapping detections cannot both
    be credited with the same node.
    """
    pairs = sorted(
        (
            (iou(box, gt_box), j, k)
            for j, box in enumerate(nodes)
            for k, (_, gt_box) in enumerate(gt_nodes)
        ),
        key=lambda row: -row[0],
    )
    out: list[str | None] = [None] * len(nodes)
    used_gt: set[int] = set()
    used_det: set[int] = set()
    for score, j, k in pairs:
        if score < threshold:
            break
        if j in used_det or k in used_gt:
            continue
        out[j] = gt_nodes[k][0]
        used_det.add(j)
        used_gt.add(k)
    return out


# -- scoring one page -----------------------------------------------------------------------


def _controls(
    texts: list[list[float]], nodes: list[list[float]], diagonal: float
) -> dict[str, list[int | None]]:
    """The baselines, each as a text-box-length list of node indices or `None`."""
    always = nearest(texts, nodes, list(range(len(texts))), diagonal, float("inf"))
    largest: dict[int, int] = {}
    for i, text in enumerate(texts):
        best, best_area = None, -1.0
        for j, node in enumerate(nodes):
            area = node[2] * node[3]
            if contains_fraction(text, node) >= CONTAIN and area > best_area:
                best, best_area = j, area
        if best is not None:
            largest[i] = best
    only = hungarian(texts, nodes, list(range(len(texts))), diagonal, HUNG_TOL)
    return {
        "reject_everything": [None] * len(texts),
        "always_nearest": [always.get(i) for i in range(len(texts))],
        "largest_containing": [largest.get(i) for i in range(len(texts))],
        "hungarian_only": [only.get(i) for i in range(len(texts))],
    }


def score_page(page: Page, stage3: str = STAGE3) -> dict:
    """Everything measurable about one page, as plain counters the caller sums."""
    texts = text_boxes(page)
    nodes = node_boxes(page)
    diagram = truth(page)
    gt_nodes = [(n.id, n.bbox) for n in diagram.nodes if n.bbox]

    page_area = max(page.size[0] * page.size[1], 1.0)
    container_ids = {
        node_id for node_id, box in gt_nodes if (box[2] * box[3]) / page_area > CONTAINER_AREA_FRAC
    }
    non_container_nodes = [(i, b) for i, b in gt_nodes if i not in container_ids]

    owners = true_owners(texts, gt_nodes)
    owners_excl = true_owners(texts, non_container_nodes)
    identity = detection_identity(nodes, gt_nodes)

    def predicted(index: int | None) -> str | None:
        return identity[index] if index is not None else None

    bindings = bind(texts, nodes, page.diagonal, stage3=stage3)
    per_stage = {
        stage: {"resolved": 0, "correct": 0, "owned": 0, "owned_correct": 0}
        for stage in (*STAGES, "unbound")
    }
    correct = owned = owned_correct = 0
    owned_excl = owned_correct_excl = 0
    for binding, owner, owner_excl in zip(bindings, owners, owners_excl, strict=True):
        row = per_stage[binding.stage]
        row["resolved"] += 1
        hit = predicted(binding.node) == owner
        row["correct"] += int(hit)
        correct += int(hit)
        if owner is not None:
            owned += 1
            owned_correct += int(hit)
            row["owned"] += 1
            row["owned_correct"] += int(hit)
        if owner_excl is not None:
            owned_excl += 1
            owned_correct_excl += int(predicted(binding.node) == owner_excl)

    controls = {}
    for name, guesses in _controls(texts, nodes, page.diagonal).items():
        hits = [predicted(g) == o for g, o in zip(guesses, owners, strict=True)]
        controls[name] = {
            "correct": int(sum(hits)),
            "owned_correct": int(
                sum(h for h, o in zip(hits, owners, strict=True) if o is not None)
            ),
        }

    return {
        "source": page.source,
        "texts": len(texts),
        "nodes": len(nodes),
        "gt_nodes": len(gt_nodes),
        "unmatched_detections": int(sum(1 for v in identity if v is None)),
        "owned": owned,
        "correct": correct,
        "owned_correct": owned_correct,
        "owned_excl_containers": owned_excl,
        "owned_correct_excl_containers": owned_correct_excl,
        "stages": per_stage,
        "controls": controls,
    }


# -- the run --------------------------------------------------------------------------------


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else float("nan")


def aggregate(rows: list[dict], target: float = 0.90) -> dict:
    texts = sum(r["texts"] for r in rows)
    owned = sum(r["owned"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    owned_correct = sum(r["owned_correct"] for r in rows)
    owned_excl = sum(r.get("owned_excl_containers", 0) for r in rows)
    owned_correct_excl = sum(r.get("owned_correct_excl_containers", 0) for r in rows)

    stages = {}
    for stage in (*STAGES, "unbound"):
        resolved = sum(r["stages"][stage]["resolved"] for r in rows)
        stages[stage] = {
            "resolved": resolved,
            "share": _ratio(resolved, texts),
            "accuracy": _ratio(sum(r["stages"][stage]["correct"] for r in rows), resolved),
            "owned": sum(r["stages"][stage]["owned"] for r in rows),
            "owned_correct": sum(r["stages"][stage]["owned_correct"] for r in rows),
        }

    controls = {
        name: {
            "accuracy": _ratio(sum(r["controls"][name]["correct"] for r in rows), texts),
            "owner_only": _ratio(sum(r["controls"][name]["owned_correct"] for r in rows), owned),
        }
        for name in rows[0]["controls"]
    }

    by_source = {}
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        sub_texts = sum(r["texts"] for r in subset)
        sub_owned = sum(r["owned"] for r in subset)
        by_source[source] = {
            "pages": len(subset),
            "texts": sub_texts,
            "owned_share": _ratio(sub_owned, sub_texts),
            "accuracy": _ratio(sum(r["correct"] for r in subset), sub_texts),
            "owner_only": _ratio(sum(r["owned_correct"] for r in subset), sub_owned),
        }

    accuracy = _ratio(correct, texts)
    return {
        "pages": len(rows),
        "text_boxes": texts,
        "node_detections": sum(r["nodes"] for r in rows),
        "unmatched_detections": sum(r["unmatched_detections"] for r in rows),
        "owned": owned,
        # The share that decides whether "accuracy" is a binding metric or a rejection metric.
        "owned_share": _ratio(owned, texts),
        "accuracy": accuracy,
        "owner_only_accuracy": _ratio(owned_correct, owned),
        # Diagnostic: the same two numbers with page-spanning containers (pools, lanes - see
        # CONTAINER_AREA_FRAC) taken out of owner candidacy, so a container handing out free
        # ownership to stroke fragments is visible rather than baked into the headline share.
        "owned_share_excl_containers": _ratio(owned_excl, texts),
        "owner_only_accuracy_excl_containers": _ratio(owned_correct_excl, owned_excl),
        "target": target,
        "target_met": bool(accuracy >= target),
        "stages": stages,
        "controls": controls,
        "by_source": by_source,
    }


def run(limit: int | None = None, n_jobs: int = 4) -> dict:
    held = pages()[:limit] if limit else pages()

    def one(page: Page) -> dict:
        return score_page(page, STAGE3)

    def one_greedy(page: Page) -> dict:
        return score_page(page, "greedy")

    rows = pmap(one, held, n_jobs=n_jobs, prefer="threads", desc="binding")
    result = aggregate(rows)

    greedy = aggregate(pmap(one_greedy, held, n_jobs=n_jobs, prefer="threads"))
    # The plan's own question: does the global optimum beat greedy nearest, and by how much?
    result["stage3_comparison"] = {
        "hungarian": {
            "resolved": result["stages"]["hungarian"]["resolved"],
            "stage_accuracy": result["stages"]["hungarian"]["accuracy"],
            "overall": result["accuracy"],
        },
        "greedy": {
            "resolved": greedy["stages"]["hungarian"]["resolved"],
            "stage_accuracy": greedy["stages"]["hungarian"]["accuracy"],
            "overall": greedy["accuracy"],
        },
        "global_optimum_beats_greedy": bool(result["accuracy"] > greedy["accuracy"]),
    }
    result["constants"] = {
        "score_floor": SCORE_FLOOR,
        "contain": CONTAIN,
        "near_tol": NEAR_TOL,
        "hung_tol": HUNG_TOL,
        "match_iou": MATCH_IOU,
        "stage3": STAGE3,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    result = run(args.limit, args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
