"""Phase 10.2.6 - confidence propagation, and the number 9.3.7 already warned this row about.

    python -m src.assemble.propagate

Three upstream stages each leave a number behind: 9.1's detector a box `score`, 9.3.7's OCR a
`path_logp` posterior on a node's text, 7.3.8's HMM a forward-backward posterior on a node's
role. A node touched by all three has three confidences and the IR has one field. This is the
rule that turns three into one, for both `Node.confidence` and `Edge.confidence` - and the
measurement of whether that rule is worth anything beyond the free number already sitting in the
detector's own output.

## The warning this row inherits

9.3.7 measured a single confidence - OCR's `path_logp` - and found **AUROC 0.9497 with ECE
0.8121**: a good ranking and a useless probability, because a CTC path is confident about the
path it walked, not about whether that path is the label. Its conclusion was explicit: *"the
number is a rank and never a probability."*

Propagation multiplies that problem by three. `det`, `ocr` and `hmm` are each miscalibrated in
their own way and for their own reason - a YOLO objectness score, a CTC path posterior, an HMM
state marginal - and nothing about combining them cancels any of the three miscalibrations. A
product of three overconfident numbers is a more overconfident fourth number, not a corrected
one. So every rule below is scored on **both** axes, separately, exactly as 9.3.7 insisted on:
AUROC (does it rank a wrong node/edge below a right one) and ECE (does the number it produces
mean what it says).

## The controls, and the one that must not lose

    detector-alone   the box `score` on its own. Free - no OCR, no HMM, no propagation logic -
                     and it is what 9.3.7's `length` control was for text: any aggregation that
                     does not beat it has added nothing.
    random           a shuffle of the winning rule's scores. AUROC 0.5 by construction.

## The four rules

    product          det * ocr * hmm over whatever components a node actually has. Treats them
                     as independent probabilities, which is the assumption 9.3.7 says is false
                     for even one of the three alone - kept as the naive baseline every other
                     rule has to beat.
    minimum          the weakest signal wins. A node perfectly detected but unreadable text is
                     not a confident node.
    weighted_mean     det weighted 0.5, ocr and hmm 0.25 each, renormalised over whichever
                     components are present. Detection is what every node has; OCR and HMM only
                     annotate a minority (text-bearing nodes, and hdbpmn/fa_bresler's labelled
                     roles), so the weights are not a claim about importance, only a default that
                     does not let a missing signal silently zero out the score the way `product`
                     would.
    noisy_or         `1 - prod(1 - c)`. A node is confident if *any* signal is confident - the
                     opposite bet from `minimum`, kept because which bet is right is exactly what
                     is not obvious in advance.

A missing component is left out of every rule rather than imputed, because 9.3.7's lesson is that
inventing a number is worse than reporting fewer of them.

## Edges: no independent detector, so the endpoints stand in for one

Nothing upstream of this row detects an edge on its own - 9.1's YOLO boxes nodes, and the
tracer that turns strokes into edges belongs to a module this task does not depend on. So an
edge's raw signal *is* its two endpoints' node confidences, and the same four rules are re-used
one level up: `min(conf_src, conf_dst)`, `product`, `weighted_mean` (equal weight, there is no
detector/OCR/HMM asymmetry between two nodes), `noisy_or`. The control is the mean of the two
endpoints' detector scores alone - the free number an edge could have without any of this file.
Ground truth for "wrong" is a synthetic pair set per page: every real edge is a positive, and an
equal number of sampled non-adjacent node pairs are negatives, seeded per page so the set is
reproducible.

## The population, and a mistake caught before this row was committed

The first pass through this file scored the *raw* detection list - all 24,385 boxes at 9.1's
`CONF_FLOOR = 0.05`, of which **16,192 (66.4%) are class `arrowhead`**: 9.1.2's derived class,
which is never a node and cannot match a ground-truth node by construction. Scoring that
population made "is this node wrong" almost exactly "is this an arrowhead", and every rule's
AUROC clustered at 0.96-0.999 for that reason rather than because confidence propagation is that
good. The fix, made before any number below was reported, is to score `10.1.1`'s *assembled*
nodes - `nodes.build(page).diagram.nodes` - which have already dropped arrowheads, applied
`MIN_SCORE = 0.30` and de-duplicated across classes. That is the population 10.3's codegen and
Phase 13's review queue actually see, and it is the one "wrong" was always meant to describe.

## What it measured

470 held-out pages, **7,729 assembled nodes** (10.1.1's own count) - only **2.08% are wrong** by
IoU 0.5 or shape, against 68.84% on the discarded raw-detection population. A much rarer, harder
target, and the margins below are the real measurement:

    node score       AUROC     ECE
    detector-alone   0.9203   0.0818   <- the free control every rule must beat
    product          0.6805   0.2467
    minimum          0.7382   0.2018
    weighted_mean    0.8756   0.1082
    noisy_or         0.9300   0.0203   <- best ranking
    random           0.4811      -

**`noisy_or` wins the ranking and beats the free detector score by 0.0097 AUROC.** That margin
survives the correction - it is smaller than the 0.0034 the contaminated run reported, but this
time the two numbers are actually measuring the same thing: on 7,729 real assembled nodes rather
than a population two-thirds arrowhead. OCR and HMM cover 1,940 and 4,720 of the 7,729 (25% and
61% - both *higher* shares than before, since arrowheads carried neither signal and diluted the
denominator); where a node has either extra signal, `noisy_or` can move it, and where it has
neither, `noisy_or` **is** the detector score.

**`product` and `minimum` are actively worse than doing nothing.** Both score *below* the free
detector control (0.6805 and 0.7382 against 0.9203) - the opposite of 9.3.7's finding, where every
model score at least matched its cheap control. The reason is visible in `product`'s own
reliability bins: its *lowest*-confidence decile (mean 0.2755) is **97.93% correct**, because one
merely-mediocre OCR or HMM posterior on an otherwise-perfect detection drags the product down
without the node being wrong at all. Multiplying independent-seeming signals punishes a node for
having *any* imperfect component, and on a population that is 98% correct that is punishing the
common case for the wrong reason.

**Calibration and ranking still disagree, the way 9.3.7 said they would, but not in the same
direction as before.** `noisy_or` is now both the best-ranking *and* the best-calibrated rule
(ECE 0.0203, against the detector-alone control's own 0.0818) - which is not a contradiction of
9.3.7's lesson, it is the same lesson from the other side: `noisy_or`'s bins are non-monotonic
(0.9335 confidence -> 99.48% correct, immediately followed by 0.9506 -> 99.74%, then 0.9608 ->
99.87%, 0.9708 -> 100%, 0.9812 -> 99.48% again), swinging by a fraction of a point around a
population that is 97.9%+ correct almost everywhere above its bottom decile. A low ECE here comes
from every bin sitting near the same high accuracy, not from the number resolving fine
differences within that band - so **`noisy_or`'s number is still a rank first**: it separates the
2.08% that are wrong from the 97.92% that are not, and the reliability curve's flatness above that
point is a property of how lopsided the population is, not evidence the score resolves confidence
*within* the correct class.

**Edges are the negative result, unchanged by the node-population fix.** With no independent edge
detector to measure against, an edge's raw signal is only its two endpoints' node confidences
(now computed from the corrected, arrowhead-free node population), scored on 8,975 synthetic
pairs (every real edge as a positive, a sampled non-adjacent pair as a negative, per page):

    edge score               AUROC
    endpoint-detector-alone  0.5514
    product                  0.5518
    minimum                  0.5580   <- best ranking, and barely
    weighted_mean            0.5514
    noisy_or                 0.5474
    random                   0.5037

**Every edge rule is within 0.007 AUROC of chance plus the free control.** `minimum` "wins" by
0.0066 over the endpoint-average baseline - not a result worth building on. A node's own
confidence carries almost no information about whether *a pair of two* nodes is really connected,
because connectivity is a property of the (undetected) edge, not of the nodes at its ends. The
honest conclusion is that edge confidence needs its own upstream signal - something 10.2's edge
tracer would have to supply - and until it exists, endpoint confidence is a placeholder that
should not be shown to a user as meaning anything about the edge itself. This arm was not
re-derived hunting for a positive margin; it was already honestly negative and stays that way.

**The IR write-back is demonstrated on 5 pages without touching the shared ground-truth corpus**:
`annotate_diagram` loads each page fresh through `Diagram.load` (not `corpus.truth()`'s rescaled
copy, and not a hand-rolled dict), matches against `nodes.build(page)`'s assembled nodes rather
than raw detections, writes `confidence` onto the matched nodes and edges, and calls
`require_valid()` before returning - the check 9.3.7's bug needed and did not have. All 5 demo
pages validate; a run over more pages is one call to `annotate_diagram` away and is not done here
because several sibling modules are reading the same IR directory concurrently this phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

from src.assemble import corpus, nodes
from src.ir.model import Diagram
from src.ocr.confidence import auroc, calibration
from src.utils.config import ROOT
from src.utils.parallel import pmap

RUNS = ROOT / "experiments" / "assemble"
OUT = RUNS / "propagate.json"

#: Same threshold `src/assemble/nodes.py` scores its strict tier at - a box that does not
#: overlap the truth this much is a miss, not a confident detection of the wrong thing.
IOU_MATCH = 0.5

#: `weighted_mean`'s default split before renormalising over whichever components a node has.
WEIGHTS = {"det": 0.5, "ocr": 0.25, "hmm": 0.25}

#: OCR's cached per-crop confidences, 9.3.7's output. Read if present; propagation degrades to
#: det (+hmm) rather than failing when it is not.
OCR_SCORES = ROOT / "experiments" / "ocr" / "confidence_scores.json"

#: Sources whose IR carries `semantic_role`, so alone eligible for an HMM confidence - the same
#: restriction 7.3.3's sequence builder states.
LABELLED_SOURCES = ("hdbpmn", "fa_bresler")

RULES = ("product", "minimum", "weighted_mean", "noisy_or")

#: How many synthetic negative node-pairs to sample per page for the edge study, seeded so a
#: re-run reproduces the same set.
EDGE_NEG_SEED = 1234


# ------------------------------------------------------------------------------------------
# the four aggregation rules - generic over however many components a node/edge actually has
# ------------------------------------------------------------------------------------------


def rule_product(components: dict[str, float]) -> float:
    value = 1.0
    for v in components.values():
        value *= v
    return float(value)


def rule_minimum(components: dict[str, float]) -> float:
    return float(min(components.values()))


def rule_weighted_mean(components: dict[str, float], weights: dict[str, float] = WEIGHTS) -> float:
    total = sum(weights.get(k, 1.0) for k in components)
    if total <= 0:
        return float(np.mean(list(components.values())))
    return float(sum(v * weights.get(k, 1.0) for k, v in components.items()) / total)


def rule_noisy_or(components: dict[str, float]) -> float:
    remain = 1.0
    for v in components.values():
        remain *= 1.0 - v
    return float(1.0 - remain)


RULE_FUNCTIONS = {
    "product": rule_product,
    "minimum": rule_minimum,
    "weighted_mean": rule_weighted_mean,
    "noisy_or": rule_noisy_or,
}


def aggregate(components: dict[str, float], rule: str) -> float:
    """Apply one named rule to whatever components a node/edge has. Empty components -> nan."""
    if not components:
        return float("nan")
    return RULE_FUNCTIONS[rule](components)


# ------------------------------------------------------------------------------------------
# reading the three signals off a page
# ------------------------------------------------------------------------------------------


@dataclass
class NodeSample:
    page: str
    source: str
    node_id: str
    components: dict[str, float]
    wrong: bool


def _greedy_match(predicted: list, truths: list) -> list[tuple[int, int]]:
    """Best-IoU-first greedy matching over two lists of `Node`-shaped objects (`.bbox`,
    `.shape`), at `IOU_MATCH` - the same rule `src/assemble/nodes.py`'s `match()` scores its
    strict tier at, called here directly rather than re-derived."""
    scored = sorted(
        (
            (corpus.iou(p.bbox, t.bbox), i, j)
            for i, p in enumerate(predicted)
            for j, t in enumerate(truths)
            if p.bbox is not None and t.bbox is not None and corpus.iou(p.bbox, t.bbox) >= IOU_MATCH
        ),
        key=lambda tup: (-tup[0], tup[1], tup[2]),
    )
    used_d: set[int] = set()
    used_t: set[int] = set()
    out = []
    for _, i, j in scored:
        if i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        out.append((i, j))
    return out


def _ocr_confidence_map() -> dict[tuple[str, str], float]:
    """`(page_id, node_id) -> confidence`, read off 9.3.7's cache. Empty if it hasn't been run."""
    if not OCR_SCORES.is_file():
        return {}
    payload = json.loads(OCR_SCORES.read_text(encoding="utf-8-sig"))
    out = {}
    for name, value in zip(payload["files"], payload["confidence"], strict=True):
        parts = name.replace(".png", "").split("__")
        if len(parts) < 3 or not parts[2].startswith("n_"):
            continue
        out[(parts[1], parts[2][2:])] = float(value)
    return out


def _hmm_confidence_map(held_out_ids: set[str]) -> dict[tuple[str, str], float]:
    """`(page_id, node_id) -> forward-backward posterior of the Viterbi role`, for pages in a
    labelled source. The model is fit on every labelled sequence *outside* `held_out_ids`, so no
    held-out page's own roles reach its own transition counts."""
    try:
        from src.parse.posteriors import forward_backward
        from src.parse.sequences import labelled_diagrams, sequence_of
        from src.parse.viterbi import build_model, decode, logs
    except Exception:
        return {}

    diagrams = labelled_diagrams()
    train_seqs, held_diagrams = [], {}
    for d in diagrams:
        seq = sequence_of(d, labelled=True)
        if seq is None:
            continue
        if d["id"] in held_out_ids:
            held_diagrams[d["id"]] = d
        else:
            train_seqs.append(seq)
    if not train_seqs or not held_diagrams:
        return {}

    alphabet = sorted({s for seq in train_seqs for s in seq["observations"]})
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    model = build_model(train_seqs, alphabet, alpha=1.0)
    parameters = logs(model)

    out: dict[tuple[str, str], float] = {}
    for page_id, d in held_diagrams.items():
        seq = sequence_of(d, labelled=False)
        if seq is None:
            continue
        symbols = [index.get(s, 0) for s in seq["observations"]]
        path = decode(symbols, *parameters)
        gamma = forward_backward(symbols, *parameters)
        for position, node_id in enumerate(seq["node_ids"]):
            out[(page_id, node_id)] = float(gamma[position, path[position]])
    return out


def _node_samples_for_page(page: corpus.Page, ocr_map: dict, hmm_map: dict) -> list[NodeSample]:
    """The population this row scores: `10.1.1`'s *assembled* nodes, not raw detections.

    `nodes.build(page)` has already dropped the `arrowhead` class (66.4% of raw detections and
    never a node by construction), applied `MIN_SCORE` and de-duplicated across classes - so
    "wrong" here means what the plan actually asks for: no ground-truth match at IoU 0.5, or the
    wrong shape, on the node population 10.1.1 hands the rest of assembly.
    """
    predicted = nodes.build(page).diagram.nodes
    truths = nodes.truth_nodes(page)
    pairs = dict(_greedy_match(predicted, truths))
    out = []
    for i, p in enumerate(predicted):
        components = {"det": float(p.confidence)}
        j = pairs.get(i)
        wrong = j is None
        if j is not None:
            t = truths[j]
            wrong = p.shape != t.shape
            ocr = ocr_map.get((page.id, t.id))
            if ocr is not None:
                components["ocr"] = ocr
            hmm = hmm_map.get((page.id, t.id))
            if hmm is not None:
                components["hmm"] = hmm
        out.append(NodeSample(page.name, page.source, p.shape, components, wrong))
    return out


def node_samples(pages_: tuple[corpus.Page, ...] | None = None) -> list[NodeSample]:
    pages_ = pages_ if pages_ is not None else corpus.pages()
    ocr_map = _ocr_confidence_map()
    hmm_map = _hmm_confidence_map({p.id for p in pages_ if p.source in LABELLED_SOURCES})
    rows = pmap(
        lambda p: _node_samples_for_page(p, ocr_map, hmm_map),
        pages_,
        n_jobs=3,
        prefer="threads",
        desc="propagate/nodes",
    )
    return [s for page_rows in rows for s in page_rows]


# ------------------------------------------------------------------------------------------
# edges: endpoint confidences standing in for a detector that does not exist yet
# ------------------------------------------------------------------------------------------


@dataclass
class EdgeSample:
    page: str
    components: dict[str, float]
    wrong: bool


def _edge_samples_for_page(
    page: corpus.Page, node_conf: dict[str, float], rng_seed: int
) -> list[EdgeSample]:
    truth = corpus.truth(page)
    ids = [n.id for n in truth.nodes]
    if len(ids) < 2:
        return []
    positives = {(e.src, e.dst) for e in truth.edges if e.src in node_conf and e.dst in node_conf}
    rng = np.random.default_rng(rng_seed)
    candidates = [ids[i] for i in range(len(ids)) if ids[i] in node_conf]
    out = []
    for src, dst in positives:
        out.append(
            EdgeSample(page.name, {"src": node_conf[src], "dst": node_conf[dst]}, wrong=False)
        )
    n_neg = len(positives)
    tries = 0
    seen = set(positives)
    while n_neg > 0 and tries < n_neg * 20 and len(candidates) >= 2:
        a, b = rng.choice(candidates, size=2, replace=False)
        if (a, b) in seen or (b, a) in seen:
            tries += 1
            continue
        seen.add((a, b))
        out.append(EdgeSample(page.name, {"src": node_conf[a], "dst": node_conf[b]}, wrong=True))
        n_neg -= 1
        tries += 1
    return out


def edge_samples(pages_: tuple[corpus.Page, ...], best_node_rule: str) -> list[EdgeSample]:
    """One `EdgeSample` per ground-truth edge (positive) and per sampled non-edge (negative),
    per page. Endpoint confidences are recomputed here keyed by truth-node id - `node_samples`
    keys its rows by detected class, which is fine for scoring detections but useless for
    looking an endpoint up by the node id an edge actually names."""
    node_conf_by_page: dict[str, dict[str, float]] = {}
    for page in pages_:
        predicted = nodes.build(page).diagram.nodes
        truths = nodes.truth_nodes(page)
        pairs = dict(_greedy_match(predicted, truths))
        conf: dict[str, float] = {}
        for j, t in enumerate(truths):
            i = next((i for i, jj in pairs.items() if jj == j), None)
            if i is None:
                continue
            components = {"det": float(predicted[i].confidence)}
            conf[t.id] = aggregate(components, best_node_rule)
        node_conf_by_page[page.name] = conf

    out = []
    for k, page in enumerate(pages_):
        out.extend(
            _edge_samples_for_page(page, node_conf_by_page.get(page.name, {}), EDGE_NEG_SEED + k)
        )
    return out


# ------------------------------------------------------------------------------------------
# scoring: AUROC (ranking) and ECE (calibration), kept as separate questions throughout
# ------------------------------------------------------------------------------------------


def score_rules(samples: list, key: str = "components") -> dict:
    wrong = np.array([s.wrong for s in samples], dtype=bool)
    out = {}
    for rule in RULES:
        values = np.array([aggregate(getattr(s, key), rule) for s in samples])
        mask = ~np.isnan(values)
        out[rule] = {
            "n": int(mask.sum()),
            "auroc": round(auroc(-values[mask], wrong[mask]), 4),
            "calibration": calibration(values[mask], ~wrong[mask]),
        }
    return out


def detector_alone(samples: list) -> dict:
    """The free control: score every sample by `det` and nothing else."""
    wrong = np.array([s.wrong for s in samples], dtype=bool)
    have_det = np.array(["det" in s.components for s in samples])
    values = np.array([s.components.get("det", np.nan) for s in samples])
    return {
        "n": int(have_det.sum()),
        "auroc": round(auroc(-values[have_det], wrong[have_det]), 4),
        "calibration": calibration(values[have_det], ~wrong[have_det]),
    }


def random_control(samples: list, seed: int = 0) -> dict:
    wrong = np.array([s.wrong for s in samples], dtype=bool)
    rng = np.random.default_rng(seed)
    values = rng.random(len(samples))
    return {"n": len(samples), "auroc": round(auroc(values, wrong), 4)}


# ------------------------------------------------------------------------------------------
# writing the aggregate into the IR, and validating against the consumer's own loader
# ------------------------------------------------------------------------------------------


def annotate_diagram(page: corpus.Page, node_rule: str, edge_rule: str) -> Diagram:
    """Write the aggregated confidence onto a *copy* of the ground-truth diagram (in memory,
    native coordinates - `Diagram.load`, not `corpus.truth()`'s rescaled one) and require it
    valid before returning it. 9.3.7's bug wrote bare id strings where the schema wanted
    objects, and it round-tripped through `json` perfectly; the check that would have caught it
    is exactly `Diagram.require_valid()`, run here rather than assumed."""
    diagram = Diagram.load(page.ir_path)
    truths = nodes.truth_nodes(page)
    predicted = nodes.build(page).diagram.nodes
    pairs = dict(_greedy_match(predicted, truths))
    ocr_map = _ocr_confidence_map()
    hmm_map = _hmm_confidence_map(set())

    node_conf: dict[str, float] = {}
    for j, t in enumerate(truths):
        i = next((i for i, jj in pairs.items() if jj == j), None)
        components = {}
        if i is not None:
            components["det"] = float(predicted[i].confidence)
        ocr = ocr_map.get((page.id, t.id))
        if ocr is not None:
            components["ocr"] = ocr
        hmm = hmm_map.get((page.id, t.id))
        if hmm is not None:
            components["hmm"] = hmm
        if components:
            node_conf[t.id] = aggregate(components, node_rule)

    for node in diagram.nodes:
        if node.id in node_conf:
            node.confidence = round(node_conf[node.id], 4)

    for edge in diagram.edges:
        if edge.src in node_conf and edge.dst in node_conf:
            edge.confidence = round(
                aggregate({"src": node_conf[edge.src], "dst": node_conf[edge.dst]}, edge_rule), 4
            )
    diagram.require_valid()
    return diagram


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def run(n_demo_pages: int = 5) -> dict:
    pages_ = corpus.pages()
    samples = node_samples(pages_)

    node_ranking = score_rules(samples)
    node_control = detector_alone(samples)
    node_random = random_control(samples)
    best_node_rule = max(RULES, key=lambda r: node_ranking[r]["auroc"])

    edges = edge_samples(pages_, best_node_rule)
    edge_ranking = score_rules(edges)
    edge_control_values = np.array(
        [0.5 * (s.components["src"] + s.components["dst"]) for s in edges]
    )
    edge_wrong = np.array([s.wrong for s in edges], dtype=bool)
    edge_random = random_control(edges)
    best_edge_rule = max(RULES, key=lambda r: edge_ranking[r]["auroc"])

    demo = []
    for page in pages_[:n_demo_pages]:
        diagram = annotate_diagram(page, best_node_rule, best_edge_rule)
        demo.append(
            {
                "page": page.name,
                "nodes_with_confidence": sum(1 for n in diagram.nodes if n.confidence != 1.0),
                "edges_with_confidence": sum(1 for e in diagram.edges if e.confidence != 1.0),
                "valid": not diagram.problems(),
            }
        )

    result = {
        "pages": len(pages_),
        "node_samples": len(samples),
        "node_wrong_rate": round(float(np.mean([s.wrong for s in samples])), 4),
        "node_components_available": {
            "det": sum(1 for s in samples if "det" in s.components),
            "ocr": sum(1 for s in samples if "ocr" in s.components),
            "hmm": sum(1 for s in samples if "hmm" in s.components),
        },
        "node_rules": node_ranking,
        "node_detector_alone": node_control,
        "node_random": node_random,
        "best_node_rule": best_node_rule,
        "node_beats_detector_by": round(
            node_ranking[best_node_rule]["auroc"] - node_control["auroc"], 4
        ),
        "edge_samples": len(edges),
        "edge_wrong_rate": round(float(edge_wrong.mean()), 4) if len(edges) else float("nan"),
        "edge_rules": edge_ranking,
        "edge_endpoint_detector_alone": {
            "auroc": round(auroc(-edge_control_values, edge_wrong), 4)
        },
        "edge_random": edge_random,
        "best_edge_rule": best_edge_rule,
        "ir_fields_written": ["Node.confidence", "Edge.confidence"],
        "ir_demo": demo,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    result = run()
    from pathlib import Path

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("node_rules", "edge_rules")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
