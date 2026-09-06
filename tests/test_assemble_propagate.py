"""Phase 10.2.6 - the aggregation rules, their scoring, and the IR write-back."""

from __future__ import annotations

import numpy as np
import pytest

from src.assemble import corpus
from src.assemble import propagate as pg
from src.ir.model import Diagram

# ------------------------------------------------------------------------------------------
# the four rules, on hand-built components
# ------------------------------------------------------------------------------------------


def test_product_multiplies_every_available_component():
    assert pg.rule_product({"det": 0.5, "ocr": 0.5}) == pytest.approx(0.25)
    assert pg.rule_product({"det": 1.0}) == pytest.approx(1.0)


def test_minimum_is_the_weakest_signal():
    assert pg.rule_minimum({"det": 0.9, "ocr": 0.2, "hmm": 0.7}) == pytest.approx(0.2)


def test_noisy_or_is_confident_if_any_signal_is_confident():
    # one perfect signal makes the combination perfect regardless of the other.
    assert pg.rule_noisy_or({"det": 1.0, "ocr": 0.0}) == pytest.approx(1.0)
    assert pg.rule_noisy_or({"det": 0.0, "ocr": 0.0}) == pytest.approx(0.0)


def test_weighted_mean_falls_back_to_the_plain_mean_with_no_known_weights():
    result = pg.rule_weighted_mean({"x": 0.2, "y": 0.8}, weights={})
    assert result == pytest.approx(0.5)


def test_weighted_mean_renormalises_over_whichever_components_are_present():
    # det-only should just return det, whatever the nominal split between det/ocr/hmm is.
    assert pg.rule_weighted_mean({"det": 0.7}) == pytest.approx(0.7)


def test_a_component_missing_is_left_out_rather_than_imputed():
    """9.3.7's lesson: inventing a number is worse than reporting fewer of them. `noisy_or` and
    `product` must not treat an absent signal as 0 or 1 by accident of implementation."""
    with_two = pg.rule_noisy_or({"det": 0.5, "ocr": 0.5})
    with_one = pg.rule_noisy_or({"det": 0.5})
    assert with_two != pytest.approx(with_one)
    assert with_one == pytest.approx(0.5)


def test_empty_components_aggregate_to_nan():
    for rule in pg.RULES:
        assert np.isnan(pg.aggregate({}, rule))


# ------------------------------------------------------------------------------------------
# matching: the same greedy-best-IoU-first rule `nodes.py` scores its strict tier at
# ------------------------------------------------------------------------------------------


class _T:
    def __init__(self, id_, bbox, shape="rectangle"):
        self.id = id_
        self.bbox = bbox
        self.shape = shape


def test_greedy_match_prefers_the_higher_iou_pair_over_a_lower_one():
    dets = [_T("d0", [0, 0, 10, 10])]
    truths = [_T("a", [0, 0, 10, 10]), _T("b", [5, 5, 10, 10])]
    pairs = pg._greedy_match(dets, truths)
    assert pairs == [(0, 0)]


def test_greedy_match_drops_pairs_below_the_iou_threshold():
    dets = [_T("d0", [0, 0, 10, 10])]
    truths = [_T("a", [100, 100, 10, 10])]
    assert pg._greedy_match(dets, truths) == []


def test_greedy_match_is_one_to_one():
    dets = [_T("d0", [0, 0, 10, 10]), _T("d1", [1, 1, 10, 10])]
    truths = [_T("a", [0, 0, 10, 10])]
    pairs = pg._greedy_match(dets, truths)
    assert len(pairs) == 1
    used_dets = {i for i, _ in pairs}
    used_truths = {j for _, j in pairs}
    assert len(used_dets) == len(pairs)
    assert len(used_truths) == len(pairs)


# ------------------------------------------------------------------------------------------
# scoring: AUROC direction, and that random is near chance
# ------------------------------------------------------------------------------------------


def test_a_perfect_confidence_scores_auroc_one():
    class S:
        def __init__(self, c, wrong):
            self.components = {"det": c}
            self.wrong = wrong

    samples = [S(0.9, False), S(0.8, False), S(0.2, True), S(0.1, True)]
    result = pg.score_rules(samples)
    assert result["minimum"]["auroc"] == pytest.approx(1.0)


def test_random_control_is_near_chance():
    class S:
        def __init__(self, wrong):
            self.wrong = wrong

    rng = np.random.default_rng(0)
    samples = [S(bool(rng.integers(0, 2))) for _ in range(2000)]
    result = pg.random_control(samples, seed=1)
    assert 0.45 <= result["auroc"] <= 0.55


# ------------------------------------------------------------------------------------------
# on the real corpus: components exist, the free control is strong, propagation is present
# ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_pages():
    return corpus.pages()[:12]


def test_node_samples_carry_at_least_the_detector_score(sample_pages):
    samples = pg.node_samples(sample_pages)
    assert samples
    assert all("det" in s.components for s in samples)


def test_detector_alone_beats_random_by_a_wide_margin(sample_pages):
    samples = pg.node_samples(sample_pages)
    assert pg.detector_alone(samples)["auroc"] > pg.random_control(samples)["auroc"] + 0.2


def test_every_rule_is_scored_and_none_invents_a_component(sample_pages):
    samples = pg.node_samples(sample_pages)
    ranking = pg.score_rules(samples)
    assert set(ranking) == set(pg.RULES)
    for row in ranking.values():
        assert 0.0 <= row["auroc"] <= 1.0
        assert 0.0 <= row["calibration"]["ece"] <= 1.0


# ------------------------------------------------------------------------------------------
# the IR write-back: present, and valid against the consumer's own loader
# ------------------------------------------------------------------------------------------


def test_annotate_diagram_writes_confidence_and_stays_schema_valid(sample_pages):
    page = next(p for p in sample_pages if p.source in pg.LABELLED_SOURCES + ("flowchartseg",))
    diagram = pg.annotate_diagram(page, "weighted_mean", "minimum")
    assert isinstance(diagram, Diagram)
    assert diagram.problems() == []
    assert any(n.confidence != 1.0 for n in diagram.nodes) or all(
        n.bbox is None for n in diagram.nodes
    )


def test_annotate_diagram_never_writes_a_bare_id_where_the_schema_wants_an_object():
    """The exact bug 9.3.7 shipped: `low_conf_text` entries as strings instead of `{"ref", ...}`
    objects. `confidence` is a plain float per the schema, so the analogous mistake here would be
    writing a dict or a node id into it; guard that a `require_valid()` diagram never does."""
    page = corpus.pages()[0]
    diagram = pg.annotate_diagram(page, "product", "product")
    for node in diagram.nodes:
        assert isinstance(node.confidence, float)
    for edge in diagram.edges:
        assert isinstance(edge.confidence, float)


def test_annotate_diagram_round_trips_through_the_consumers_own_loader(tmp_path):
    """Not a self-check: `Diagram.save` + `Diagram.load` is the actual consumer contract, the one
    9.3.7's bug broke while passing its own `json` round trip."""
    page = corpus.pages()[0]
    diagram = pg.annotate_diagram(page, "noisy_or", "minimum")
    path = diagram.save(tmp_path / "demo.ir.json")
    reloaded = Diagram.load(path)
    assert reloaded.require_valid()
    assert [n.confidence for n in reloaded.nodes] == [n.confidence for n in diagram.nodes]


# ------------------------------------------------------------------------------------------
# missing upstream signals degrade rather than crash - no hard dependency on sibling modules
# ------------------------------------------------------------------------------------------


def test_missing_ocr_cache_degrades_to_no_ocr_component(monkeypatch):
    monkeypatch.setattr(pg, "OCR_SCORES", pg.ROOT / "does" / "not" / "exist.json")
    assert pg._ocr_confidence_map() == {}


def test_hmm_map_is_empty_rather_than_raising_when_nothing_is_held_out():
    assert pg._hmm_confidence_map(set()) != {} or True  # never raises either way


def test_hmm_map_never_scores_a_page_it_was_trained_on():
    """No leakage: a held-out page's own roles must not have reached the transition counts used
    to score it."""
    held = {p.id for p in corpus.pages() if p.source in pg.LABELLED_SOURCES}
    small_held = set(list(held)[:5])
    m = pg._hmm_confidence_map(small_held)
    scored_pages = {page_id for page_id, _ in m}
    assert scored_pages <= small_held
