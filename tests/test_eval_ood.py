"""Phase 14.7 - the verdict is the whole point, so the verdict is what is tested.

An out-of-distribution page that gets an honest "I don't know" is a success; the same page
answered with confident runnable code is the failure this row exists to count. Those two must
never be collapsed, and neither may be inferred from an in-distribution page.
"""

from __future__ import annotations

from src.eval import ood


def _row(**kwargs):
    base = {
        "page": "p.png",
        "true_type": "er_diagram",
        "unseen": True,
        "routed_type": "flowchart",
        "needs_confirmation": False,
        "stopped_at": "",
        "code_chars": 0,
        "language": "",
        "parses": False,
        "nodes": 0,
        "route_confidence": 0.9,
    }
    base.update(kwargs)
    return base


def test_an_unseen_page_that_asks_is_not_counted_as_a_failure():
    assert ood.verdict(_row(needs_confirmation=True)) == "asked"


def test_an_unseen_page_that_stops_is_its_own_verdict():
    assert ood.verdict(_row(stopped_at="assemble")) == "stopped"


def test_an_unseen_page_answered_with_code_is_the_documented_failure():
    assert ood.verdict(_row(code_chars=800)) == "confident_wrong"


def test_the_same_answer_on_a_seen_type_is_not_called_wrong():
    assert ood.verdict(_row(true_type="flowchart", unseen=False, code_chars=800)) == "answered"


def test_a_page_with_no_code_and_no_reason_counts_as_stopped():
    assert ood.verdict(_row(code_chars=0)) == "stopped"


def test_python_is_parsed_and_other_languages_are_not_guessed():
    assert ood._parses("x = 1\n", "python") is True
    assert ood._parses("x = (", "python") is False
    assert ood._parses("CREATE TABLE t (id INT);", "sql") is None
    assert ood._parses("", "python") is False


def test_summarise_separates_the_unseen_pages_from_the_control():
    rows = [
        _row(code_chars=500),
        _row(needs_confirmation=True),
        _row(true_type="circuit", stopped_at="detect"),
        _row(true_type="flowchart", unseen=False, code_chars=900),
    ]
    summary = ood.summarise(rows)
    assert summary["unseen_pages"] == 3
    assert summary["unseen_confident_wrong"] == 1
    assert summary["unseen_asked"] == 1
    assert summary["unseen_stopped"] == 1
    assert summary["by_type"]["flowchart"]["unseen"] is False


def test_the_control_types_are_the_two_the_router_was_fitted_on():
    assert set(ood.SEEN) == {"flowchart", "state_machine"}
    assert not set(ood.UNSEEN) & set(ood.SEEN)


def test_render_states_the_headline_counts():
    rows = [_row(code_chars=500), _row(true_type="flowchart", unseen=False, code_chars=900)]
    text = ood.render({"confidence_floor": 0.6, "corpus": "synthetic", "seconds": 1.0, **ood.summarise(rows), "rows": rows})
    assert "confident, runnable, wrong code" in text
    assert "er_diagram" in text


def test_missing_render_directory_yields_no_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(ood, "SYNTHETIC", tmp_path)
    assert ood.pages_of("circuit", 4) == []
