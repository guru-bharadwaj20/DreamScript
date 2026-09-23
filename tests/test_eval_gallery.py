"""Phase 14.9 - a gallery is a selection, so the selection rule is what has to be tested.

The failure mode for a qualitative section is choosing the examples that flatter the system.
These tests pin that the order comes from the criterion's own report, that both ends are taken
symmetrically, and that what is shown is the program emitted from the *predicted* IR rather than
the reference the model was trained towards.
"""

from __future__ import annotations

import json

from src.eval import gallery


def _rows():
    return [
        {
            "page": f"p{index}",
            "source": "hdbpmn",
            "ged": float(index),
            "node_f1": 1.0,
            "edge_f1": 1.0,
        }
        for index in range(10)
    ]


def test_pages_are_ordered_by_the_criterion(monkeypatch, tmp_path):
    report = tmp_path / "s5.json"
    report.write_text(json.dumps({"pages": list(reversed(_rows()))}), encoding="utf-8")
    monkeypatch.setattr(gallery, "S5_REPORT", report)
    ordered = gallery.ranked_pages()
    assert [row["ged"] for row in ordered] == sorted(row["ged"] for row in _rows())


def test_ties_are_broken_by_name_so_the_order_is_stable(monkeypatch, tmp_path):
    report = tmp_path / "s5.json"
    rows = [{"page": "b", "source": "s", "ged": 3.0}, {"page": "a", "source": "s", "ged": 3.0}]
    report.write_text(json.dumps({"pages": rows}), encoding="utf-8")
    monkeypatch.setattr(gallery, "S5_REPORT", report)
    assert [row["page"] for row in gallery.ranked_pages()] == ["a", "b"]


def test_the_report_names_the_file_the_order_came_from():
    text = gallery.render({"each": 2, "order": "reports/s5_test_large.json", "entries": []})
    assert "reports/s5_test_large.json" in text
    assert "hand-picked" in text


def test_both_ends_are_shown_and_labelled():
    entries = [
        {
            "band": band,
            "page": f"{band}_page",
            "source": "hdbpmn",
            "ged": 0.0 if band == "best" else 40.0,
            "node_f1": 1.0,
            "edge_f1": 1.0,
            "edits": {},
            "functional_predicted": band == "best",
            "functional_reason": "equal" if band == "best" else "order_violation",
            "functional_gold": True,
            "image": None,
            "code_head": "class Machine:\n    pass",
            "code_lines": 2,
            "note": "",
        }
        for band in ("best", "worst")
    ]
    text = gallery.render({"each": 1, "order": "o", "entries": entries})
    assert "## Best cases" in text and "## Worst cases" in text
    assert "best_page" in text and "worst_page" in text
    assert "order_violation" in text


def test_a_page_the_emitter_refused_is_shown_with_its_reason():
    entry = {
        "band": "worst",
        "page": "p",
        "source": "hdbpmn",
        "ged": 40.0,
        "node_f1": 0.1,
        "edge_f1": 0.0,
        "edits": {},
        "functional_predicted": False,
        "functional_reason": "no_emission",
        "functional_gold": False,
        "image": None,
        "code_head": "",
        "code_lines": 0,
        "note": "the emitter refused the predicted IR",
    }
    text = gallery.render({"each": 1, "order": "o", "entries": [entry]})
    assert "the emitter refused the predicted IR" in text
    assert "```python" not in text


def test_a_page_missing_from_the_propagation_run_is_labelled_not_assumed():
    entry = {
        "band": "best",
        "page": "p",
        "source": "hdbpmn",
        "ged": 0.0,
        "node_f1": 1.0,
        "edge_f1": 1.0,
        "edits": {},
        "functional_predicted": None,
        "functional_reason": None,
        "functional_gold": None,
        "image": None,
        "code_head": "x = 1",
        "code_lines": 1,
        "note": "",
    }
    text = gallery.render({"each": 1, "order": "o", "entries": [entry]})
    assert "not in the 14.3 run" in text


def test_only_the_head_of_a_long_program_is_shown():
    assert gallery.CODE_LINES <= 30


def test_propagation_index_is_empty_when_the_run_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "PROPAGATION", tmp_path / "absent.json")
    assert gallery.propagation_index() == {}
