"""Phase 15.7 - a correction store is training data, so what it refuses matters most.

Anything this store accepts will eventually be trained on. The tests are therefore mostly about
rejection: a correction against a page that does not exist, a node that does not exist, or a value
that is not actually a change, must never reach an export.
"""

from __future__ import annotations

import json

import pytest

from src.mlops import corrections as C


@pytest.fixture()
def store(tmp_path):
    return tmp_path / "corrections.jsonl"


@pytest.fixture()
def diagram():
    return {
        "id": "probe",
        "nodes": [
            {"id": "n1", "text": "q0", "shape": "circle", "semantic_role": "initial-state"},
            {"id": "n2", "text": "q1", "shape": "circle", "semantic_role": "state"},
        ],
        "edges": [{"id": "e1", "src": "n1", "dst": "n2", "label": "a"}],
    }


def entry(**overrides):
    base = {
        "diagram_id": "probe/page",
        "target": "node",
        "target_id": "n1",
        "field": "text",
        "new_value": "q7",
    }
    base.update(overrides)
    return base


def test_a_valid_correction_keeps_the_old_value(diagram):
    stored = C.validate(entry(), diagram)
    assert stored["old_value"] == "q0"
    assert stored["new_value"] == "q7"
    # The old value is what makes it a training pair rather than just a label.
    assert stored["field"] == "text"


def test_a_correction_that_changes_nothing_is_rejected(diagram):
    """A user confirming is not a user correcting; it carries no gradient."""
    with pytest.raises(C.Rejected, match="nothing was corrected"):
        C.validate(entry(new_value="q0"), diagram)


def test_a_correction_to_an_unknown_node_is_rejected(diagram):
    with pytest.raises(C.Rejected, match="not in"):
        C.validate(entry(target_id="n99"), diagram)


def test_a_correction_to_an_unknown_field_is_rejected(diagram):
    with pytest.raises(C.Rejected, match="field must be one of"):
        C.validate(entry(field="colour"), diagram)


def test_a_correction_to_an_unknown_target_kind_is_rejected(diagram):
    with pytest.raises(C.Rejected, match="target must be one of"):
        C.validate(entry(target="page"), diagram)


def test_a_missing_required_key_is_rejected(diagram):
    for key in ("diagram_id", "target", "target_id", "field", "new_value"):
        broken = entry()
        del broken[key]
        with pytest.raises(C.Rejected, match=key):
            C.validate(broken, diagram)


def test_an_edge_correction_addresses_the_edge_collection(diagram):
    stored = C.validate(
        entry(target="edge", target_id="e1", field="edge_label", new_value="b"), diagram
    )
    assert stored["old_value"] == "a"


def test_the_store_is_append_only(store, diagram, monkeypatch):
    monkeypatch.setattr(C, "load_ir", lambda _: diagram)
    C.record(entry(new_value="q7"), store=store)
    C.record(entry(new_value="q8"), store=store)
    rows = C.read(store)
    # Two records, not one updated: which came second is what training wants to know.
    assert len(rows) == 2
    assert [r["new_value"] for r in rows] == ["q7", "q8"]


def test_last_wins_collapses_a_user_correcting_themselves(store, diagram, monkeypatch):
    monkeypatch.setattr(C, "load_ir", lambda _: diagram)
    C.record(entry(new_value="q7", recorded_at="2026-01-01T00:00:00+00:00"), store=store)
    C.record(entry(new_value="q8", recorded_at="2026-01-02T00:00:00+00:00"), store=store)
    final = C.latest(C.read(store))
    assert len(final) == 1
    assert final[0]["new_value"] == "q8"


def test_last_wins_keeps_corrections_to_different_fields_apart(store, diagram, monkeypatch):
    monkeypatch.setattr(C, "load_ir", lambda _: diagram)
    C.record(entry(field="text", new_value="q7"), store=store)
    C.record(entry(field="shape", new_value="square"), store=store)
    assert len(C.latest(C.read(store))) == 2


def test_a_correction_is_stale_once_the_ir_changes(store, diagram, monkeypatch):
    monkeypatch.setattr(C, "load_ir", lambda _: diagram)
    C.record(entry(), store=store)
    rows = C.read(store)
    assert C.stale(rows) == []

    moved = {**diagram, "nodes": [{**diagram["nodes"][0], "text": "something else"}]}
    monkeypatch.setattr(C, "load_ir", lambda _: moved)
    flagged = C.stale(rows)
    assert len(flagged) == 1
    assert "changed" in flagged[0]["why"]


def test_stale_corrections_are_excluded_from_the_export(store, diagram, monkeypatch):
    monkeypatch.setattr(C, "load_ir", lambda _: diagram)
    C.record(entry(), store=store)
    rows = C.read(store)
    assert len(C.export(rows=rows)) == 1

    # The pipeline re-runs and the IR moves under the correction; it may no longer apply.
    moved = {**diagram, "nodes": [{**diagram["nodes"][0], "text": "moved"}]}
    monkeypatch.setattr(C, "load_ir", lambda _: moved)
    assert len(C.export(rows=rows)) == 0


def test_export_has_the_columns_training_needs(store, diagram, monkeypatch):
    monkeypatch.setattr(C, "load_ir", lambda _: diagram)
    C.record(entry(), store=store)
    frame = C.export(rows=C.read(store))
    for column in ("diagram_id", "target_id", "field", "old_value", "new_value"):
        assert column in frame.columns


def test_an_empty_store_reports_empty_rather_than_failing(tmp_path):
    result = C.stats(tmp_path / "nothing.jsonl")
    assert result["records"] == 0
    assert result["exists"] is False
    assert result["note"]  # says plainly that no real corrections exist yet


def test_the_value_table_matches_s5s_own_numbers():
    """The row's justification is measured, so it must come from the report and not a constant."""
    worth = C.value()
    if not worth.get("available"):
        pytest.skip("needs reports/s5_test_large.json")
    # sub_text is the majority of edit mass; this is the claim the design rests on.
    assert worth["text_share"] > 0.5
    # And perfect text buys more than perfect edges.
    assert worth["median_ged_with_perfect_text"] < worth["median_ged_with_perfect_edges"]
    assert worth["median_ged_with_perfect_text"] < worth["median_ged_as_assembled"]


def test_a_malformed_line_does_not_lose_the_store(store, diagram, monkeypatch):
    monkeypatch.setattr(C, "load_ir", lambda _: diagram)
    C.record(entry(), store=store)
    with store.open("a", encoding="utf-8") as handle:
        handle.write("\n")  # a blank line, which a partial write can leave
    assert len(C.read(store)) == 1
