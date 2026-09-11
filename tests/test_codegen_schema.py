"""Phase 12.1.1 - the pair schema, and the pair builder that must satisfy it.

The valid record is built by `pairs.make_pair` from a generated diagram, so the suite needs no
dataset; the corpus tests re-check the written shards when they exist. Each rule in
`schema.validate` gets a corruption that must trip it - a validator that accepts everything
passes every happy-path test there is.
"""

from __future__ import annotations

import copy
import json

import pytest

from src.codegen import pairs, schema, serialise, splits
from src.synth import graphs

SHARDS = pytest.mark.skipif(
    not (pairs.PAIRS_DIR / "train.jsonl").is_file(),
    reason="needs python -m src.codegen.pairs build && merge",
)


def valid_record(diagram_type: str = "flowchart", structure: str = "branching") -> dict:
    record = pairs.make_pair(graphs.random_diagram(diagram_type, structure, 3), "synthetic")
    return pairs.assign_splits([record])[0]


@pytest.mark.parametrize("diagram_type", graphs.DIAGRAM_TYPES)
def test_a_built_pair_is_valid_for_every_type(diagram_type):
    record = valid_record(diagram_type)
    assert schema.validate(record) == []
    assert list(record) == list(schema.FIELDS)
    assert record["split"] == "train" and record["split_basis"] == "synthetic_train_only"


def test_er_is_canonicalised_to_the_ir_enum_everywhere():
    record = valid_record("er")
    assert record["diagram_type"] == "er_diagram"
    assert record["ir_text"].startswith("T|er_diagram\n")
    assert record["language"] == "sql"


def test_target_never_carries_the_diagram_id():
    diagram = graphs.random_diagram("state_machine", "looping", 7)
    record = pairs.make_pair(diagram, "synthetic")
    assert diagram["id"] not in record["target_code"]
    assert "DiagramMachine" in record["target_code"]


def test_traversal_is_the_o_line():
    record = valid_record("wireframe", "nested")
    assert record["traversal"] == serialise.parse(record["ir_text"])["traversal"]


def _corrupt(record: dict, name: str) -> dict:
    bad = copy.deepcopy(record)
    if name == "missing_key":
        bad.pop("language")
    elif name == "extra_key":
        bad["prompt"] = "x"
    elif name == "wrong_type":
        bad["traversal"] = "n0 n1"
    elif name == "scribe_type":
        bad["scribe"] = 3
    elif name == "meta_type":
        bad["meta"] = []
    elif name == "empty_target":
        bad["target_code"] = "  \n"
    elif name == "unknown_type":
        bad["diagram_type"] = "er"
    elif name == "unknown_language":
        bad["language"] = "tsx"
    elif name == "unknown_split":
        bad["split"] = "val"
    elif name == "unknown_source":
        bad["source"] = "selftest"
    elif name == "traversal_extra_id":
        bad["traversal"] = [*bad["traversal"], "ghost"]
    elif name == "traversal_missing_id":
        bad["traversal"] = bad["traversal"][:-1]
    elif name == "traversal_reordered":
        bad["traversal"] = list(reversed(bad["traversal"]))
    elif name == "wrong_language_for_type":
        bad["language"] = "react"
    elif name == "type_disagrees_with_ir_text":
        bad["diagram_type"] = "state_machine"
    elif name == "no_t_line":
        bad["ir_text"] = bad["ir_text"].split("\n", 1)[1]
    elif name == "id_leaks_into_code":
        bad["target_code"] += f"\n# {bad['diagram_id']}\n"
    elif name == "heldout_without_scribe":
        bad["source"], bad["split"] = "hdbpmn", "test"
    elif name == "synthetic_outside_train":
        bad["split"], bad["scribe"] = "validation", "w1"
    elif name == "empty_split_basis":
        bad["split_basis"] = ""
    else:  # pragma: no cover
        raise KeyError(name)
    return bad


CORRUPTIONS = (
    "missing_key",
    "extra_key",
    "wrong_type",
    "scribe_type",
    "meta_type",
    "empty_target",
    "unknown_type",
    "unknown_language",
    "unknown_split",
    "unknown_source",
    "traversal_extra_id",
    "traversal_missing_id",
    "traversal_reordered",
    "wrong_language_for_type",
    "type_disagrees_with_ir_text",
    "no_t_line",
    "id_leaks_into_code",
    "heldout_without_scribe",
    "synthetic_outside_train",
    "empty_split_basis",
)


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_every_corruption_is_caught(name):
    record = valid_record()
    assert schema.validate(record) == []
    assert schema.validate(_corrupt(record, name)), name


def test_validate_reports_every_problem_not_the_first():
    bad = valid_record()
    bad.update(language="tsx", split="val", source="selftest")
    assert len(schema.validate(bad)) >= 3


def test_escaped_pipe_in_a_label_does_not_break_the_cross_check():
    diagram = graphs.random_diagram("flowchart", "linear", 1)
    diagram["nodes"][0]["text"] = "a|b \\ c"
    record = pairs.assign_splits([pairs.make_pair(diagram, "synthetic")])[0]
    assert "\\p" in record["ir_text"]
    assert schema.validate(record) == []


def test_jsonl_round_trip_is_byte_stable(tmp_path):
    records = [valid_record(t) for t in graphs.DIAGRAM_TYPES]
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert schema.write_jsonl(records, first) == 5
    schema.write_jsonl(list(schema.read_jsonl(first)), second)
    assert first.read_bytes() == second.read_bytes()


def test_write_refuses_an_invalid_record(tmp_path):
    bad = _corrupt(valid_record(), "unknown_split")
    with pytest.raises(ValueError, match="split"):
        schema.write_jsonl([bad], tmp_path / "bad.jsonl")


def test_the_schema_agrees_with_the_modules_it_sits_between():
    from src.codegen import quality, targets

    assert schema.SPLITS == splits.SPLITS
    assert set(schema.LANGUAGES) <= set(quality.CHECKS)
    assert set(targets.LANGUAGES.values()) == set(schema.LANGUAGES)
    ir_enum = json.loads((pairs.ROOT / "schemas" / "ir.schema.json").read_text(encoding="utf-8"))
    assert list(schema.DIAGRAM_TYPES) == ir_enum["properties"]["diagram_type"]["enum"]


def test_splits_does_not_adopt_a_scribeless_manifest_entry():
    record = {"diagram_id": "d1", "source": "didi", "diagram_type": "flowchart"}
    index = {"d1": {"split": "test", "scribe": None, "source": "didi", "basis": "manifest"}}
    out = splits.assign([record], index=index)[0]
    assert (out["split"], out["split_basis"]) == ("train", "no_scribe_train_only")


def test_load_pairs_filters_and_rejects_unknown_splits(tmp_path):
    records = [valid_record(t) for t in graphs.DIAGRAM_TYPES]
    schema.write_jsonl(records, tmp_path / "train.jsonl")
    for name in ("validation", "test"):
        schema.write_jsonl([], tmp_path / f"{name}.jsonl")
    got = list(pairs.load_pairs("train", diagram_types=["circuit"], root=tmp_path))
    assert [r["diagram_type"] for r in got] == ["circuit"]
    assert len(list(pairs.load_pairs(root=tmp_path))) == 5
    with pytest.raises(ValueError):
        list(pairs.load_pairs("val", root=tmp_path))


@SHARDS
def test_written_splits_are_valid_and_leak_free():
    report = pairs.stats()
    assert report["invalid"] == 0
    records = list(pairs.load_pairs())
    summary = splits.summarize(records)
    assert all(summary["checks"].values()), summary["checks"]
    ids = [r["diagram_id"] for r in records]
    assert len(ids) == len(set(ids))
