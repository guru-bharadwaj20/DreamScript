"""Phase 12.1.4 - synthetic pairs: determinism, schema validity, recoverability, dedup, rendering."""

from __future__ import annotations

import json
import re
from collections import defaultdict

import pytest

from src.codegen import pairs, quality, schema, serialise, synthetic, targets
from src.eval import react, spice
from src.ir.model import Diagram

TOOLS = pytest.mark.skipif(
    not (react.toolchain_ready() and spice.available()), reason="needs node and ngspice"
)
SHARD = pairs.PAIRS_DIR / "synthetic.jsonl"


@pytest.mark.parametrize("diagram_type", synthetic.TYPES)
def test_generation_is_a_pure_function_of_type_and_seed(diagram_type):
    a = json.dumps(synthetic.random_diagram(diagram_type, 41), sort_keys=True)
    b = json.dumps(synthetic.random_diagram(diagram_type, 41), sort_keys=True)
    assert a == b
    assert a != json.dumps(synthetic.random_diagram(diagram_type, 42), sort_keys=True)


@pytest.mark.parametrize("diagram_type", synthetic.TYPES)
def test_every_family_is_schema_valid_ir(diagram_type):
    for seed in range(25):  # 5 seeds per structure family
        diagram = synthetic.random_diagram(diagram_type, seed)
        assert Diagram.from_dict(diagram).problems() == [], (diagram_type, seed)


def test_circuit_netlist_is_recoverable_from_ir_text():
    """Every card's ref, value and both nets must be readable in ir_text - not only in attrs."""
    for seed in range(20):
        record = pairs.make_pair(synthetic.random_diagram("circuit", seed), "synthetic")
        parsed = serialise.parse(record["ir_text"])
        text = {n["id"]: n["text"] for n in parsed["nodes"]}
        nets = defaultdict(dict)
        for edge in parsed["edges"]:
            nets[edge["src"]][edge["label"]] = text[edge["dst"]]
        expected = sorted(f"{text[p].split()[0]} {nets[p]['+']} {nets[p]['-']}" for p in nets)
        cards = sorted(
            " ".join(line.split()[:3])
            for line in record["target_code"].splitlines()
            if line and line[0] in "RCLV"
        )
        assert cards == expected


def test_er_columns_are_attribute_nodes():
    record = pairs.make_pair(synthetic.random_diagram("er_diagram", 3), "synthetic")
    for column in re.findall(r"^    (\w+) (TEXT|REAL|INTEGER)$", record["target_code"], re.M):
        assert f"{column[0]}: {column[1].lower()}" in record["ir_text"]


def test_wire_nodes_are_not_emitted_as_parts():
    record = pairs.make_pair(synthetic.random_diagram("circuit", 5), "synthetic")
    parts = [
        n for n in serialise.parse(record["ir_text"])["nodes"] if n["semantic_role"] == "component"
    ]
    cards = [ln for ln in record["target_code"].splitlines() if ln and ln[0] in "RCLV"]
    assert len(cards) == len(parts)
    assert "R_TERM" not in record["target_code"] and "VSRC_AUTO" not in record["target_code"]
    assert targets.language_for("er_diagram") == "sql"


def test_dedupe_drops_exact_near_and_capped_duplicates():
    base = pairs.make_pair(synthetic.random_diagram("flowchart", 7), "synthetic")
    near = dict(base, ir_text=base["ir_text"].replace("start", "begin", 1))
    state = {"exact": set(), "buckets": defaultdict(list)}
    kept, why = synthetic.dedupe([base, dict(base), near], state)
    assert len(kept) == 1
    assert why["exact_duplicate"] == 1 and why["near_duplicate"] == 1


def test_structure_signature_ignores_text_only():
    text = "T|flowchart\nN|n0|rectangle|fetch order|process\nE|n0|n1|yes\nO|n0"
    renamed = text.replace("fetch order", "audit cart").replace("|yes", "|no")
    rewired = text.replace("E|n0|n1", "E|n1|n0")
    assert synthetic.structure_signature(text) == synthetic.structure_signature(renamed)
    assert synthetic.structure_signature(text) != synthetic.structure_signature(rewired)


@pytest.mark.parametrize("diagram_type", synthetic.TYPES)
def test_render_writes_a_nonblank_png(diagram_type, tmp_path):
    import cv2

    path = tmp_path / f"{diagram_type}.png"
    width, height = synthetic.render(synthetic.random_diagram(diagram_type, 11), path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (height, width)
    assert (image < 128).mean() > 0.002  # ink was drawn


@TOOLS
def test_generated_targets_pass_the_quality_filter():
    records = pairs.assign_splits(
        [
            pairs.make_pair(synthetic.random_diagram(t, s), "synthetic")
            for t in synthetic.TYPES
            for s in range(10)
        ]
    )
    assert all(r["ok"] for r in quality.check_records(records))


@pytest.mark.skipif(not SHARD.is_file(), reason="needs python -m src.codegen.synthetic build")
def test_written_shard_meets_the_row():
    records = list(schema.read_jsonl(SHARD))
    assert len(records) >= 10_000
    assert {r["split"] for r in records} == {"train"}
    assert len({r["ir_text"] for r in records}) == len(records)
    assert {r["diagram_type"] for r in records} == set(synthetic.TYPES)
    assert all((pairs.ROOT / r["meta"]["image"]).is_file() for r in records[::500])
