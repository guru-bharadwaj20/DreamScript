"""Phase 12.1.2 - the compact IR text format.

The three properties worth a test are determinism, losslessness over the fields it claims to
carry, and the token win. The first is tested three ways because it breaks three ways: across
runs, across input list order, and across `PYTHONHASHSEED` - and only the third one catches a
set or dict iteration that has leaked into the writer, so it is run in a real subprocess with
the seed actually set rather than asserted in-process where it cannot fail.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.codegen.serialise import (
    PRUNED_EDGE_FIELDS,
    PRUNED_NODE_FIELDS,
    escape,
    normalise,
    parse,
    prune,
    serialise,
    unescape,
    variants,
)

ROOT = Path(__file__).resolve().parents[1]


# -- fixtures ------------------------------------------------------------------------------


def _diagram() -> dict:
    """A small graph with everything the format has to cope with: a cycle, an undirected edge,
    a parallel pair, an unlabelled edge, an unknown role, and a node the traversal misses."""
    return {
        "ir_version": "1.0",
        "id": "fixture",
        "diagram_type": "flowchart",
        "nodes": [
            {
                "id": "n0",
                "shape": "oval",
                "text": "start",
                "semantic_role": "start",
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "confidence": 0.91,
                "source_id": "s0",
                "attrs": {},
            },
            {
                "id": "n1",
                "shape": "diamond",
                "text": "ok?",
                "semantic_role": "decision",
                "bbox": [5.5, 6.5, 7.5, 8.5],
                "confidence": 0.7,
                "source_id": "s1",
                "attrs": {},
            },
            {
                "id": "n2",
                "shape": "box",
                "text": "",
                "semantic_role": "unknown",
                "bbox": [9.0, 9.0, 9.0, 9.0],
                "confidence": 0.4,
                "source_id": "s2",
                "attrs": {},
            },
            {"id": "orphan", "shape": "box", "text": "off to one side"},
        ],
        "edges": [
            {
                "id": "e0",
                "src": "n0",
                "dst": "n1",
                "directed": True,
                "label": "",
                "polyline": [[0.0, 0.0], [1.0, 1.0]],
                "confidence": 0.8,
                "source_id": "t0",
            },
            {"id": "e1", "src": "n1", "dst": "n2", "directed": True, "label": "yes"},
            {"id": "e2", "src": "n1", "dst": "n0", "directed": True, "label": "retry"},
            {"src": "n1", "dst": "n2", "directed": True, "label": "also"},
            {"src": "n1", "dst": "n2", "directed": False, "label": "also"},
            {"id": "dangling", "src": "n0", "dst": "gone", "directed": True, "label": "x"},
        ],
    }


TRAVERSAL = ["n0", "n1", "n2"]


@pytest.fixture(scope="module")
def corpus() -> list[tuple[dict, list[str]]]:
    """Real IR plus its real traversal, or a skip if the Phase 2 corpus is not built here."""
    from src.parse.sequences import IR_DIR, load_ir, traversal

    if not IR_DIR.is_dir():
        pytest.skip("no IR corpus on this machine; run the Phase 2 converters")
    out = []
    for source in sorted(p.name for p in IR_DIR.iterdir() if p.is_dir()):
        for diagram in load_ir([source], limit=24)[:12]:
            order, _ = traversal(diagram)
            out.append((diagram, order))
    if not out:
        pytest.skip("IR corpus is empty")
    return out


# -- the format itself ---------------------------------------------------------------------


def test_shape_of_the_output():
    text = serialise(_diagram(), TRAVERSAL)
    assert text.split("\n") == [
        "T|flowchart",
        "N|n0|oval|start|start",
        "N|n1|diamond|ok?|decision",
        "N|n2|box",
        "N|orphan|box|off to one side",
        "E|n0|n1",
        "E|n1|n0|retry",
        "E|n1|n2|also|-",
        "E|n1|n2|also",
        "E|n1|n2|yes",
        "O|n0 n1 n2 orphan",
        "R|n1>n0",
    ]


def test_unknown_role_and_empty_text_are_trimmed_not_spelled_out():
    assert "N|n2|box" in serialise(_diagram(), TRAVERSAL).split("\n")


def test_edge_to_a_missing_node_is_dropped():
    assert "gone" not in serialise(_diagram(), TRAVERSAL)


def test_node_missed_by_the_traversal_is_still_emitted_last_and_by_id():
    lines = serialise(_diagram(), TRAVERSAL).split("\n")
    assert lines[4] == "N|orphan|box|off to one side"
    assert lines[-2] == "O|n0 n1 n2 orphan"


def test_return_edge_line_lists_only_edges_that_go_backwards():
    assert serialise(_diagram(), TRAVERSAL).split("\n")[-1] == "R|n1>n0"


def test_no_return_line_when_the_graph_is_a_forward_chain():
    diagram = {
        "diagram_type": "flowchart",
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [{"src": "a", "dst": "b"}],
    }
    assert serialise(diagram, ["a", "b"]).split("\n")[-1] == "O|a b"


def test_self_loop_counts_as_a_return_edge():
    diagram = {"nodes": [{"id": "a"}], "edges": [{"src": "a", "dst": "a"}]}
    assert serialise(diagram, ["a"]).split("\n")[-1] == "R|a>a"


def test_empty_diagram_does_not_crash():
    assert serialise({}, []) == "T|unknown\nO"


def test_geometry_never_reaches_the_text():
    text = serialise(_diagram(), TRAVERSAL)
    assert not any(token in text for token in ("bbox", "polyline", "confidence", "source_id"))
    assert "0.91" not in text and "5.5" not in text


# -- escaping ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "plain",
        "a|b",
        "a\\b",
        "a\\pb",
        "line\nbreak",
        "crlf\r\nbreak",
        "  padded  ",
        "|",
        "\\",
        "\\\\|\\n",
        "emoji \U0001f600 and é",
        "",
    ],
)
def test_escape_round_trips_through_unescape(raw):
    assert unescape(escape(normalise(raw))) == normalise(raw)


def test_escaped_field_never_contains_a_delimiter_or_a_newline():
    cooked = escape(normalise("a|b\nc\\d"))
    assert "|" not in cooked and "\n" not in cooked


def test_normalise_is_idempotent():
    for raw in ("  x \r\n", "a\r\nb", "\n\n", "x"):
        assert normalise(normalise(raw)) == normalise(raw)


def test_a_pipe_in_node_text_does_not_add_a_field():
    diagram = {"nodes": [{"id": "a", "shape": "box", "text": "yes|no"}], "edges": []}
    line = serialise(diagram, ["a"]).split("\n")[1]
    assert line == "N|a|box|yes\\pno"
    assert len(line.split("|")) == 4


# -- determinism, the property most likely to break silently ---------------------------------


def test_identical_across_repeated_calls():
    diagram, expected = _diagram(), serialise(_diagram(), TRAVERSAL)
    assert all(serialise(diagram, TRAVERSAL) == expected for _ in range(50))


def test_invariant_to_node_and_edge_list_order():
    expected = serialise(_diagram(), TRAVERSAL)
    rng = random.Random(0)
    for _ in range(50):
        shuffled = _diagram()
        rng.shuffle(shuffled["nodes"])
        rng.shuffle(shuffled["edges"])
        assert serialise(shuffled, TRAVERSAL) == expected


def test_invariant_to_parallel_edges_with_no_id():
    """The inherited sort key ended at `str(e.get("id") or "")`, so two id-less parallel edges
    were ordered by `list.sort`'s stability - i.e. by input position. This is that bug."""
    nodes = [{"id": "a", "shape": "box"}, {"id": "b", "shape": "box"}]
    edges = [{"src": "a", "dst": "b", "label": "x"}, {"src": "a", "dst": "b", "label": "y"}]
    forward = serialise({"nodes": nodes, "edges": edges}, ["a", "b"])
    backward = serialise({"nodes": nodes, "edges": list(reversed(edges))}, ["a", "b"])
    assert forward == backward


def _shuffle_keys(value, rng):
    if isinstance(value, dict):
        items = [(k, _shuffle_keys(v, rng)) for k, v in value.items()]
        rng.shuffle(items)
        return dict(items)
    if isinstance(value, list):
        return [_shuffle_keys(v, rng) for v in value]
    return value


def test_invariant_to_dict_key_insertion_order():
    expected = serialise(_diagram(), TRAVERSAL)
    rng = random.Random(7)
    for _ in range(50):
        assert serialise(_shuffle_keys(_diagram(), rng), TRAVERSAL) == expected


PROBE = textwrap.dedent(
    """
    import hashlib, json, sys
    sys.path.insert(0, sys.argv[1])
    from src.codegen.serialise import serialise
    diagram = json.loads(sys.argv[2])
    print(hashlib.sha256(serialise(diagram, ["n0", "n1", "n2"]).encode()).hexdigest())
    """
)


@pytest.mark.parametrize("seed", ["0", "1", "12345", "random"])
def test_identical_across_pythonhashseed_in_a_real_subprocess(seed):
    """A set or dict iteration in the writer only shows up here - the seed is fixed for the
    life of an interpreter, so an in-process assertion of this cannot fail."""
    env = dict(os.environ, PYTHONHASHSEED=seed)
    baseline = serialise(_diagram(), TRAVERSAL)
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT), json.dumps(_diagram())],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    import hashlib

    assert out.stdout.strip() == hashlib.sha256(baseline.encode()).hexdigest()


def test_determinism_over_the_real_corpus(corpus):
    for diagram, order in corpus:
        first = serialise(diagram, order)
        rng = random.Random(3)
        shuffled = _shuffle_keys(diagram, rng)
        rng.shuffle(shuffled["nodes"])
        rng.shuffle(shuffled["edges"])
        assert serialise(shuffled, order) == first


# -- round trip: what it preserves, stated as an equality ------------------------------------


def test_round_trip_of_the_fixture():
    assert parse(serialise(_diagram(), TRAVERSAL)) == prune(_diagram(), TRAVERSAL)


def test_round_trip_over_real_corpus(corpus):
    for diagram, order in corpus:
        assert parse(serialise(diagram, order)) == prune(diagram, order)


def test_round_trip_preserves_every_field_the_format_claims():
    recovered = parse(serialise(_diagram(), TRAVERSAL))
    assert set(recovered["nodes"][0]) == set(PRUNED_NODE_FIELDS)
    assert set(recovered["edges"][0]) == set(PRUNED_EDGE_FIELDS)
    assert recovered["diagram_type"] == "flowchart"
    assert recovered["traversal"] == ["n0", "n1", "n2", "orphan"]


def test_undirected_edges_survive_the_round_trip(corpus):
    diagram = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [{"src": "a", "dst": "b", "directed": False}],
    }
    assert parse(serialise(diagram, ["a", "b"]))["edges"][0]["directed"] is False


def test_ids_in_the_corpus_carry_no_whitespace(corpus):
    """`O` and `R` are space-separated, so an id containing a space would be unparseable. This
    asserts the corpus assumption rather than leaving it implicit."""
    for diagram, _ in corpus:
        for node in diagram.get("nodes") or []:
            assert not any(c.isspace() or c == ">" for c in str(node["id"]))


# -- the benchmark plumbing ------------------------------------------------------------------


def test_prune_and_compact_carry_the_same_information(corpus):
    """The benchmark is only honest if the JSON baseline is not charged for extra fields."""
    for diagram, order in corpus:
        pruned = prune(diagram, order)
        assert parse(serialise(diagram, order)) == pruned


def test_every_variant_is_non_empty_and_deterministic(corpus):
    for diagram, order in corpus[:5]:
        first = variants(diagram, order)
        assert set(first) == {
            "raw_json",
            "pruned_json",
            "short_key_json",
            "keyed_lines",
            "compact_header",
            "compact",
        }
        assert all(v for v in first.values())
        assert variants(diagram, order) == first


def test_compact_is_shorter_than_pruned_json_on_every_real_diagram(corpus):
    for diagram, order in corpus:
        v = variants(diagram, order)
        assert len(v["compact"]) < len(v["pruned_json"])
