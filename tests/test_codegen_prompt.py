"""Phase 12.2.4 - the frozen prompt template: pinned bytes, a contract the targets obey."""

from __future__ import annotations

import pytest

from src.codegen import pairs, prompt, schema, synthetic, targets

SPLITS = pytest.mark.skipif(
    not (pairs.PAIRS_DIR / "train.jsonl").is_file(), reason="needs the merged pair files"
)


def test_template_is_pinned_byte_for_byte():
    assert prompt.pin_digest() == prompt.PINNED_SHA256, "wording changed: bump TEMPLATE_ID"
    assert prompt.TEMPLATE_ID == "dreamscript-codegen/v1"


def test_example_record_is_a_valid_pair_whose_target_is_the_real_emitter_output():
    assert schema.validate(prompt.EXAMPLE_RECORD) == []
    parsed = __import__("src.codegen.serialise", fromlist=["parse"]).parse(
        prompt.EXAMPLE_RECORD["ir_text"]
    )
    diagram = {
        "id": pairs.CANONICAL_ID,
        "diagram_type": "flowchart",
        "nodes": parsed["nodes"],
        "edges": parsed["edges"],
    }
    emitted = targets.emit_flowchart_python(diagram, parsed["traversal"])
    assert emitted == prompt.EXAMPLE_RECORD["target_code"]


@pytest.mark.parametrize("diagram_type", synthetic.TYPES)
def test_every_part_of_the_prompt_is_present(diagram_type):
    record = pairs.make_pair(synthetic.random_diagram(diagram_type, 2), "synthetic")
    system, user = prompt.build_messages(record)
    assert system["role"] == "system" and "Output contract" in system["content"]
    assert f"tagged {prompt.fence_for(record['language'])}" in system["content"]
    assert f"diagram_type: {record['diagram_type']}" in user["content"]
    assert record["ir_text"] in user["content"]
    assert "\nO|" in user["content"]  # the traversal travels inside the IR


@pytest.mark.parametrize("diagram_type", synthetic.TYPES)
def test_completion_round_trips_through_the_strict_extractor(diagram_type):
    record = pairs.make_pair(synthetic.random_diagram(diagram_type, 4), "synthetic")
    body = prompt.extract_code(prompt.completion(record))
    assert body.strip() == record["target_code"].strip()


@pytest.mark.parametrize(
    "reply",
    [
        "Here is the code:\n```python\nx = 1\n```",
        "```python\nx = 1\n```\nHope this helps",
        "```python\nx = 1\n```\n```python\ny = 2\n```",
        "x = 1",
        "```python x = 1```",
    ],
)
def test_contract_violations_extract_to_none(reply):
    assert prompt.extract_code(reply) is None


def test_whitespace_around_a_single_block_is_not_a_violation():
    assert prompt.extract_code("\n```python\nx = 1\n```\n\n") == "x = 1\n"


def test_training_example_shape():
    example = prompt.training_example(prompt.EXAMPLE_RECORD)
    assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
    assert example["messages"][-1]["content"] == example["completion"]


@SPLITS
def test_every_real_completion_obeys_the_contract():
    for record in pairs.load_pairs():
        body = prompt.extract_code(prompt.completion(record))
        assert body is not None and body.strip() == record["target_code"].strip()
