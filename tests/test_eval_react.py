"""Phase 12.3.6 - React build + render in node: working components render, broken ones do not."""

from __future__ import annotations

import pytest

from src.codegen import pairs, quality, synthetic
from src.eval import react

NODE = pytest.mark.skipif(not react.toolchain_ready(), reason="needs node + esbuild + react")

EXPECTED = {
    "drop_closing_tag": "react.build_error",
    "undefined_identifier": "react.render_error",
    "missing_import": "react.render_error",
    "infinite_loop": "react.timeout",
    "no_export": "react.no_component",
    "invalid_expression": "react.build_error",
    "throws_in_render": "react.render_error",
    "returns_object": "react.render_error",
}


def component(seed: int) -> str:
    return pairs.make_pair(synthetic.random_diagram("wireframe", seed), "synthetic")["target_code"]


def test_unavailable_toolchain_is_reported_not_passed(monkeypatch):
    monkeypatch.setattr(react, "toolchain_ready", lambda: False)
    assert react.check(component(1))["kind"] == "react.unavailable"


def test_every_mutation_has_a_site_in_an_emitted_component():
    code = component(2)
    for kind in react.MUTATIONS:
        mutated = react.mutate(code, kind)
        assert mutated is not None and mutated != code, kind


@NODE
def test_emitted_components_render_to_html():
    results = react.check_many([component(s) for s in range(6)], keep_html=True)
    assert all(r["ok"] and r["html_bytes"] > 0 and r["html"].startswith("<div") for r in results)


@NODE
def test_each_injected_defect_is_rejected_with_its_own_kind():
    code = component(3)
    kinds = react.MUTATIONS
    results = react.check_many([react.mutate(code, k) for k in kinds], timeout_ms=300)
    assert {k: r["kind"] for k, r in zip(kinds, results, strict=True)} == EXPECTED


@NODE
def test_render_stage_catches_what_the_structural_parse_cannot():
    code = component(4)
    for kind in ("undefined_identifier", "missing_import", "throws_in_render", "returns_object"):
        mutated = react.mutate(code, kind)
        assert quality.check_react(mutated)[0], kind
        assert not react.check(mutated)["ok"], kind


@NODE
def test_build_only_mode_does_not_execute():
    looping = react.mutate(component(5), "infinite_loop")
    assert react.check(looping, render=False)["ok"]
