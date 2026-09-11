"""Phase 12.3 - scoring: extraction, syntax, one denominator, hallucination counts."""

from __future__ import annotations

from src.llm import score
from tests.test_llm_functional import FLOW, GOOD


def test_code_of_takes_the_first_fence_or_the_reply() -> None:
    assert score.code_of("prose\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert score.code_of("x = 1") == "x = 1"
    assert score.code_of("```python\nx = 1\n") == "x = 1\n"  # unterminated at the token limit


def test_syntax_uses_compile_not_just_parse() -> None:
    assert score.syntax_ok("x = 1")[0]
    assert not score.syntax_ok("def f(a, a):\n    return a")[0]
    assert not score.syntax_ok("return 1")[0]


def test_hallucination_counts_dropped_and_invented_operations() -> None:
    clean = score.hallucination(GOOD, FLOW)
    assert clean["expected"] == 5 and clean["dropped"] == [] and clean["invented"] == []
    bad = GOOD.replace("send_pin(ctx)", "notify_the_regulator(ctx)")
    out = score.hallucination(bad, FLOW)
    assert out["dropped"] == ["p"] and out["invented"] == ["notify_the_regulator"]


def test_summary_rates_share_one_denominator() -> None:
    rows = [
        {
            "source": "hdbpmn",
            "contract": True,
            "syntax": ok,
            "executes": ok,
            "functional": False,
            "functional_reason": "x",
            "exec_kind": "ok",
            "hit_limit": False,
            "hallucination": {"expected": 2, "dropped": [], "invented": []},
        }
        for ok in (True, False, False, True)
    ]
    summary = score.summarise(rows)
    assert summary["n"] == 4 and summary["syntax"] == 0.5 and summary["executes"] == 0.5
