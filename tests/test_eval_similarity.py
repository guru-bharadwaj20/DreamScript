"""Phase 12.3.5 - the similarity metric, and the ways it is allowed to be wrong.

This metric's docstring makes a strong claim about itself: it punishes a harmless rename far
harder than a harmful dropped statement, so it must never be read as evidence of correctness.
These tests pin that ordering, because it is the reason the report says what it says - if a
change ever made CodeBLEU track behaviour, the surrounding prose would be wrong and should be
rewritten rather than quietly left in place.
"""

from __future__ import annotations

from src.eval import similarity as S

GOOD = '''def main():
    """Generated from a flowchart."""
    value = receive_order()
    if check_stock(value):
        ship_order(value)
    else:
        reject_order(value)
    return value
'''


def test_a_program_is_identical_to_itself_on_every_component():
    got = S.score(GOOD, GOOD)
    assert got["codebleu"] == 1.0
    assert got["exact_match"] is True
    assert got["edit_similarity"] == 1.0
    assert set(got["components"]) == set(S.WEIGHTS)


def test_reformatting_changes_nothing_but_a_dropped_docstring_does():
    reformatted = GOOD.replace("\n", "\n\n") + "   \n"
    assert S.score(reformatted, GOOD)["exact_match"] is True
    # Comments and docstrings are deliberately not stripped: 12.1.6's targets carry one.
    stripped = "\n".join(ln for ln in GOOD.split("\n") if '"""' not in ln)
    assert S.score(stripped, GOOD)["exact_match"] is False


def test_the_metric_punishes_a_harmless_rename_harder_than_a_harmful_deletion():
    """The headline finding, and the reason 12.3.3 is the metric that settles correctness."""
    renamed = GOOD.replace("value", "v0")
    lines = [ln for ln in GOOD.split("\n") if ln.strip()]
    del lines[len(lines) // 2]
    dropped = "\n".join(lines)

    assert S.score(renamed, GOOD)["codebleu"] < S.score(dropped, GOOD)["codebleu"]


def test_reordering_destroys_behaviour_and_the_ngram_components_barely_notice():
    reversed_code = "\n".join(reversed([ln for ln in GOOD.split("\n") if ln.strip()]))
    got = S.score(reversed_code, GOOD)
    assert got["codebleu"] > 0.5
    # Only edit similarity, which makes no semantic claim, sees the damage.
    assert got["edit_similarity"] < got["codebleu"]


def test_an_undefined_component_is_dropped_rather_than_scored_zero():
    """A perfect SQL answer must score 1.0, not 0.5 for lacking a Python AST."""
    ddl = "CREATE TABLE customer(id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
    got = S.score(ddl, ddl, language="sql")
    assert got["codebleu"] == 1.0
    assert got["components"] == ["ngram_match", "weighted_ngram_match"]
    assert got["parsed"] is None


def test_unparseable_python_keeps_its_token_components_and_loses_the_tree_ones():
    got = S.score("def (:", GOOD)
    assert got["parsed"] is False
    assert "syntax_match" not in got["components"]
    assert got["codebleu"] < 0.5


def test_a_keyword_is_worth_more_than_a_name():
    """Weighted n-gram is the component that separates `while` from a variable spelling."""
    keyword_wrong = GOOD.replace("if check_stock(value):", "while check_stock(value):")
    name_wrong = GOOD.replace("check_stock", "check_inventory")
    a = S.score(keyword_wrong, GOOD)
    b = S.score(name_wrong, GOOD)
    assert a["weighted_ngram_match"] < b["weighted_ngram_match"]


def test_score_many_reports_per_language_and_macro_means():
    got = S.score_many([(GOOD, GOOD, "python"), ("SELECT 1;", "SELECT 2;", "sql")])
    assert got["n"] == 2
    assert set(got["by_language"]) == {"python", "sql"}
    assert got["by_language"]["python"]["codebleu"] == 1.0
    assert len(got["per_row"]) == 2


def test_edit_distance_is_symmetric_and_zero_only_on_equality():
    assert S.edit_distance("abc", "abc") == 0
    assert S.edit_distance("abc", "abd") == S.edit_distance("abd", "abc") == 1
    assert S.edit_distance("", "abc") == 3


def test_an_empty_candidate_is_scored_rather_than_crashing():
    got = S.score("", GOOD)
    assert got["codebleu"] == 0.0
    assert got["exact_match"] is False
