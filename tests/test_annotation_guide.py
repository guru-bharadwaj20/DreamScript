"""Phase 2.2.3 - the annotation guide must keep covering what the schema can express.

A guide that falls behind the schema is worse than no guide: annotators follow it, and the
fields it forgot get filled inconsistently or not at all. These are cheap coverage checks, not
a substitute for reading it.
"""

from __future__ import annotations

import pytest

from src.ir import labelstudio
from src.utils.config import ROOT

GUIDE = ROOT / "docs" / "annotation_guide.md"


@pytest.fixture(scope="module")
def text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_guide_exists(text):
    assert len(text) > 2000


@pytest.mark.parametrize("field", ["unresolved_edges", "crossed_out", "low_conf_text"])
def test_every_ambiguity_field_has_a_rule(text, field):
    assert field in text


@pytest.mark.parametrize("flag", ["crossed-out", "text-uncertain", "shape-uncertain"])
def test_every_annotator_flag_is_explained(text, flag):
    assert flag in text


def test_every_shape_in_the_vocabulary_appears_in_the_shape_table(text):
    from src.ir.vocab import SHAPES

    table = text.split("## 2. Shape before role")[1].split("## 3.")[0]
    for shape in SHAPES:
        assert f"`{shape}`" in table, f"{shape} has no rule for when to choose it"


def test_the_hard_cases_named_in_the_plan_are_covered(text):
    """plan.md 2.2.3 names two by hand; both must have their own rule."""
    lowered = text.lower()
    assert "broken arrow" in lowered
    assert "overlapping" in lowered


def test_guide_separates_shape_from_role(text):
    """The single most damaging labelling error, so it has to be stated, not implied."""
    assert "Label what is on the paper, not what the writer meant." in text


def test_relation_types_in_the_config_are_all_described(text):
    for name in labelstudio.RELATIONS:
        assert name in text, f"relation `{name}` is offered in the UI but not in the guide"
