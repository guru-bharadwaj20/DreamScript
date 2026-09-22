"""Phase 14.1 - the master table must read its numbers, not carry them.

The failure this suite is built against is a table that keeps printing a figure after the
artefact behind it has changed or gone. So the tests drive the collector over a temporary tree
whose contents they control, and assert that what comes out is what is on disk - including the
cases where nothing is on disk at all.
"""

from __future__ import annotations

import json

import pytest

from src.eval import master


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "reports").mkdir()
    return tmp_path


def write_json(tree, name, payload):
    path = tree / "reports" / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_json_reads_a_nested_key(tree):
    write_json(tree, "x.json", {"overall": {"median_ged": 13.0}})
    assert master.Json("reports/x.json", "overall", "median_ged").read(tree) == 13.0


def test_json_missing_file_and_missing_key_are_both_none(tree):
    write_json(tree, "x.json", {"overall": {}})
    assert master.Json("reports/x.json", "overall", "median_ged").read(tree) is None
    assert master.Json("reports/absent.json", "cer").read(tree) is None


def test_json_provenance_names_the_path_and_the_keys():
    assert master.Json("reports/x.json", "a", "b").provenance == "reports/x.json:a.b"


MD = """# report

| model | macro F1 | balanced accuracy |
| :--- | ---: | ---: |
| `logreg` | **0.7917** | 0.7891 |
| `majority` | **0.1237** | 0.2000 |

| model | CER |
| :--- | ---: |
| `crnn` | 0.6864 |
"""


def test_mdcell_reads_the_first_matching_table(tree):
    (tree / "reports" / "r.md").write_text(MD, encoding="utf-8")
    cell = master.MdCell("reports/r.md", "model", "logreg", "macro F1")
    assert cell.read(tree) == pytest.approx(0.7917)
    assert master.MdCell("reports/r.md", "model", "majority", "balanced accuracy").read(tree) == 0.2


def test_mdcell_is_strict_about_a_renamed_column_or_row(tree):
    (tree / "reports" / "r.md").write_text(MD, encoding="utf-8")
    assert master.MdCell("reports/r.md", "model", "logreg", "macro-F1").read(tree) is None
    assert master.MdCell("reports/r.md", "model", "logistic", "macro F1").read(tree) is None
    assert master.MdCell("reports/r.md", "estimator", "logreg", "macro F1").read(tree) is None


def test_percentages_become_fractions(tree):
    (tree / "reports" / "p.md").write_text(
        "| model | functional |\n| :--- | ---: |\n| `lora` | 70.4% |\n", encoding="utf-8"
    )
    assert master.MdCell("reports/p.md", "model", "lora", "functional").read(tree) == pytest.approx(
        0.704
    )


def test_collect_grades_against_the_target_in_both_directions(tree):
    write_json(tree, "a.json", {"cer": 0.2006, "map50": 0.9107})
    rows = (
        master.Row("ocr", "m", "CER", master.Json("reports/a.json", "cer"), 0.15, False),
        master.Row("detect", "d", "mAP", master.Json("reports/a.json", "map50"), 0.80),
        master.Row("ocr", "m", "WER", master.Json("reports/a.json", "wer")),
    )
    cells = master.collect(tree, rows)
    assert [cell["passes"] for cell in cells] == [False, True, None]
    assert [cell["available"] for cell in cells] == [True, True, False]


def test_a_missing_cell_renders_as_missing_and_never_as_a_number(tree):
    rows = (master.Row("ocr", "m", "CER", master.Json("reports/gone.json", "cer"), 0.15, False),)
    text = master.render(master.collect(tree, rows))
    row = [line for line in text.splitlines() if line.startswith("| m |")][0]
    value = row.split("|")[3]
    assert value.strip() == "*missing*"
    assert not any(char.isdigit() for char in value)


def test_every_declared_row_has_a_readable_provenance_string():
    for row in master.ROWS:
        assert row.source.provenance
        assert row.source.path.endswith((".json", ".md"))


def test_a_registry_row_cannot_carry_a_result(tree):
    """The drift this module exists to stop is a stored number, so ``Row`` has nowhere to put one."""
    fields = set(master.Row.__dataclass_fields__)
    assert "value" not in fields and "result" not in fields
    write_json(tree, "a.json", {"cer": 0.31})
    row = master.Row("ocr", "m", "CER", master.Json("reports/a.json", "cer"))
    assert master.collect(tree, (row,))[0]["value"] == 0.31
    write_json(tree, "a.json", {"cer": 0.29})
    assert master.collect(tree, (row,))[0]["value"] == 0.29


def test_the_real_tree_resolves_most_of_its_cells():
    cells = master.collect()
    assert len(cells) > 40
    assert sum(cell["available"] for cell in cells) >= len(cells) - 5


def test_render_is_deterministic_for_the_same_input(tree):
    write_json(tree, "a.json", {"cer": 0.2})
    rows = (master.Row("ocr", "m", "CER", master.Json("reports/a.json", "cer"), 0.15, False),)
    cells = master.collect(tree, rows)
    assert master.render(cells) == master.render(cells)
