"""Phase 12.1.8 - the pair split, and the leak it exists to prevent.

The happy path here is cheap: `assign` copies a split off an index. The tests that earn their
keep are the ones that build a *leaking* input on purpose and assert it is caught, because a
writer leak is silent - every check passes, every number looks plausible, and the Phase 14 score
measures memorised penmanship. So each check in `summarize()["checks"]` gets a test that drives
it False, not only one that drives it True.
"""

from __future__ import annotations

import pytest

from src.codegen import splits as S

CORPUS = pytest.mark.skipif(
    not S.MANIFEST.is_file(), reason="needs data/processed/manifest.parquet"
)


def pair(diagram_id: str, **overrides) -> dict:
    """A 12.1.1-shaped pair record, with `split` unset."""
    record = {
        "diagram_id": diagram_id,
        "diagram_type": "flowchart",
        "ir_text": "n0 start; n1 end; e0 n0->n1",
        "traversal": ["n0", "n1"],
        "target_code": "def main():\n    pass\n",
        "language": "python",
        "source": "unit_test",
        "split": None,
    }
    record.update(overrides)
    return record


def index_of(*entries: tuple[str, str, str | None]) -> dict[str, dict]:
    """A stand-in split index: (diagram_id, split, scribe) triples."""
    return {
        diagram_id: {
            "split": split,
            "scribe": scribe,
            "source": "unit_test",
            "basis": "manifest",
        }
        for diagram_id, split, scribe in entries
    }


# ---------------------------------------------------------------------------------------
# adoption
# ---------------------------------------------------------------------------------------


def test_normalise_reads_val_and_validation_as_one_split():
    assert S._normalise("val") == "validation"
    assert S._normalise("validation") == "validation"
    assert S._normalise("VAL ") == "validation"
    assert S._normalise(None) is None
    assert S._normalise("nan") is None


def test_assign_adopts_the_index_rather_than_inventing_a_split():
    index = index_of(("d0", "train", "w0"), ("d1", "validation", "w1"), ("d2", "test", "w2"))
    out = S.assign([pair("d0"), pair("d1"), pair("d2")], index=index)
    assert [r["split"] for r in out] == ["train", "validation", "test"]
    assert {r["split_basis"] for r in out} == {"manifest"}
    assert [r["scribe"] for r in out] == ["w0", "w1", "w2"]


def test_assign_resolves_a_bare_id_against_a_namespaced_index_key():
    index = index_of(("hdbpmn/ex00/ex00_writer0001", "test", "hdbpmn:writer0001"))
    index["ex00_writer0001"] = index["hdbpmn/ex00/ex00_writer0001"]
    (out,) = S.assign([pair("ex00_writer0001")], index=index)
    assert out["split"] == "test"


def test_assign_copies_and_preserves_order_and_keys():
    given = [pair("d0"), pair("d1")]
    index = index_of(("d0", "train", "w0"), ("d1", "train", "w0"))
    out = S.assign(given, index=index)
    assert [r["diagram_id"] for r in out] == ["d0", "d1"]
    assert given[0]["split"] is None and "scribe" not in given[0]
    assert set(given[0]) <= set(out[0])
    assert out[0]["target_code"] == given[0]["target_code"]


def test_every_pair_is_assigned_one_of_the_three_splits():
    index = index_of(("d0", "train", "w0"))
    out = S.assign([pair("d0"), pair("unknown"), pair("s/x", source="synthetic")], index=index)
    assert all(r["split"] in S.SPLITS for r in out)


# ---------------------------------------------------------------------------------------
# scribe-less pairs: synthetic and merely anonymous
# ---------------------------------------------------------------------------------------


def test_synthetic_pairs_land_in_train_only():
    records = [
        pair(f"synthetic/g{i}", source=source) for i, source in enumerate(S.SYNTHETIC_SOURCES)
    ]
    out = S.assign(records, index={})
    assert {r["split"] for r in out} == {"train"}
    assert {r["split_basis"] for r in out} == {"synthetic_train_only"}


def test_a_synthetic_pair_stays_in_train_even_when_the_index_says_test():
    """Adoption stops at the pairs nobody wrote: the index cannot pull one into test."""
    index = index_of(("synthetic/g0", "test", None))
    (out,) = S.assign([pair("synthetic/g0", source="synthetic")], index=index)
    assert out["split"] == "train"


def test_a_pair_with_no_scribe_never_lands_in_val_or_test():
    """The sketch2code case: hand-drawn, but no writer was ever published for it."""
    out = S.assign([pair(f"s2c_{i}", source="sketch2code") for i in range(200)], index={})
    assert {r["split"] for r in out} == {"train"}
    assert {r["split_basis"] for r in out} == {"no_scribe_train_only"}
    assert {r["scribe"] for r in out} == {None}


def test_a_scribed_pair_outside_the_index_is_derived_whole_by_writer():
    records = [
        pair(f"d{i}", scribe=f"w{i % 4}", source="new_source", diagram_type="state_machine")
        for i in range(40)
    ]
    out = S.assign(records, index={})
    assert {r["split_basis"] for r in out} == {"derived_scribe_disjoint"}
    by_writer: dict[str, set[str]] = {}
    for record in out:
        by_writer.setdefault(record["scribe"], set()).add(record["split"])
    assert all(len(v) == 1 for v in by_writer.values()), by_writer


# ---------------------------------------------------------------------------------------
# the leak tests: build it broken, assert it is caught
# ---------------------------------------------------------------------------------------


def test_summarize_catches_a_scribe_on_both_sides_of_the_split():
    leaking = [
        pair("d0", scribe="writer0065", split="train"),
        pair("d1", scribe="writer0065", split="test"),
        pair("d2", scribe="writer0002", split="validation"),
    ]
    summary = S.summarize(leaking, index={})
    assert summary["checks"]["no_scribe_spans_splits"] is False
    assert summary["leaking_scribes"] == ["writer0065"]


def test_a_poisoned_index_that_leaks_a_writer_is_caught_end_to_end():
    """The failure mode that matters: the *source* of the split is wrong, not the summary.

    If `assign` ever adopted a split that puts one writer's two diagrams on opposite sides, the
    per-record output would look entirely normal. Only the check catches it - so the check is
    driven from a poisoned index rather than from hand-set `split` fields.
    """
    index = index_of(
        ("d0", "train", "hdbpmn:writer0065"),
        ("d1", "test", "hdbpmn:writer0065"),
        ("d2", "validation", "hdbpmn:writer0002"),
    )
    out = S.assign([pair("d0"), pair("d1"), pair("d2")], index=index)
    assert [r["split"] for r in out] == ["train", "test", "validation"]  # silent, and wrong
    summary = S.summarize(out, index=index)
    assert summary["checks"]["no_scribe_spans_splits"] is False
    assert summary["leaking_scribes"] == ["hdbpmn:writer0065"]
    assert summary["scribes_per_split"]["train"] == 1
    assert summary["scribes_per_split"]["test"] == 1


def test_summarize_catches_one_diagram_in_two_splits():
    duplicated = [
        pair("d0", scribe="w0", split="train"),
        pair("d0", scribe="w0", split="test"),
    ]
    summary = S.summarize(duplicated, index={})
    assert summary["checks"]["no_diagram_spans_splits"] is False
    assert summary["leaking_diagrams"] == ["d0"]


def test_summarize_catches_a_scribeless_pair_in_test():
    records = [
        pair("d0", scribe=None, split="test", source="sketch2code"),
        pair("d1", scribe="w1", split="train"),
    ]
    summary = S.summarize(records, index={})
    assert summary["checks"]["heldout_pairs_all_have_a_scribe"] is False
    assert summary["scribeless_per_split"]["test"] == 1


def test_summarize_catches_a_synthetic_pair_outside_train():
    records = [pair("synthetic/g0", source="synthetic", scribe=None, split="validation")]
    summary = S.summarize(records, index={})
    assert summary["checks"]["synthetic_only_in_train"] is False


def test_summarize_catches_an_unassigned_pair():
    summary = S.summarize([pair("d0", scribe="w0", split=None)], index={})
    assert summary["checks"]["all_pairs_assigned"] is False


def test_summarize_catches_a_missing_split():
    records = [pair(f"d{i}", scribe=f"w{i}", split="train") for i in range(3)]
    summary = S.summarize(records, index={})
    assert summary["checks"]["three_splits_present"] is False


def test_a_clean_split_passes_every_check():
    index = index_of(
        *[(f"d{i}", ("train", "validation", "test")[i % 3], f"w{i}") for i in range(30)]
    )
    out = S.assign([pair(f"d{i}") for i in range(30)], index=index)
    summary = S.summarize(out, index=index)
    assert all(summary["checks"].values()), summary["checks"]


# ---------------------------------------------------------------------------------------
# the real corpus
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_split():
    records = S._pairs_from_targets_index() + S.synthetic_stand_ins(600)
    assigned = S.assign(records)
    return assigned, S.summarize(assigned)


@CORPUS
def test_corpus_split_passes_every_check(corpus_split):
    _, summary = corpus_split
    assert all(summary["checks"].values()), summary["checks"]


@CORPUS
def test_corpus_split_sizes_are_the_docstring_numbers(corpus_split):
    _, summary = corpus_split
    assert summary["total"] == 2077
    assert summary["sizes"] == {"train": 1739, "validation": 176, "test": 162}
    assert summary["scribed_sizes"] == {"train": 655, "validation": 176, "test": 162}
    assert summary["pairs_with_scribe"] == 993
    assert summary["distinct_scribes"] == 130
    assert summary["distinct_diagrams"] == 2077
    assert summary["by_basis"] == {
        "manifest": 693,
        "synthetic_train_only": 600,
        "no_scribe_train_only": 484,
        "fa_writer_split": 300,
    }


@CORPUS
def test_corpus_val_and_test_are_entirely_scribed(corpus_split):
    records, summary = corpus_split
    assert summary["scribeless_per_split"]["validation"] == 0
    assert summary["scribeless_per_split"]["test"] == 0
    held_out = [r for r in records if r["split"] in ("validation", "test")]
    assert held_out and all(r["scribe"] for r in held_out)
    assert not any(S.is_synthetic(r) for r in held_out)


@CORPUS
def test_corpus_scribe_and_diagram_sets_are_pairwise_disjoint(corpus_split):
    """Recomputed from the records rather than read off `summarize`, so both must agree."""
    records, _ = corpus_split
    scribes = {name: set() for name in S.SPLITS}
    diagrams = {name: set() for name in S.SPLITS}
    for record in records:
        diagrams[record["split"]].add(record["diagram_id"])
        if record["scribe"]:
            scribes[record["split"]].add(record["scribe"])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert scribes[a] & scribes[b] == set()
        assert diagrams[a] & diagrams[b] == set()
    assert len(scribes["train"] | scribes["validation"] | scribes["test"]) == 130


@CORPUS
def test_corpus_split_is_deterministic():
    records = S._pairs_from_targets_index()
    first = [(r["diagram_id"], r["split"]) for r in S.assign(records)]
    second = [(r["diagram_id"], r["split"]) for r in S.assign(records)]
    assert first == second


@CORPUS
def test_adopting_fa_agrees_with_deriving_it():
    agreement = S.fa_derivation_agreement()
    assert agreement["available"] is True
    assert agreement["diagrams"] == 300
    assert agreement["writers"] == 25
    assert agreement["disagree"] == 0, agreement["examples"]


@CORPUS
def test_label_crop_index_agrees_with_the_manifest():
    agreement = S.crop_index_agreement()
    if not agreement["available"]:
        pytest.skip("no label-crop index")
    assert agreement["resolved"] == 693
    assert agreement["disagree"] == 0, agreement["examples"]


@CORPUS
def test_main_exits_zero():
    assert S.main([]) == 0
