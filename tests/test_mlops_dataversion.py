"""Phase 15.2 - the versioning report has to be checkable, not reassuring.

The failure this row responds to is a store that was empty when it was needed. So the tests pin
that the inventory measures the tree rather than asserting about it, that an exclusion always
carries a reason, and that `--check` can actually fail - a check that cannot fail is decoration.
"""

from __future__ import annotations

import json

import pytest

from src.mlops import dataversion


def test_every_tracked_path_declares_why_it_is_tracked():
    for path, reason in dataversion.TRACKED_REASON.items():
        assert path.startswith("data/")
        assert len(reason) > 20, path


def test_every_exclusion_declares_why_it_is_excluded():
    for path, reason in dataversion.UNTRACKED_REASON.items():
        assert path.startswith("data/")
        assert len(reason) > 20, path


def test_nothing_is_both_tracked_and_excluded():
    assert not set(dataversion.TRACKED_REASON) & set(dataversion.UNTRACKED_REASON)


def test_sizes_are_measured_from_the_tree(tmp_path, monkeypatch):
    (tmp_path / "data" / "x").mkdir(parents=True)
    (tmp_path / "data" / "x" / "a.bin").write_bytes(b"0" * 1_500_000)
    (tmp_path / "data" / "x" / "b.bin").write_bytes(b"0" * 500_000)
    monkeypatch.setattr(dataversion, "ROOT", tmp_path)
    monkeypatch.setattr(dataversion, "TRACKED_REASON", {"data/x": "a reason long enough to pass"})
    monkeypatch.setattr(dataversion, "UNTRACKED_REASON", {})

    inventory = dataversion.inventory()

    assert inventory["tracked"][0]["megabytes"] == pytest.approx(2.0)
    assert inventory["tracked"][0]["exists"] is True


def test_a_path_that_is_gone_is_reported_as_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(dataversion, "ROOT", tmp_path)
    monkeypatch.setattr(dataversion, "TRACKED_REASON", {"data/gone": "a reason long enough here"})
    monkeypatch.setattr(dataversion, "UNTRACKED_REASON", {})
    row = dataversion.inventory()["tracked"][0]
    assert row["exists"] is False
    assert row["megabytes"] is None


def test_the_pointer_file_is_reported_when_present(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "t.parquet").write_bytes(b"x")
    (tmp_path / "data" / "t.parquet.dvc").write_text("outs: []", encoding="utf-8")
    monkeypatch.setattr(dataversion, "ROOT", tmp_path)
    monkeypatch.setattr(
        dataversion, "TRACKED_REASON", {"data/t.parquet": "a reason long enough to pass"}
    )
    monkeypatch.setattr(dataversion, "UNTRACKED_REASON", {})
    assert dataversion.inventory()["tracked"][0]["pointer"] == "t.parquet.dvc"


def test_check_fails_when_a_remote_is_missing_data(monkeypatch, tmp_path):
    monkeypatch.setattr(dataversion, "ROOT", tmp_path)
    monkeypatch.setattr(dataversion, "remotes", lambda: {"raw": "somewhere"})
    monkeypatch.setattr(dataversion, "TRACKED_REASON", {})
    monkeypatch.setattr(dataversion, "UNTRACKED_REASON", {})
    monkeypatch.setattr(
        dataversion,
        "store_status",
        lambda: {"raw": {"exit_code": 0, "up_to_date": False, "missing_entries": 3, "sample": []}},
    )
    out = tmp_path / "r.md"
    code = dataversion.main(["--check", "--out", str(out), "--json", str(tmp_path / "r.json")])
    assert code == 1
    assert "**no**" in out.read_text(encoding="utf-8")


def test_check_passes_when_every_remote_is_in_sync(monkeypatch, tmp_path):
    monkeypatch.setattr(dataversion, "ROOT", tmp_path)
    monkeypatch.setattr(dataversion, "remotes", lambda: {"raw": "somewhere"})
    monkeypatch.setattr(dataversion, "TRACKED_REASON", {})
    monkeypatch.setattr(dataversion, "UNTRACKED_REASON", {})
    monkeypatch.setattr(
        dataversion,
        "store_status",
        lambda: {"raw": {"exit_code": 0, "up_to_date": True, "missing_entries": 0, "sample": []}},
    )
    code = dataversion.main(
        ["--check", "--out", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")]
    )
    assert code == 0


def test_the_report_says_the_remotes_are_not_off_machine_durability(monkeypatch, tmp_path):
    monkeypatch.setattr(dataversion, "ROOT", tmp_path)
    monkeypatch.setattr(dataversion, "remotes", lambda: {"raw": str(tmp_path)})
    monkeypatch.setattr(dataversion, "store_status", lambda: {})
    text = dataversion.render(dataversion.collect(check_store=False))
    assert "off-machine durability" in text


def test_the_json_report_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(dataversion, "ROOT", tmp_path)
    monkeypatch.setattr(dataversion, "remotes", lambda: {})
    monkeypatch.setattr(dataversion, "store_status", lambda: {})
    monkeypatch.setattr(dataversion, "TRACKED_REASON", {})
    monkeypatch.setattr(dataversion, "UNTRACKED_REASON", {})
    path = tmp_path / "r.json"
    dataversion.main(["--out", str(tmp_path / "r.md"), "--json", str(path)])
    assert json.loads(path.read_text(encoding="utf-8"))["totals"]["tracked_paths"] == 0
