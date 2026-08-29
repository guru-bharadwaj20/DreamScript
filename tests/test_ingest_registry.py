"""Phase 1.1 acceptance tests — the dataset registry, audit, and availability claims.

These tests encode the *conclusions* of the Phase 1.1 acquisition so they cannot rot silently:
if a dead source comes back to life, or a license changes, a test tells us.
"""

from __future__ import annotations

import pytest

from src.ingest import audit
from src.ingest.registry import REGISTRY, Availability, Redistribution

NETWORK = pytest.mark.slow  # these hit the network; excluded from `make test-fast`


def test_every_planned_source_is_registered():
    expected = {
        "hdbpmn",  # 1.1.1
        "fc_bresler",  # 1.1.2 (unavailable)
        "flowchartseg",  # 1.1.2 substitute
        "didi",  # 1.1.3
        "iam_line",  # 1.1.4
        "sketch2code_ms",  # 1.1.5 (unavailable)
        "sketch2code",  # 1.1.5 substitute
    }
    assert expected <= set(REGISTRY)


def test_every_dataset_names_a_data_card():
    for ds in REGISTRY.values():
        assert ds.data_card.startswith("docs/data_cards/")


def test_unavailable_sources_have_a_documented_reason():
    for ds in REGISTRY.values():
        if ds.availability is Availability.UNAVAILABLE:
            assert ds.notes, f"{ds.slug} is unavailable but gives no reason"
            assert "Substituted" in ds.notes or "substitut" in ds.notes.lower()


def test_unknown_redistribution_is_treated_as_no():
    """The audit must never grant raw-file publication rights to an unlicensed source."""
    text = audit.PUBLISHABLE[Redistribution.UNKNOWN]
    assert "metrics and models only" in text
    assert "raw" not in text.split("-")[0]


def test_audit_report_generates():
    path = audit.write_report()
    content = path.read_text(encoding="utf-8")
    assert "# License and Availability Audit" in content
    for slug in ["hdbpmn", "didi", "iam_line", "sketch2code", "flowchartseg"]:
        assert REGISTRY[slug].name in content


def test_audit_flags_the_unlicensed_substitute():
    """flowchartseg has no declared license; the audit must keep saying so until resolved."""
    if REGISTRY["flowchartseg"].present:
        assert "flowchartseg" in audit.unresolved()


@NETWORK
def test_fc_original_still_unavailable():
    """If FC-A/FC-B is restored upstream, this fails and we reconsider the substitution."""
    from src.ingest.datasets.flowcharts import probe_original

    assert probe_original().get("download") == 404


@NETWORK
def test_microsoft_sketch2code_demo_still_gone():
    from src.ingest.datasets.sketch2code import probe_microsoft

    assert probe_microsoft().get("demo") != 200
