"""Phase 13.4 - the classifier rung is wired on a control, so the control has to be real.

The arms that matter are not the headline. What these tests pin is that `cross_domain` asks a
question a renderer detector cannot pass, that the identity columns never reach the model, and
that the verdict follows the control rather than the score everyone would rather quote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import fallback

#: `_model()` is HistGradientBoostingClassifier, whose `min_samples_leaf` defaults to 20. A
#: fixture smaller than that cannot split at all and silently collapses to a majority-class
#: constant - which passes or fails for reasons that have nothing to do with what is being tested.
MIN_ROWS_PER_GROUP = 30


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["domain"] = df["source"].map(fallback.DOMAIN)
    df["group"] = df["scribe_id"].fillna(df["id"])
    return df


def page(page_id: str, source: str, diagram_type: str, signal: float, **extra) -> dict:
    """One row whose single real feature is `signal`, so a test can decide what is learnable."""
    return {
        "id": page_id,
        "source": source,
        "diagram_type": diagram_type,
        "split": "train",
        "scribe_id": extra.get("scribe_id"),
        "adverse": False,
        "synthetic": False,
        "node_count": signal,
        "edge_count": signal * 2,
    }


def test_identity_columns_never_reach_the_model():
    """4.1.5's leak: a model trained on `source` has learned the corpus, not the drawing."""
    df = frame([page("a/1", "hdbpmn", "flowchart", 1.0)])
    _, columns = fallback.features(df)
    for banned in fallback.IDENTITY:
        assert banned not in columns
    assert "domain" not in columns and "group" not in columns
    assert "node_count" in columns


def test_every_source_declares_a_domain():
    """A source with no declared domain would silently become NaN and drop out of the control."""
    table = fallback.load() if fallback.TABLE.is_file() else None
    if table is None:
        pytest.skip("needs data/features/handcrafted.parquet")
    assert table["domain"].notna().all()
    assert set(table["source"]) <= set(fallback.DOMAIN)


def test_cross_domain_trains_with_the_domain_held_constant():
    """The training set must not let domain stand in for the label, or the arm proves nothing."""
    rows = [page(f"fs/{i}", "flowchartseg", "flowchart", 1.0 + i * 0.01) for i in range(40)]
    rows += [page(f"fa/{i}", "fa_bresler", "state_machine", 5.0 + i * 0.01) for i in range(40)]
    rows += [page(f"hd/{i}", "hdbpmn", "flowchart", 1.0 + i * 0.01) for i in range(30)]
    result = fallback.cross_domain(frame(rows))
    assert result["train"]["sources"] == ["flowchartseg", "fa_bresler"]
    assert result["test"]["source"] == "hdbpmn"
    # Both training sources are rendered, so within training the domain is a constant.
    assert result["train"]["n"] == 80
    assert result["test"]["n"] == 30


def test_a_rule_that_does_not_transfer_fails_the_cross_domain_arm():
    """The arm's whole purpose: a rule learned in one domain that is wrong in the other.

    Note what this fixture cannot do, because it is the point of the design: a
    "rendered => state_machine" rule is **unlearnable** from this training set, since both
    training sources are rendered and the domain is therefore constant. What the arm can still
    catch is a model whose learned feature happens to track the renderer rather than the shape -
    so here the state machines read 9.0 in training, and the photographed flowcharts also read
    9.0. A model keying on that feature calls every one of them a state machine.
    """
    rows = [page(f"fs/{i}", "flowchartseg", "flowchart", 1.0) for i in range(40)]
    rows += [page(f"fa/{i}", "fa_bresler", "state_machine", 9.0) for i in range(40)]
    rows += [page(f"hd/{i}", "hdbpmn", "flowchart", 9.0) for i in range(30)]
    result = fallback.cross_domain(frame(rows))
    assert result["accuracy"] < fallback.CROSS_DOMAIN_FLOOR
    assert result["predicted_counts"].get("state_machine", 0) > 0


def test_a_shape_learner_passes_the_cross_domain_arm():
    """The mirror: when the feature really is the diagram type, the arm must let it through."""
    rows = [page(f"fs/{i}", "flowchartseg", "flowchart", 1.0) for i in range(40)]
    rows += [page(f"fa/{i}", "fa_bresler", "state_machine", 9.0) for i in range(40)]
    rows += [page(f"hd/{i}", "hdbpmn", "flowchart", 1.0) for i in range(30)]
    result = fallback.cross_domain(frame(rows))
    assert result["accuracy"] >= fallback.CROSS_DOMAIN_FLOOR


def test_the_floor_is_above_the_majority_class():
    """The test set is all one class, so anything at or under 0.5 is not evidence of transfer."""
    assert fallback.CROSS_DOMAIN_FLOOR > 0.5


def test_ocr_chain_degrades_without_ever_running_out_of_identifiers():
    rungs = fallback.ocr_rungs()
    assert len(rungs) >= 3
    # Every rung must produce a usable identifier; a chain whose last rung is empty is not a
    # fallback, it is a failure with extra steps.
    for rung in rungs:
        assert rung["identifier"].strip()
        assert rung["verified"]


def test_degradations_cover_every_kind_the_robustness_suite_sweeps():
    """If 14.5 gains a degradation, this arm should not quietly keep testing the old four."""
    from src.eval.robust import SWEEPS

    assert {kind for kind, _ in fallback.DEGRADATIONS} == set(SWEEPS)


def test_every_degradation_severity_is_one_the_sweep_actually_uses():
    from src.eval.robust import SWEEPS

    for kind, severity in fallback.DEGRADATIONS:
        assert severity in SWEEPS[kind], (kind, severity)


def test_render_reports_the_worst_degradation_not_only_the_mean():
    """A mean over five degradations hides the one that breaks the rung."""
    result = {
        "table": {"rows": 10, "classes": {"flowchart": 5, "state_machine": 5}},
        "arms": {
            "full": {"accuracy": 0.99, "macro_f1": 0.99},
            "rendered_only": {"accuracy": 0.99, "macro_f1": 0.99},
            "domain_probe": {"accuracy": 0.99, "macro_f1": 0.99},
            "cross_domain": {
                "accuracy": 0.86,
                "test": {"n": 704},
                "predicted_counts": {"flowchart": 604},
                "reading": "r",
            },
            "degraded_rendering": {
                "per_degradation": {
                    "resolution": {
                        "severity": 0.5,
                        "pages": 120,
                        "routed_state_machine": 63,
                        "accuracy": 0.525,
                    }
                },
                "mean": 0.84,
                "worst": 0.525,
            },
        },
        "verdict": {"classifier_rung_wired": True, "criterion": "c"},
        "ocr_chain": [],
        "model_chain": {"where": "w", "behaviour": "b"},
    }
    text = fallback.render(result)
    assert "0.5250" in text
    assert "worst" in text.lower()


def test_the_verdict_reads_the_control_and_not_the_headline(monkeypatch):
    """The wiring decision must come from `cross_domain`, however good the headline looks."""
    monkeypatch.setattr(fallback, "load", lambda: frame([page("a/1", "hdbpmn", "flowchart", 1.0)]))
    monkeypatch.setattr(
        fallback, "out_of_fold", lambda *_, **__: {"accuracy": 1.0, "macro_f1": 1.0}
    )
    monkeypatch.setattr(fallback, "domain_probe", lambda *_: {"accuracy": 1.0, "macro_f1": 1.0})
    monkeypatch.setattr(
        fallback,
        "degraded_rendering",
        lambda *_, **__: {"per_degradation": {}, "mean": None, "worst": None},
    )
    # A perfect headline and a failing control: the verdict must follow the control.
    monkeypatch.setattr(
        fallback,
        "cross_domain",
        lambda *_: {
            "accuracy": 0.51,
            "test": {"n": 10},
            "predicted_counts": {"state_machine": 10},
            "reading": "r",
        },
    )
    assert fallback.collect()["verdict"]["classifier_rung_wired"] is False

    monkeypatch.setattr(
        fallback,
        "cross_domain",
        lambda *_: {
            "accuracy": 0.86,
            "test": {"n": 10},
            "predicted_counts": {"flowchart": 9},
            "reading": "r",
        },
    )
    assert fallback.collect()["verdict"]["classifier_rung_wired"] is True


def test_features_are_finite_or_nan_never_infinite():
    """HistGradientBoosting takes NaN by design; an inf is a bug that would train silently."""
    df = frame([page("a/1", "hdbpmn", "flowchart", 1.0), page("a/2", "hdbpmn", "flowchart", 2.0)])
    X, _ = fallback.features(df)
    assert not np.isinf(X).any()
