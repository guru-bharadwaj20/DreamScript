"""Phase 1.1 dataset downloaders. Importing this package registers every dataset."""

from __future__ import annotations

from src.ingest.datasets import didi, extra, flowcharts, hdbpmn, iam, sketch2code  # noqa: F401

__all__ = ["didi", "extra", "flowcharts", "hdbpmn", "iam", "sketch2code"]
