"""Phase 1.1 dataset downloaders. Importing this package registers every dataset."""

from __future__ import annotations

from src.ingest.datasets import flowcharts, hdbpmn  # noqa: F401

__all__ = ["flowcharts", "hdbpmn"]
