"""src.ingest - corpus intake (Phase 1).

Loads raw photos and public datasets, builds data/manifest.parquet, deduplicates by
perceptual hash, produces scribe-disjoint splits, and applies the augmentation policy.
"""

from __future__ import annotations

PHASE = "1"
__all__: list[str] = []
