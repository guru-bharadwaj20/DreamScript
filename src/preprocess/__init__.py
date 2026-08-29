"""src.preprocess - photo to clean strokes (Phase 3).

Page detection, perspective rectification, illumination correction, binarization,
deskew, ruled-line suppression, skeletonization, and geometric primitive extraction.
"""

from __future__ import annotations

PHASE = "3"
__all__: list[str] = []
