"""src.embed - learned image representations (Phase 6.1).

Frozen convolutional and transformer backbones turned into a cached embedding table, so
Phase 6's MLP and SVM can be fitted on a representation the geometry of Phase 4 never had to
produce. Nothing here is trained; the backbones are used as fixed functions and the only
question asked of them is whether their features carry the label.
"""

from __future__ import annotations

PHASE = "6"
__all__: list[str] = []
