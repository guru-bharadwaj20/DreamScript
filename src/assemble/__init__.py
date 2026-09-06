"""src.assemble - graph assembly and the validated IR (Phase 10).

Detections, strokes and recognised text in; a validated, typed, confidence-carrying
DreamScript graph out. Node instantiation and text binding, skeleton edge tracing with
broken-arrow repair, direction and label resolution, crossing disambiguation, containment
nesting, then semantic validation, repair, loop marking, diffing and serialization.

This is the contract between vision and codegen: everything upstream produces evidence,
everything downstream consumes `Diagram`.
"""

from __future__ import annotations

PHASE = "10"
__all__: list[str] = []
