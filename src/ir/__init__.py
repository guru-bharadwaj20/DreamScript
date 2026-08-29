"""src.ir - the DreamScript intermediate representation (Phase 2).

One schema all five diagram types serialize into: nodes with a shape and a semantic role,
edges with direction and a drawn path, and the ambiguity a hand-drawn diagram leaves behind.
Everything downstream of Phase 2 reads this and never the original annotation format.
"""

from __future__ import annotations

PHASE = "2"
__all__: list[str] = []
