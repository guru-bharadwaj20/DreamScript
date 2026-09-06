"""src.parse - structure recovery (Phases 7.3, 7.4).

HMM role decoding over node sequences, the GMM shape vocabulary, and the posterior-gated
ambiguity repair of 7.3.10.

Graph assembly, broken-arrow repair, type-specific validation and IR serialization were
planned to live here and were built in `src.assemble` instead, where Phase 10 keeps its own
corpus, coordinate frame and detection cache.
"""

from __future__ import annotations

PHASE = "7"
__all__: list[str] = []
