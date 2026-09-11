"""src.llm - Phase 12.2 / 12.3: QLoRA fine-tuning, inference and model-output evaluation.

The data side of Phase 12 (pair schema, synthetic pairs, frozen prompt template, quality
filter) lives in `src.codegen`; this package only consumes it. Weights, adapters and exports
are written under `experiments/llm/` and never committed.
"""

from __future__ import annotations

PHASE = "12"
__all__: list[str] = []
