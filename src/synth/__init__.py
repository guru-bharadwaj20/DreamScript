"""src.synth - code generation (Phase 12).

Prompt construction from the IR, QLoRA fine-tuning of the 7B base model, inference, and
the repair loop for generations that fail to run.
"""

from __future__ import annotations

PHASE = "12"
__all__: list[str] = []
