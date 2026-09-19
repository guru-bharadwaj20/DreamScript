"""Phase 13 - the end-to-end pipeline: one photograph in, runnable code out.

Phases 1-12 each produce a stage that is measured on its own split. This package is the first
thing that composes them, which means it is also the first place the project's actual claim -
*a drawn diagram becomes code* - can be measured rather than assumed.

    from src.pipeline import DreamScriptPipeline
    DreamScriptPipeline().run("page.png")      # -> Result(ir=..., code=..., stages=[...])

The composition order is the one the phases already imply:

    load -> classify -> detect -> assemble -> traverse -> serialise -> generate -> verify

Every stage is a `Stage` with a typed input and output (13.2), every stage is cached on the hash
of its input and the configuration that produced it (13.6), every stage fails soft with a reason
rather than an exception (13.8), and every stage records its own wall time so the budget in 13.7
is a table rather than a single number.
"""

from __future__ import annotations

PHASE = "13"

from src.pipeline.contracts import Outcome, Result, StageReport  # noqa: E402
from src.pipeline.core import DreamScriptPipeline  # noqa: E402

__all__ = ["DreamScriptPipeline", "Outcome", "Result", "StageReport"]
