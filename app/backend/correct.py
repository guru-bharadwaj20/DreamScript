"""Phase 16.2.8 - a tap that fixes a label, and the code that follows from it.

    from app.backend.correct import apply_correction, regenerate
    diagram, changed = apply_correction(record["result"]["ir"], correction)
    code, language, degraded, reason = regenerate(diagram, traversal, "flowchart")

## Why this is worth a module

The plan calls the correction screen the highest-value one in the app, and the arithmetic behind
that is 14's error taxonomy: **`sub_text` is 11 of hdbpmn's median 21 edits**. More than half of
what separates a read page from a correct one is a word the recogniser got wrong - so one tap on
one label moves the result further than a recogniser retrain would, and it moves it *now*, for the
person holding the phone, rather than in a week for everybody else.

The second half is that the taps are training data. 15.7 built a correction store for exactly this,
and 16.1.1's `feedback.jsonl` is never evicted while the predictions it refers to are.

## Why the regeneration happens here and not on the model server

The emitter is `src.codegen.targets` and it is **not a model** - a fact worth checking rather than
assuming, and checked: importing `src.pipeline.generate` pulls in no torch, no transformers, no
ultralytics, no cv2. So a corrected label can be turned back into code in this process, in
milliseconds, without a second round trip to a GPU box that would have nothing to do.

That keeps the app's promise about itself intact: this backend still holds no model. It holds an
emitter, which is a template.

## What a correction may and may not change

Only `sub_text` rewrites the IR today, and the restraint is deliberate. Retyping a label is a local
edit with an obvious meaning; deleting a node or adding an edge changes the graph's topology, which
changes the traversal, which changes which branch the emitter writes - and a person tapping a wrong
word has not asked for that. Every other kind is **logged and not applied**, which is stated in the
response rather than silently ignored, so the correction is still training data and nobody is told a
change happened that did not.
"""

from __future__ import annotations

import copy
from typing import Any

#: Correction kinds that rewrite the IR. The rest are recorded and reported as unapplied.
APPLIED_KINDS = ("sub_text",)


def apply_correction(ir: Any, correction: dict[str, Any]) -> tuple[Any, bool]:
    """A corrected copy of `ir`, and whether anything changed.

    A copy, never in place: the caller still holds the record it read, and a store write that
    half-succeeds must not leave a mutated object behind claiming otherwise.
    """
    if not isinstance(ir, dict):
        return ir, False
    kind = str(correction.get("kind") or "")
    if kind not in APPLIED_KINDS:
        return ir, False

    node_id = str(correction.get("node") or "")
    now = correction.get("now")
    if not node_id or now is None:
        return ir, False

    updated = copy.deepcopy(ir)
    changed = False
    for node in updated.get("nodes") or []:
        if node.get("id") != node_id:
            continue
        if node.get("text") == now:
            # Already says that. Not an error and not a change - a double tap, or two phones.
            break
        node["text"] = str(now)
        # The label is now a person's, not the recogniser's. Saying so matters downstream: 16.2.9
        # flags low-confidence text, and a corrected label must stop being flagged.
        node["confidence"] = 1.0
        node["corrected"] = True
        changed = True
        break
    return updated, changed


def regenerate(
    diagram: Any, traversal: list[str], diagram_type: str
) -> tuple[str | None, str | None, bool, str]:
    """Re-emit code from a corrected IR.

    Returns `(code, language, degraded, reason)`. `degraded` and `reason` come straight from
    13.4's `Outcome`, because the rule that a caller must be able to tell which rung answered does
    not stop applying when the input was edited by hand.
    """
    from src.pipeline.generate import emit, language_for

    try:
        language = language_for(diagram_type)
    except Exception as error:  # noqa: BLE001 - an unknown type is an answer, not a crash
        return None, None, False, f"no language for {diagram_type!r}: {error}"

    outcome = emit(diagram, list(traversal), diagram_type, language)
    if not outcome.ok:
        return None, language, True, outcome.reason or "the emitter produced nothing"
    code, emitted_language = outcome.value
    return code, emitted_language, bool(outcome.degraded), outcome.reason or ""
