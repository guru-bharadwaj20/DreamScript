"""Phase 13.3 / 13.4 - producing the code, and what to fall back to when the model cannot.

The pipeline's last stage is the one with a choice in it. There are two ways to turn an IR into
code in this repo and they are not equivalent:

    model      12.2's fine-tuned adapter, served over llama.cpp. What the project is *for*, and
               what 12.3 measures: 99.4% of its programs parse and run, 20.4% pass their tests.
    emitter    12.1.6's deterministic transcription. Not a model at all - it is the function the
               model was trained to imitate, and it passes those same tests at 78.4%.

The emitter is therefore **not a degraded fallback in quality**, and saying so plainly matters:
it scores higher than the model on functional correctness. What it cannot do is generalise - it
reads the IR by rule, so anything the IR does not carry, it cannot invent, while the model can
write plausible code for a diagram the emitter would refuse. 13.4's chain is `model -> emitter`
because the model is the claim being tested; a result produced by the emitter is marked
`degraded=True` so a caller can tell which one answered, and no measurement mixes them.

A served model is optional. With none reachable, the pipeline still returns code and says how.
"""

from __future__ import annotations

from typing import Any

from src.pipeline.contracts import Outcome


def language_for(kind: str) -> str:
    """The language this diagram type is emitted in - 12.1.6's assignment, asked for rather than
    restated.

    **There were two of these tables and they disagreed in both directions.** This module carried
    its own, keyed `er_diagram` with no aliases, behind a `.get(kind, "python")` default;
    `codegen.targets` keys its emitters on `"er"` with aliases for `bpmn`, `erd`, `er_diagram`,
    `state-machine` and `ui`, and raises `KeyError` on anything else because "a diagram type
    nobody wrote a generator for is a gap to report, not a silent mis-emission into Python".

    So `emit` took the language from one table and the emitter from the other: `"er"` selected
    `emit_er_sql` and was labelled `python`, and `"bpmn"` selected the flowchart emitter and was
    labelled `python` by accident rather than by the table. Raising together is the point - the
    gap `targets` refuses to paper over is not one this module should paper over on its behalf.
    """
    from src.codegen.targets import language_for as _language_for

    return _language_for(kind)


def emit(
    diagram: dict, order: list[str], kind: str, language: str, *, model_url: str | None = None
) -> Outcome[tuple[str, str]]:
    """Code for this diagram: the served model if there is one, else 12.1.6's emitter.

    **`degraded` means a rung below the first choice answered, and it was true on every run.**
    The emitter's result came back through `Outcome.fallback` unconditionally, so
    `summarise()["degraded"]` and the API's `degraded` field were 100% always - including on a
    deployment with no model configured, where the emitter is not a fallback but the only rung
    there is. A flag that is always set carries no information, and this one is the flag the
    module argues at length is what lets a caller trust the answer.

    So the three cases are distinguished, and the reason says which happened:

        no model configured   not degraded. The emitter answered because it is the chain.
        model asked, answered not degraded. The first rung answered.
        model asked, failed   degraded, carrying the model's own failure reason - this is what
                              the flag was for.
    """
    fell_back = ""
    if model_url:
        served = _from_model(diagram, order, kind, model_url)
        if served.ok:
            return served
        fell_back = served.reason or "the served model did not answer"

    try:
        from src.codegen.targets import for_type

        code = for_type(kind)(diagram, order)
    except KeyError:
        return Outcome.failed(f"no target generator for {kind!r}")
    except Exception as error:  # noqa: BLE001 - a malformed IR is a soft failure here
        return Outcome.failed(f"emitter failed: {type(error).__name__}: {error}")

    if fell_back:
        return Outcome.fallback(
            (code, language),
            f"{fell_back}; 12.1.6's emitter answered instead",
            confidence=1.0,
        )
    return Outcome(
        value=(code, language),
        reason="no model configured; 12.1.6's emitter is the whole chain",
        confidence=1.0,
    )


def _from_model(diagram: dict, order: list[str], kind: str, url: str) -> Outcome[Any]:
    """Ask a served adapter, through 12.2.4's frozen template."""
    import json
    import urllib.error
    import urllib.request

    from src.codegen import prompt as template
    from src.codegen.serialise import serialise

    pair = {
        "diagram_type": kind,
        "ir_text": serialise(diagram, order),
        "traversal": str(order),
        "language": language_for(kind),
    }
    try:
        messages = template.build_messages(pair)
        body = json.dumps({"messages": messages, "temperature": 0.0, "stream": False})
        request = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
        return Outcome.failed(f"served model unreachable: {type(error).__name__}: {error}")

    from src.llm.score import code_of

    code = code_of(text)
    if not code.strip():
        return Outcome.failed("the served model returned no code block")
    return Outcome(value=(code, language_for(kind)))
