"""Phase 12.2.4 - the frozen prompt template: system + diagram type + IR + traversal + contract.

    python -m src.codegen.prompt --show          # render the pinned example

## Frozen means three specific things, all of them checkable

1. **Versioned.** `TEMPLATE_ID` is `dreamscript-codegen/v1` and it names the IR format version
   too, which is why `src.codegen.serialise` does not spend 8 tokens per example on its own
   header (12.1.2 measured that: 3,200 tokens over 400 diagrams, ~80K over 12.1.4's 10K pairs).
   Any change to the wording is a new id, because 12.2.7's three-way comparison is only a
   comparison if zero-shot, few-shot and LoRA saw the same words.
2. **Pinned by a test.** `tests/test_codegen_prompt.py` holds the full rendered output for one
   fixed record, byte for byte. Editing the template fails that test loudly, which is the
   point - a prompt that drifts silently between 12.2.5's training run and 12.2.9's serving is
   the standard way a fine-tune quietly stops working.
3. **The output contract is unambiguous**, and it is unambiguous about the failure cases rather
   than the happy path - see below.

## The output contract, and why it is worded the way it is

    Reply with exactly one fenced code block, ```{language}, and nothing else.
    No prose before or after it. No second block. No `TODO`, no `pass`, no
    placeholder body: every node in the IR must appear in the code. If the
    diagram is incomplete, still emit a complete program - guess nothing that
    is not in the IR, and represent an unreadable label by its node id.

Every clause is a failure this rule exists to make gradable rather than a style preference.
**"Exactly one fenced block, nothing else"** is what 12.3's parser depends on: an extractor that
tolerates prose has to guess which block is the answer, and 12.1.7 requires 100% of targets to
compile, so the training targets are already bare code - a template that permitted commentary
would train the model to emit something the targets never contain. **"No TODO / no pass"**
exists because a stub compiles: S6 (executability >= 85%) is trivially gamed by `def main():
pass`, so the contract forbids the shape S6 cannot see and 12.3's functional tests can.
**"Represent an unreadable label by its node id"** is the honest handling of 9.3.4's real
numbers - handwriting recognition lands at CER 0.68 and exact-match 0.1185, so a large fraction
of `text` fields are wrong or empty, and the alternative to naming the node `n7` is the model
inventing an identifier that no evaluation can match.

## What was measured, Qwen2.5-Coder-7B-Instruct tokenizer, offline, tokenizer files only

    fixed overhead (system message + user scaffolding, identical on every example)
        SYSTEM_TOKENS tokens system, SCAFFOLD_TOKENS tokens of user scaffolding =
        FIXED_TOKENS tokens per example, of which the format legend is LEGEND_TOKENS.

    against 12.1.2's measured IR sizes (p50 253, p90 2,542 tokens) the fixed cost is
    OVERHEAD_P50 of a median prompt and OVERHEAD_P90 of a p90 one.

    full prompt over the same 400 diagrams: p50 P50 / p90 P90 / p99 P99 tokens.
    P99_FITS

The legend is carried on **every** example rather than being left to the fine-tune to absorb,
which is a deliberate cost: 12.2.7 compares LoRA against zero-shot and few-shot on the *same*
template, and a zero-shot model that has never seen `N|id|shape|text|role` cannot read the input
at all, so removing the legend would make that comparison meaningless in exactly the direction
that flatters fine-tuning.

## Rejected

    a bare completion string     REJECTED. Qwen2.5-Coder-*-Instruct is chat-tuned; training on
    with no chat roles           raw text and serving through the chat template is the
                                 train/serve mismatch 12.2.9 would then have to debug.
                                 `build_messages` is primary; `build` renders the same messages
                                 to plain text for a base model, and both are pinned.
    few-shot exemplars in the    REJECTED for the frozen template. An exemplar is 253-2,542
    template                     tokens of the same budget on every one of 12.1.4's 10K pairs,
                                 and 12.2.7 owns the few-shot arm; baking it into the template
                                 would make the three-way comparison a comparison of two.
    restating the traversal      REJECTED. `serialise` already emits it as the `O` line; a
    outside `ir_text`            second copy in the prompt is a second thing to disagree. The
                                 record still stores `traversal` as a list (see
                                 `src.codegen.schema`), but the prompt reads the `O` line.
    naming the shapes vocabulary REJECTED. `src.ir.vocab.SHAPES` is 30+ entries and would double
    in the legend                the fixed overhead to describe values the model can read
                                 literally in the rows it is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

#: Bump on any wording change. `v1` is the IR text format of `src.codegen.serialise` plus the
#: wording below; 12.2.5's adapter and 12.2.7's comparison are both valid only within one id.
TEMPLATE_ID = "dreamscript-codegen/v1"

#: The compact-IR legend. Stated once per prompt, never per example row (12.1.2).
LEGEND = """\
The IR is line-oriented and pipe-delimited. Trailing empty fields are omitted.
  T|<diagram_type>
  N|<id>|<shape>|<text>|<role>      a node; <role> omitted when unknown
  E|<src>|<dst>|<label>|<dir>       an edge; <dir> omitted when directed, "-" when not
  O|<id> <id> ...                   reading order; this is the order to emit statements in
  R|<src>><dst> ...                 edges that return to an earlier node - these are loops
In text fields, \\p is a literal "|", \\n a line break and \\\\ a backslash."""

SYSTEM = """\
You convert the intermediate representation of a hand-drawn diagram into working code.

{legend}

The diagram was read from handwriting, so labels may be misspelled, truncated or empty. Use \
what is there; never invent structure the IR does not state.

Output contract:
Reply with exactly one fenced code block, ```{{language}}, and nothing else. No prose before or \
after it. No second block. No TODO, no bare pass, no placeholder body: every node in the IR \
must appear in the code. If the diagram is incomplete, still emit a complete program - guess \
nothing that is not in the IR, and represent an unreadable label by its node id."""

USER = """\
diagram_type: {diagram_type}
target_language: {language}

IR:
{ir_text}

Emit the {language} program."""


def system_message(language: str = "{language}") -> str:
    """The system text. `language` is substituted so the fence tag is stated concretely."""
    return SYSTEM.format(legend=LEGEND).replace("{language}", language)


def build_messages(record: Any) -> list[dict[str, str]]:
    """Chat messages for one pair. Primary renderer - Qwen2.5-Coder-Instruct is chat-tuned."""
    data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    return [
        {"role": "system", "content": system_message(str(data["language"]))},
        {
            "role": "user",
            "content": USER.format(
                diagram_type=data["diagram_type"],
                language=data["language"],
                ir_text=data["ir_text"],
            ),
        },
    ]


def build(record: Any) -> str:
    """The same prompt as flat text, for a base (non-chat) model. Roles become `<role>:` heads."""
    return "\n\n".join(f"{m['role']}:\n{m['content']}" for m in build_messages(record)).strip()


def completion(record: Any) -> str:
    """The target side: exactly what the contract demands the model produce, and nothing more."""
    data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    return f"```{data['language']}\n{str(data['target_code']).strip()}\n```"


def training_example(record: Any) -> dict[str, Any]:
    """One supervised example: `{template_id, messages, prompt, completion}`."""
    return {
        "template_id": TEMPLATE_ID,
        "messages": build_messages(record) + [{"role": "assistant", "content": completion(record)}],
        "prompt": build(record),
        "completion": completion(record),
    }


def extract_code(reply: str) -> str | None:
    """Pull the single fenced block back out. `None` when the reply breaks the contract.

    Strict on purpose: a reply with prose, no fence, or a second block is a contract violation
    and 12.3 must be able to count those, not paper over them.
    """
    parts = reply.split("```")
    if len(parts) != 3 or parts[0].strip() or parts[2].strip():
        return None
    body = parts[1]
    return body.split("\n", 1)[1] if "\n" in body else ""


EXAMPLE_RECORD: dict[str, Any] = {
    "diagram_id": "example",
    "diagram_type": "flowchart",
    "ir_text": "T|flowchart\nN|n0|oval|start|start\nN|n1|diamond|ok?|decision\n"
    "N|n2|rectangle|retry|process\nE|n0|n1\nE|n1|n2|no\nE|n2|n1\nO|n0 n1 n2\nR|n2>n1",
    "traversal": ["n0", "n1", "n2"],
    "target_code": "def main():\n    while not ok():\n        retry()\n",
    "language": "python",
    "source": "handwritten",
    "split": "train",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.2.4 - the frozen prompt template")
    parser.add_argument("--show", action="store_true", help="render EXAMPLE_RECORD as text")
    args = parser.parse_args(argv)
    if args.show:
        sys.stdout.write(build(EXAMPLE_RECORD) + "\n")
    else:
        json.dump(training_example(EXAMPLE_RECORD), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
