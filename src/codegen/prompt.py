"""Phase 12.2.4 - the frozen prompt template: system + diagram type + IR + traversal + contract.

    python -m src.codegen.prompt --show                  # render the pinned example
    python -m src.codegen.prompt --lengths               # token lengths over the pair files

    from src.codegen.prompt import build_messages, completion, training_example, extract_code
    messages = build_messages(record)          # [system, user] for a chat model
    target = completion(record)                # the assistant turn: one fenced block

## Frozen means three checkable things

1. **Versioned.** `TEMPLATE_ID = "dreamscript-codegen/v1"` names the wording *and* 12.1.2's IR
   format, which is why `serialise` does not spend a header on every example. Any wording change
   is a new id: 12.2.7's zero-shot / few-shot / LoRA comparison is a comparison only if all
   three arms saw the same words.
2. **Pinned byte for byte.** `tests/test_codegen_prompt.py` holds the SHA-256 of the rendered
   `EXAMPLE_RECORD` messages and completion (`PINNED_SHA256`). Editing a character fails it.
3. **The output contract states only what the training targets actually satisfy.** A contract
   the targets break teaches the model that the contract is decoration.

## The output contract, and what was deleted from the inherited draft

    Reply with exactly one fenced code block tagged {fence}, and nothing else: no prose
    before or after it and no second block. Emit a complete program for the whole diagram,
    in the traversal order given by the O line. Use only names, labels and connections
    that appear in the IR; do not invent nodes, edges, columns, parts or values.

**"Exactly one fenced block"** is what `extract_code` - and so 12.3.1-12.3.9 - depends on, and
`completion()` produces exactly that shape, so every training target obeys it by construction.
**"Use only what the IR states"** is the rule 12.1.1 and 12.1.4 were rebuilt to make true: the
schema refuses a target naming the diagram id, and synthetic ER columns and circuit nets were
moved out of `attrs` into the IR text so they are visible.

The inherited draft's contract had three clauses the targets contradict, each measured on the
4,477 real pairs of 12.1.1, and all three are deleted rather than kept as aspirations:

    "No bare pass"                   146 targets contain a bare `pass` (70 didi, 76 hdbpmn):
                                     12.1.6's emitter writes one into a branch whose only node
                                     is a start/event comment, because Python needs a statement.
    "every node in the IR must       false by design - a `wire` net node becomes a number on a
    appear in the code"              card, a start node a comment, a wireframe container a div
                                     with no name. 12.3.4 measures correspondence properly.
    "represent an unreadable label   no emitter does this: flowchart steps fall back to `step`,
    by its node id"                  ER tables to `entity`. A contract clause no target shows.

The draft also left its token section as unfilled placeholders (`SYSTEM_TOKENS`, `P99_FITS`); the
measured numbers are in `reports/codegen_prompt.md` and the 12.2.4 row.

## Rejected

    a bare completion string     Qwen2.5-Coder-*-Instruct is chat-tuned; training on raw text and
                                 serving through the chat template is a train/serve mismatch.
                                 `build_messages` is primary, `build` flattens for a base model.
    few-shot exemplars inside    12.2.7 owns the few-shot arm; baking exemplars in would make the
    the template                 three-way comparison a comparison of two, and cost every example.
    restating the traversal      `ir_text`'s `O` line already carries it; a second copy is a
    outside `ir_text`            second thing that can disagree.
    dropping the legend          a zero-shot model that has never seen `N|id|shape|text|role`
                                 cannot read the input, so removing it biases 12.2.7 toward LoRA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

TEMPLATE_ID = "dreamscript-codegen/v1"

#: The fence tag per `language`. `react` is the pair field; the fence says `jsx`, the tag every
#: code model has seen for JSX, and `extract_code` accepts any tag.
FENCE: dict[str, str] = {"python": "python", "sql": "sql", "react": "jsx", "spice": "spice"}

LEGEND = """\
The IR is line-oriented and pipe-delimited. Trailing empty fields are omitted.
  T|<diagram_type>
  N|<id>|<shape>|<text>|<role>      a node; <role> omitted when unknown
  E|<src>|<dst>|<label>|<dir>       an edge; <dir> omitted when directed, "-" when not
  O|<id> <id> ...                   traversal order: emit the program in this order
  R|<src>><dst> ...                 edges that return to an earlier node in that order
In text fields, \\p is a literal "|", \\n a line break and \\\\ a backslash."""

SYSTEM = """\
You convert the intermediate representation (IR) of a hand-drawn diagram into working code.

{legend}

Output contract:
Reply with exactly one fenced code block tagged {fence}, and nothing else: no prose before or \
after it and no second block. Emit a complete program for the whole diagram, in the traversal \
order given by the O line. Use only names, labels and connections that appear in the IR; do not \
invent nodes, edges, columns, parts or values."""

USER = """\
diagram_type: {diagram_type}
target_language: {language}

IR:
{ir_text}"""

#: SHA-256 of `_pin_text()` for EXAMPLE_RECORD. Changing any wording must change TEMPLATE_ID too.
PINNED_SHA256 = "3d3f0f31067f3ad2863e8af2f54c885fbd283c14a86b4a9dd39c9bbb1439e81a"


def fence_for(language: str) -> str:
    return FENCE.get(str(language), str(language))


def _data(record: Any) -> dict:
    return record.to_dict() if hasattr(record, "to_dict") else dict(record)


def system_message(language: str) -> str:
    return SYSTEM.format(legend=LEGEND, fence=fence_for(language))


def build_messages(record: Any) -> list[dict[str, str]]:
    """`[system, user]` chat messages for one pair (12.1.1 record or any dict with its keys)."""
    data = _data(record)
    user = USER.format(
        diagram_type=data["diagram_type"], language=data["language"], ir_text=data["ir_text"]
    )
    return [
        {"role": "system", "content": system_message(data["language"])},
        {"role": "user", "content": user},
    ]


def build(record: Any) -> str:
    """The same prompt as flat text for a base (non-chat) model."""
    return "\n\n".join(f"{m['role']}:\n{m['content']}" for m in build_messages(record))


def completion(record: Any) -> str:
    """The assistant turn the contract demands: one fenced block holding the target."""
    data = _data(record)
    return f"```{fence_for(data['language'])}\n{str(data['target_code']).strip()}\n```"


def training_example(record: Any) -> dict[str, Any]:
    """`{template_id, diagram_id, messages, prompt, completion}` - one supervised example."""
    data = _data(record)
    return {
        "template_id": TEMPLATE_ID,
        "diagram_id": data.get("diagram_id"),
        "messages": build_messages(data) + [{"role": "assistant", "content": completion(data)}],
        "prompt": build(data),
        "completion": completion(data),
    }


def extract_code(reply: str) -> str | None:
    """The single fenced block's body, or `None` when the reply breaks the contract.

    Strict on purpose: prose around the block, no fence, or a second block are contract
    violations that 12.3 must be able to count, not paper over. Surrounding whitespace is allowed.
    """
    parts = reply.strip().split("```")
    if len(parts) != 3 or parts[0].strip() or parts[2].strip():
        return None
    body = parts[1]
    if "\n" not in body:
        return None
    return body.split("\n", 1)[1]


EXAMPLE_RECORD: dict[str, Any] = {
    "diagram_id": "example",
    "diagram_type": "flowchart",
    "ir_text": "T|flowchart\nN|n0|ellipse|start|start\nN|n1|diamond|ok?|decision\n"
    "N|n2|rectangle|retry|process\nN|n3|ellipse|end|end\nE|n0|n1\nE|n1|n2|no\nE|n1|n3|yes\n"
    "E|n2|n1\nO|n0 n1 n2 n3\nR|n2>n1",
    "traversal": ["n0", "n1", "n2", "n3"],
    "target_code": "def run_diagram(ctx):\n    # start: start\n    while ok(ctx):\n"
    "        ctx = retry(ctx)\n    return ctx\n",
    "language": "python",
    "source": "handwritten",
    "split": "train",
    "split_basis": "example",
    "scribe": None,
    "meta": {},
}


def _pin_text() -> str:
    return json.dumps(training_example(EXAMPLE_RECORD), sort_keys=True, ensure_ascii=False)


def pin_digest() -> str:
    return hashlib.sha256(_pin_text().encode("utf-8")).hexdigest()


def token_lengths(records, tokenizer=None) -> dict:
    """Chat-template token counts (prompt, completion, total) per source, Qwen2.5-Coder tokenizer."""
    if tokenizer is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    rows: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    system_tokens: Counter = Counter()
    for record in records:
        messages = build_messages(record)
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion(record)}]
        )
        for key in (record["source"], "all"):
            rows[key]["prompt"].append(len(prompt_ids))
            rows[key]["completion"].append(len(full_ids) - len(prompt_ids))
            rows[key]["total"].append(len(full_ids))
        system_tokens[record["language"]] = len(
            tokenizer.apply_chat_template(messages[:1] + [{"role": "user", "content": ""}])
        )

    def stats(values: list[int]) -> dict:
        v = sorted(values)
        return {
            "n": len(v),
            "p50": v[len(v) // 2],
            "p90": v[int(len(v) * 0.9)],
            "p99": v[min(len(v) - 1, int(len(v) * 0.99))],
            "max": v[-1],
            "over_4096": sum(x > 4096 for x in v),
            "over_8192": sum(x > 8192 for x in v),
        }

    return {
        "fixed_overhead_by_language": dict(system_tokens),
        "by_source": {s: {k: stats(v) for k, v in d.items()} for s, d in sorted(rows.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.2.4 - the frozen prompt template")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--lengths", action="store_true")
    args = parser.parse_args(argv)
    if args.lengths:
        from src.codegen.pairs import load_pairs

        json.dump(token_lengths(list(load_pairs())), sys.stdout, indent=2)
    elif args.show:
        sys.stdout.write(build(EXAMPLE_RECORD) + "\n\nassistant:\n" + completion(EXAMPLE_RECORD))
    else:
        json.dump(training_example(EXAMPLE_RECORD), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
