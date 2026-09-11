# Phase 12.2.4 — the frozen prompt template (`dreamscript-codegen/v1`)

Module: `src/codegen/prompt.py`. Pinned by `tests/test_codegen_prompt.py` (SHA-256 of the rendered
example, `3d3f0f31…`). Numbers from `python -m src.codegen.prompt --lengths`, Qwen2.5-Coder-7B-Instruct
tokenizer (local cache, offline, CPU, tokenizer files only), through its own chat template, over all
16,477 pairs in `data/processed/codegen/pairs/{train,validation,test}.jsonl`.

## Usage for 12.2.x / 12.3.x

```python
from src.codegen.pairs import load_pairs
from src.codegen.prompt import build_messages, completion, training_example, extract_code

for record in load_pairs("train"):
    example = training_example(record)   # {template_id, diagram_id, messages, prompt, completion}
code = extract_code(model_reply)         # None on any contract violation
```

The assistant turn is exactly one fenced block (`python`, `sql`, `jsx`, `spice`).

## Token lengths (prompt includes system, user and generation prompt; total adds the answer)

Fixed overhead (system message + chat scaffolding): **247 tokens**, the same for all four languages.

| source | n | prompt p50 / p90 / p99 / max | completion p50 / p90 / p99 / max | total p50 / p90 / p99 / max | total > 4,096 | total > 8,192 |
| :--- | ---: | :--- | :--- | :--- | ---: | ---: |
| all | 16,477 | 517 / 953 / 3,175 / 6,223 | 145 / 632 / 2,901 / 7,343 | 749 / 1,386 / 6,101 / 13,479 | 274 | 98 |
| synthetic | 12,000 | 557 / 895 / 1,253 / 2,412 | 158 / 568 / 1,038 / 2,171 | 802 / 1,279 / 2,005 / 4,583 | 1 | 0 |
| didi | 3,000 | 333 / 396 / 443 / 492 | 38 / 158 / 201 / 229 | 377 / 551 / 643 / 705 | 0 | 0 |
| hdbpmn | 693 | 1,009 / 1,367 / 1,812 / 2,231 | 541 / 902 / 1,187 / 1,411 | 1,519 / 2,260 / 2,978 / 3,642 | 0 | 0 |
| fa_bresler | 300 | 476 / 571 / 636 / 654 | 453 / 559 / 688 / 688 | 932 / 1,130 / 1,324 / 1,342 | 0 | 0 |
| sketch2code | 484 | 2,491 / 5,785 / 6,136 / 6,223 | 2,187 / 4,993 / 5,836 / 7,343 | 4,620 / 10,752 / 12,022 / 13,479 | 273 | 98 |

A 4,096-token training cut keeps 16,203 of 16,477 examples whole; 273 of the 274 it truncates are
sketch2code wireframes. Every validation and test pair (hdbpmn, fa_bresler) fits in 3,642 tokens.

## Contract clauses deleted because the targets contradict them

| inherited clause | measured on the 4,477 real targets |
| :--- | :--- |
| "No bare `pass`" | 146 targets contain one (didi 70, hdbpmn 76), written by 12.1.6 into comment-only branches |
| "every node must appear in the code" | false by design: net nodes, start comments, unnamed containers |
| "represent an unreadable label by its node id" | no emitter does this (fallbacks are `step`, `entity`, `q`) |

## Known target issue surfaced while pinning the example (not fixed here)

12.1.6's structured flowchart emitter picks a loop's body by reachability, not by the edge label:
in the pinned example the `no` branch loops back and the emitted code is `while ok(ctx): retry`,
so the loop condition's polarity ignores `yes`/`no`. This is 12.1.6's emitter, it is shared by
every flowchart target, and structural metrics cannot see it (12.3.4).
