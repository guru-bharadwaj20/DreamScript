"""Phase 12.2.7 - the few-shot arm: k solved train pairs as prior chat turns, chosen leak-free.

Exemplars are real train pairs of the **same diagram type and source** as the query (so a
BPMN page is shown BPMN pages and an automaton is shown automata), chosen deterministically as
the `k` whose IR length is nearest the query's under a per-exemplar size cap. Two exclusions are
enforced, not hoped for:

* no exemplar's `ir_text` equals any held-out (validation or test) `ir_text`. fa_bresler is 12
  exercises drawn by 25 writers, so a scribe-disjoint split still leaves identical IR across
  splits (measured: 20 test and 22 validation IR texts also occur in train); showing one of those
  as an exemplar would hand the model the answer.
* no exemplar shares the query's scribe or diagram id (guaranteed by 12.1.8, re-asserted here).

The frozen template's system and user wording is reused verbatim; exemplars are only extra
`user`/`assistant` turns, so the few-shot arm differs from zero-shot by the exemplars alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class FewShot:
    def __init__(
        self,
        train: Sequence[dict],
        held_out: Sequence[dict],
        template: Any,
        k: int = 2,
        max_exemplar_chars: int = 6000,
    ) -> None:
        banned = {p["ir_text"] for p in held_out}
        self.template = template
        self.k = k
        self.pool: dict[tuple[str, str], list[dict]] = {}
        self.excluded_identical_ir = 0
        for p in train:
            if p["ir_text"] in banned:
                self.excluded_identical_ir += 1
                continue
            size = len(p["ir_text"]) + len(p["target_code"])
            if size > max_exemplar_chars:
                continue
            self.pool.setdefault((p["diagram_type"], p["source"]), []).append(p)
        for items in self.pool.values():
            items.sort(key=lambda p: (len(p["ir_text"]), p["diagram_id"]))

    def exemplars(self, query: dict) -> list[dict]:
        items = self.pool.get((query["diagram_type"], query["source"]), [])
        items = [
            p
            for p in items
            if p["diagram_id"] != query["diagram_id"]
            and (not query.get("scribe") or p.get("scribe") != query.get("scribe"))
        ]
        target = len(query["ir_text"])
        ranked = sorted(items, key=lambda p: (abs(len(p["ir_text"]) - target), p["diagram_id"]))
        return ranked[: self.k]

    def messages(self, query: dict) -> list[dict[str, str]]:
        zero = self.template.build_messages(query)
        system, user = zero[0], zero[1:]
        turns: list[dict[str, str]] = [system]
        for ex in self.exemplars(query):
            turns += self.template.build_messages(ex)[1:]
            turns.append({"role": "assistant", "content": self.template.completion(ex)})
        return turns + user

    def builder(self) -> Callable[[dict], list[dict[str, str]]]:
        return self.messages
