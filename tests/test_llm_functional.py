"""Phase 12.3.3 - the per-diagram functional tests: IR semantics, the driver, and their teeth."""

from __future__ import annotations

from src.llm import functional as F

FLOW = {
    "diagram_type": "flowchart",
    "nodes": [
        {"id": "s", "text": "received", "semantic_role": "start"},
        {"id": "a", "text": "evaluate application", "semantic_role": "process"},
        {"id": "d", "text": "", "semantic_role": "decision"},
        {"id": "r", "text": "reject application", "semantic_role": "process"},
        {"id": "o", "text": "open account", "semantic_role": "process"},
        {"id": "f", "text": "", "semantic_role": "fork"},
        {"id": "c", "text": "send card", "semantic_role": "process"},
        {"id": "p", "text": "send pin", "semantic_role": "process"},
        {"id": "e", "text": "done", "semantic_role": "end"},
    ],
    "edges": [
        {"src": "s", "dst": "a"},
        {"src": "a", "dst": "d"},
        {"src": "d", "dst": "r", "label": "no"},
        {"src": "d", "dst": "o", "label": "yes"},
        {"src": "r", "dst": "e"},
        {"src": "o", "dst": "f"},
        {"src": "f", "dst": "c"},
        {"src": "f", "dst": "p"},
        {"src": "c", "dst": "e"},
        {"src": "p", "dst": "e"},
    ],
}

GOOD = """
def run_diagram(ctx):
    ctx = evaluate_application(ctx)
    if approved(ctx):
        ctx = open_account(ctx)
        ctx = send_pin(ctx)
        ctx = send_card(ctx)
    else:
        ctx = reject_application(ctx)
    return ctx
"""


def _verdict(code: str) -> tuple[bool, str]:
    return F.compare(F.signature(code, FLOW), F.expected(FLOW, ""))


def test_a_correct_program_passes_whatever_its_polarity_and_fork_order() -> None:
    assert _verdict(GOOD) == (True, "equal")


def test_dropping_a_drawn_operation_fails() -> None:
    assert _verdict(GOOD.replace("        ctx = send_pin(ctx)\n", "")) == (
        False,
        "missing_operation",
    )


def test_running_both_exclusive_branches_fails() -> None:
    both = GOOD.replace("    else:\n        ctx = reject_application(ctx)\n", "") + ""
    both = both.replace("    return ctx", "    ctx = reject_application(ctx)\n    return ctx")
    assert _verdict(both)[0] is False


def test_reversing_the_drawn_order_fails() -> None:
    swapped = GOOD.replace(
        "    ctx = evaluate_application(ctx)\n    if approved(ctx):\n        ctx = open_account(ctx)",
        "    if approved(ctx):\n        ctx = open_account(ctx)\n        ctx = evaluate_application(ctx)",
    )
    assert _verdict(swapped) == (False, "order_violation")


def test_a_crashing_or_non_python_program_fails() -> None:
    assert _verdict("def run_diagram(ctx):\n    raise ValueError('x')\n")[0] is False
    assert _verdict("this is not python")[0] is False


SM = {
    "diagram_type": "state_machine",
    "nodes": [
        {"id": "s0", "text": "q0", "semantic_role": "initial-state"},
        {"id": "s1", "text": "q1", "semantic_role": "state"},
        {"id": "s2", "text": "q2", "semantic_role": "final-state"},
    ],
    "edges": [
        {"src": "s0", "dst": "s0", "label": "a"},
        {"src": "s0", "dst": "s1", "label": "b"},
        {"src": "s1", "dst": "s2", "label": "a,b"},
    ],
}


def test_comma_labels_are_two_symbols_in_the_automaton() -> None:
    verdicts = F.ir_verdicts(SM, [["b", "a"], ["b", "b"], ["a", "b", "a"], ["b"], []])
    assert list(verdicts.values()) == [True, True, True, False, False]


def test_a_stepper_class_is_recognised_and_scored_against_the_drawing() -> None:
    right = """
class M:
    def __init__(self):
        self.state = "q0"
    def transition(self, sym):
        table = {"q0": {"a": "q0", "b": "q1"}, "q1": {"a": "q2", "b": "q2"}}
        self.state = table[self.state][sym]
    def is_final(self):
        return self.state == "q2"
"""
    expected = F.expected(SM, "")
    assert F.compare(F.signature(right, SM), expected) == (True, "equal")
    # 12.1.6's collapsed trigger ("a_b") rejects "b a", which the drawing accepts.
    wrong = right.replace('"q1": {"a": "q2", "b": "q2"}', '"q1": {"a_b": "q2"}')
    assert F.compare(F.signature(wrong, SM), expected) == (False, "verdicts_differ")


def test_mutants_include_the_expected_kinds() -> None:
    kinds = set(F.mutants(GOOD))
    assert {"drop_operation", "swap_operations", "swap_branch_polarity"} <= kinds


def test_a_signature_past_the_sandboxs_default_keep_cap_still_parses() -> None:
    """The depth-MAX_BITS tree of a wide flowchart serialises past 1 MiB.

    The whole `quality` stage used to die on it with a JSONDecodeError at char 1048564, because
    the sandbox keeps only its first MiB of stdout by default and the parent parsed the fragment.
    """
    n = 26
    nodes = [
        {"id": f"node_{i:02d}_long_identifier_text", "text": f"step number {i}",
         "semantic_role": "process"}
        for i in range(n)
    ]
    nodes[0]["semantic_role"] = "start"
    diagram = {
        "diagram_type": "flowchart",
        "nodes": nodes,
        "edges": [
            {"src": nodes[i]["id"], "dst": nodes[i + 1]["id"]} for i in range(n - 1)
        ],
    }
    body = []
    for i in range(12):
        body += [
            f"    if ctx.flag_{i}:",
            f"        node_{i:02d}_long_identifier_text()",
            "    else:",
            f"        node_{(i + 13) % n:02d}_long_identifier_text()",
        ]
    code = (
        "".join(f"def node_{i:02d}_long_identifier_text():\n    pass\n" for i in range(n))
        + "\ndef main(ctx):\n"
        + "\n".join(body)
        + "\n    return 1\n"
    )

    sig = F.signature(code, diagram, timeout_s=180.0)
    assert sig["ok"] is True, sig
    assert len(sig["paths"]) == 2 ** F.MAX_BITS
