# Phase 12.3.9 - one-shot repair loop (test)

Every LoRA test generation that failed its functional test (48 of 162) gets one repair turn: the original messages, the model's reply, and the failure stated in words (syntax error text, sandbox exception, or the functional test's reason), with the same adapter and decoding.

|  | value |
| :--- | :--- |
| failed before | 48 |
| fixed by one repair | 1 |
| success after repair | 2.1% |
| pass@1 before | 70.4% |
| pass after one repair | 71.0% |
| repaired programs that no longer parse | 2 |
| repair batch / ms per sample | 8 / 26041.7 |

Remaining failure reasons: `{"verdicts_differ": 15, "syntax": 2, "equal": 1, "missing_operation": 15, "order_violation": 7, "never_returns": 1, "crash_or_nontermination": 4, "exclusive_branches_both_ran": 2, "candidate_sandbox": 1}`
