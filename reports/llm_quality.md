# Phase 12.3.1 / 12.3.2 / 12.3.3 / 12.3.8 - quality of the LoRA model's test outputs

162 test generations from 12.2.7's LoRA arm; the reference column scores 12.1.6's reference programs through the identical scorer.

| metric | LoRA | reference programs |
| :--- | :--- | :--- |
| contract | 100.0% | 100.0% |
| syntax | 99.4% | 100.0% |
| executes | 99.4% | 100.0% |
| functional | 20.4% | 78.4% |
| dropped_node_rate | 16.8% | 16.8% |
| programs_with_dropped | 25.3% | 26.5% |
| invented_per_program | 0.858 | 0.852 |
| programs_with_invented | 30.9% | 31.5% |

`src.eval.codecheck` (data agent's helpers): syntax 99.4% `{'syntax.ok': 161, 'syntax.python': 1}`, executes 99.4% `{'exec.ok': 161, 'exec.no_entry': 1}`

Functional failure reasons: `{"verdicts_differ": 39, "equal": 33, "missing_operation": 88, "crash_or_nontermination": 1, "syntax": 1}`

Sandbox kinds: `{"ok": 161, "syntax_error": 1}`

## Failures (first 8)

- `writer015_fa_001` (fa_bresler): verdicts_differ; syntax True, executes True, dropped [], invented []
- `writer015_fa_002` (fa_bresler): verdicts_differ; syntax True, executes True, dropped [], invented []
- `writer015_fa_004` (fa_bresler): verdicts_differ; syntax True, executes True, dropped [], invented []
- `writer015_fa_005` (fa_bresler): verdicts_differ; syntax True, executes True, dropped ['A', 'B', 'C', 'D', 'F1'], invented []
- `writer015_fa_006` (fa_bresler): verdicts_differ; syntax True, executes True, dropped ['S', '{f}', '{q1}', '{q2,f}', '{q2}'], invented ['q1', 'q2']
- `writer015_fa_007` (fa_bresler): verdicts_differ; syntax True, executes True, dropped [], invented []
- `writer015_fa_008` (fa_bresler): verdicts_differ; syntax True, executes True, dropped [], invented []
- `writer015_fa_009` (fa_bresler): verdicts_differ; syntax True, executes True, dropped ['E', 'F', 'G', 'Q', 'R'], invented []
