# Phase 12.2.1 - zero-shot base-model benchmark (validation, 176 real pairs)

Identical messages (12.2.4 template as committed at `a7d945b`, loaded from the git object), each rendered through the candidate's own chat template; greedy, 1,600 new tokens max, 4-bit NF4, SDPA. Scored by `src.llm.score` over all 176 generations.

| candidate | contract | syntax | executes | functional pass@1 | dropped nodes | invented/program | hit limit | batch | generate s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 91.5% | 96.6% | 58.5% | 13.6% | 31.1% | 1.761 | 2 | 32 | 1364.8 |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | 12.5% | 60.8% | 26.7% | 1.1% | 89.1% | 1.631 | 73 | 8 | 5982.8 |

## qwen7b

- fa_bresler (n=48): syntax 100.0%, executes 93.8%, functional 22.9%
- hdbpmn (n=128): syntax 95.3%, executes 45.3%, functional 10.2%
- `src.eval.codecheck`: syntax 96.6%, executes 0.0%
- functional failure reasons: `{"verdicts_differ": 34, "equal": 24, "no_acceptor": 3, "order_violation": 10, "exclusive_branches_both_ran": 3, "missing_operation": 74, "crash_or_nontermination": 9, "syntax": 6, "candidate_ValueError": 1, "candidate_AttributeError": 4, "never_returns": 1, "candidate_sandbox": 2, "candidate_ModuleNotFoundError": 2, "candidate_RecursionError": 2, "candidate_TypeError": 1}`
- flags read back: `{"config_use_cache": true, "generation_config_use_cache": true, "attn_implementation": "sdpa", "padding_side": "left", "pad_token_set": true, "eos_token_id": [151645, 151643], "torch_dtype": "torch.bfloat16", "bnb_compute_dtype": "torch.bfloat16", "training_mode": false}`

Batch sweep (32-prompt length-stratified sample):

| batch | ms/sample | gen tokens/s | peak alloc GB | peak reserved GB |
| :--- | :--- | :--- | :--- | :--- |
| 8 | 7355.3 | 38.4 | 7.855 | 8.617 |
| 16 | 17862.8 | 20.5 | 10.094 | 10.756 |
| 32 | 3929.3 | 68.4 | 14.577 | 17.509 |

## dscoder67b

- fa_bresler (n=48): syntax 75.0%, executes 64.6%, functional 4.2%
- hdbpmn (n=128): syntax 55.5%, executes 12.5%, functional 0.0%
- `src.eval.codecheck`: syntax 60.8%, executes 0.0%
- functional failure reasons: `{"verdicts_differ": 29, "equal": 2, "syntax": 69, "no_acceptor": 4, "candidate_Budget": 1, "missing_operation": 17, "candidate_ModuleNotFoundError": 50, "candidate_AttributeError": 3, "candidate_TypeError": 1}`
- flags read back: `{"config_use_cache": true, "generation_config_use_cache": true, "attn_implementation": "sdpa", "padding_side": "left", "pad_token_set": true, "eos_token_id": 32021, "torch_dtype": "torch.bfloat16", "bnb_compute_dtype": "torch.bfloat16", "training_mode": false}`

Batch sweep (32-prompt length-stratified sample):

| batch | ms/sample | gen tokens/s | peak alloc GB | peak reserved GB |
| :--- | :--- | :--- | :--- | :--- |
| 8 | 26017.6 | 40.3 | 17.151 | 21.475 |
| 16 | OOM |  |  |  |

**Selected: `Qwen/Qwen2.5-Coder-7B-Instruct`** (highest functional pass@1, then executability, then syntax; validation only).
