# Phase 12.2.8 - adapter merge and export

| step | result |
| :--- | :--- |
| merge (bf16, CPU) | {"merge_s": 37.2, "merged_gb": 15.23} |
| merged vs unmerged logits | {"prompts": 8, "max_abs_logit_diff": 0.25427, "argmax_agree": 8, "greedy_64_identical": 8} |
| GGUF f16 | {"path": "C:\\Users\\Temp\\Desktop\\DreamScript\\experiments\\llm\\export\\artifacts\\model-f16.gguf", "gb": 15.24} |
| GGUF Q8_0 | {"path": "C:\\Users\\Temp\\Desktop\\DreamScript\\experiments\\llm\\export\\artifacts\\model-Q8_0.gguf", "gb": 8.1, "quantize_s": 28.3} |
| GGUF Q4_K_M | {"path": "C:\\Users\\Temp\\Desktop\\DreamScript\\experiments\\llm\\export\\artifacts\\model-Q4_K_M.gguf", "gb": 4.68, "quantize_s": 47.0} |
| llama-server Q4_K_M smoke | {"diagram_id": "writer014_fa_001", "parses": true, "new_tokens": 334, "seconds": 4.29} |

llama.cpp release `b10909`; artefacts under `experiments/llm/export/artifacts` (gitignored). vLLM does not run on native Windows; the merged safetensors directory is the vLLM-loadable form and is untested here.
