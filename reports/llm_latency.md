# Phase 12.2.9 - inference latency

32 test prompts stratified by IR size, one request at a time, greedy, streaming (time to first token and to the end of the program).

| path | TTFT p50 s | total p50 s | total p90 s | max s | new tokens p50 | syntax | functional |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| HF NF4 + LoRA (batch 1, streaming) | 0.404 | 27.112 | 43.164 | 57.206 | 522.5 | 100.0% | 15.6% |
| llama.cpp Q4_K_M (batch 1, streaming) | 0.155 | 6.678 | 10.872 | 13.825 | 518.5 | 100.0% | 15.6% |
| llama.cpp Q8_0 (batch 1, streaming) | 0.164 | 10.806 | 17.395 | 22.356 | 516.0 | 100.0% | 15.6% |

KV cache ablation (HF NF4 + LoRA, 128 forced tokens): `[{"diagram_id": "writer015_fa_004", "use_cache": true, "new_tokens": 128, "ms_per_token": 51.2}, {"diagram_id": "writer015_fa_004", "use_cache": false, "new_tokens": 128, "ms_per_token": 157.2}, {"diagram_id": "writer015_fa_001", "use_cache": true, "new_tokens": 128, "ms_per_token": 50.1}, {"diagram_id": "writer015_fa_001", "use_cache": false, "new_tokens": 128, "ms_per_token": 162.8}, {"diagram_id": "writer018_fa_003", "use_cache": true, "new_tokens": 128, "ms_per_token": 57.2}, {"diagram_id": "writer018_fa_003", "use_cache": false, "new_tokens": 128, "ms_per_token": 180.3}]`

Batching (llama.cpp Q4_K_M, 4 slots, all requests concurrent): `{"slots": 4, "requests": 32, "wall_s": 74.9, "throughput_req_per_min": 25.64, "per_request_p50_s": 9.04, "per_request_p90_s": 14.411}`

**Serving path: llama.cpp Q4_K_M (batch 1, streaming)** (fastest p90 among paths within 5 points of the HF adapter's functional pass rate). Budget < 8 s at p90: **False**
