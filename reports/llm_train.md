# Phase 12.2.5 - full QLoRA training run

Base `Qwen/Qwen2.5-Coder-7B-Instruct`; overrides `lora.r=64 lora.alpha=128 data.synthetic_ratio=1.0 train.epochs=1.0 train.eval_every=50`; run directory `experiments/llm/runs/20260912-155733_p12-qwen7b-full-s42`.

| arm | overrides | val loss @0 | best val loss | best step | steps | tokens | wall min | peak alloc GB | peak reserved GB | tokens/s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| full | lora.r=64 lora.alpha=128 data.synthetic_ratio=1.0 train.epochs=1.0 train.eval_every=50 | 0.35647 | 0.01091 | 161 | 161 | 5315661 | 80.8 | 12.828 | 20.722 | 1096.0 |

## Data

```json
{
 "didi": {
  "examples": 3000,
  "tokens": 1227337
 },
 "fa_bresler": {
  "examples": 204,
  "tokens": 193664
 },
 "hdbpmn": {
  "examples": 451,
  "tokens": 708698
 },
 "sketch2code": {
  "examples": 211,
  "tokens": 527248
 },
 "synthetic": {
  "examples": 11999,
  "tokens": 10299403
 },
 "synthetic_used": {
  "examples": 3135,
  "tokens": 2656666
 }
}
```

Dropped for exceeding 4,096 tokens (never truncated): `{'sketch2code': 273, 'synthetic': 1}`
Train/test contamination (exact IR text, ids, scribes): `{'diagram_ids': 0, 'scribes': 0, 'ir_texts': 20}`
Runtime checks: `{"attn_implementation": "sdpa", "gradient_checkpointing_flag": true, "training_mode": true, "gradient_checkpointing_effective": true, "linear4bit_modules": 196, "quant_types": ["nf4"], "trainable_params": 161480704, "stored_params": 4514452992, "weights_footprint_gb": 6.089, "gpu": "NVIDIA RTX 4500 Ada Generation", "gpu_total_gb": 25.76}`

Pack: `{"examples": 7001, "rows": 2574, "tokens": 5313613, "supervised_tokens": 1529177, "fill": 0.9744, "tokens_per_row": 2064.3, "own_row": 246, "truncated": 0}`

![loss](figures/p12_train_loss.png)
