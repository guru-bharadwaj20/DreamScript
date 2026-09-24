"""Phase 12.2.3 / 12.2.5 / 12.2.6 - one QLoRA training run from `configs/llm.yaml`.

    python -m src.llm.run --config configs/llm.yaml lora.r=32 train.lr=1e-4 train.token_budget=1500000

Writes `experiments/llm/runs/<timestamp>_<run_name>/` with `config.yaml`, `env.json`, `run.log`,
`train.jsonl` (per step: loss, lr, grad norm, tokens/s, peak VRAM, wall time; val loss every
`eval_every` steps), `summary.json`, `data.json` (what went into the run and what was dropped)
and `adapter/`. The prompt is 12.2.4's frozen template; its `TEMPLATE_ID` is recorded so an
adapter is never served through a different wording than it was trained on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from typing import Any


def build_examples(cfg, tokenizer) -> tuple[list, list, dict[str, Any]]:
    from src.codegen import prompt as template
    from src.llm import pairs, train

    eos = tokenizer.eos_token
    rng = random.Random(int(cfg.seed))
    sources = list(cfg.data.sources)
    raw = pairs.load("train", sources=sources)
    report: dict[str, Any] = {"template_id": getattr(template, "TEMPLATE_ID", None)}
    kept: dict[str, list] = {}
    dropped: Counter = Counter()
    for pair in raw:
        ex = train.tokenise(
            tokenizer, template.build_messages(pair), template.completion(pair), eos
        )
        if len(ex) > int(cfg.data.max_example_tokens):
            dropped[pair["source"]] += 1
            continue
        kept.setdefault(pair["source"], []).append(ex)
    real = [e for s, v in kept.items() if s != "synthetic" for e in v]
    synthetic = kept.get("synthetic", [])
    if cfg.data.synthetic_ratio is not None and synthetic:
        real_tokens = sum(len(e) for e in real)
        rng.shuffle(synthetic)
        budget, chosen, used = float(cfg.data.synthetic_ratio) * real_tokens, [], 0
        for e in synthetic:
            if used + len(e) > budget:
                break
            chosen.append(e)
            used += len(e)
        synthetic = chosen
    examples = real + synthetic
    report["train"] = {
        s: {"examples": len(v), "tokens": sum(len(e) for e in v)} for s, v in kept.items()
    }
    if "synthetic" in kept:
        report["train"]["synthetic_used"] = {
            "examples": len(synthetic),
            "tokens": sum(len(e) for e in synthetic),
        }
    report["dropped_over_max_tokens"] = dict(dropped)
    val = [
        train.tokenise(tokenizer, template.build_messages(p), template.completion(p), eos)
        for p in pairs.load("validation", sources=list(cfg.data.val_sources))
    ]
    report["validation"] = {"examples": len(val), "tokens": sum(len(e) for e in val)}
    report["contamination_vs_test"] = pairs.contamination(raw, pairs.load("test"))
    return examples, val, report


def run(cfg) -> dict[str, Any]:  # pragma: no cover - GPU
    from peft import get_peft_model

    from src.llm import quant, train
    from src.utils.config import save_config
    from src.utils.logging import start_run

    active = start_run(cfg)
    save_config(cfg, active.dir / "config.yaml")
    quant.cap_memory(float(cfg.model.vram_cap_gib))
    model, tokenizer = quant.load_model(
        cfg.model.id, quantize=bool(cfg.model.quantize), for_training=True
    )
    model = get_peft_model(
        model,
        quant.lora_config(
            r=int(cfg.lora.r),
            alpha=int(cfg.lora.alpha),
            dropout=float(cfg.lora.dropout),
            targets=tuple(cfg.lora.targets),
        ),
    )
    model.train()
    checks = quant.runtime_checks(model)
    assert checks["gradient_checkpointing_effective"], "gradient checkpointing is not applied"
    examples, val, data_report = build_examples(cfg, tokenizer)
    data_report["runtime_checks"] = checks
    (active.dir / "data.json").write_text(
        json.dumps(data_report, indent=2) + "\n", encoding="utf-8"
    )
    active.log.info("data: %s", json.dumps(data_report["train"]))
    summary = train.train(
        model,
        examples,
        val,
        run_dir=active.dir,
        lr=float(cfg.train.lr),
        epochs=float(cfg.train.epochs),
        max_len=int(cfg.train.bin_tokens),
        tokens_per_step=int(cfg.train.tokens_per_step),
        warmup_frac=float(cfg.train.warmup_frac),
        weight_decay=float(cfg.train.weight_decay),
        max_grad_norm=float(cfg.train.max_grad_norm),
        eval_every=int(cfg.train.eval_every),
        token_budget=int(cfg.train.token_budget) if cfg.train.token_budget else None,
        seed=int(cfg.seed),
    )
    summary["template_id"] = data_report["template_id"]
    summary["trainable_params"] = checks["trainable_params"]
    if cfg.train.save_adapter:
        model.save_pretrained(active.dir / "adapter")
        summary["adapter"] = str(active.dir / "adapter")
    # `train.train` already wrote summary.json, so these last three fields only ever existed in
    # the returned dict - and every consumer reads the *file* back (12.2's pipeline resumes an arm
    # from it). Losing `adapter` that way is not cosmetic: it silently turned 12.2.7's LoRA arm
    # back into the base model and crashed 12.2.8's export on Path(None).
    (active.dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    active.log_metrics(summary)
    active.finish()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    from src.utils.config import add_config_args, load_config

    parser = argparse.ArgumentParser(description="Phase 12.2 - QLoRA training run")
    add_config_args(parser, default="configs/llm.yaml")
    args = parser.parse_args(argv)
    run(load_config(args.config, args.overrides))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
