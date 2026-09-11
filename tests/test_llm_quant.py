"""Phase 12.2.2 - the NF4 config, LoRA target set and saturation rule (no GPU, no weights)."""

from __future__ import annotations

import torch

from src.llm import quant


def test_bnb_config_is_nf4_with_bf16_compute() -> None:
    cfg = quant.bnb_config()
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_compute_dtype == torch.bfloat16
    assert cfg.bnb_4bit_use_double_quant


def test_lora_targets_every_linear_projection() -> None:
    cfg = quant.lora_config(r=16)
    assert set(cfg.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert cfg.lora_alpha == 32 and cfg.r == 16


def test_saturation_picks_the_smallest_batch_near_the_best_throughput() -> None:
    cells = [
        {"seq_len": 1024, "batch": 1, "status": "ok", "tokens_per_s": 968.0},
        {"seq_len": 1024, "batch": 2, "status": "ok", "tokens_per_s": 989.1},
        {"seq_len": 1024, "batch": 4, "status": "ok", "tokens_per_s": 937.9},
        {"seq_len": 1024, "batch": 8, "status": "oom"},
        {"seq_len": 4096, "batch": 1, "status": "ok", "tokens_per_s": 888.4},
        {"seq_len": 4096, "batch": 2, "status": "oom"},
    ]
    assert quant.saturation(cells) == {1024: 1, 4096: 1}
    assert quant.saturation(cells, tolerance=0.01) == {1024: 2, 4096: 1}


def test_vram_cap_leaves_room_below_the_card() -> None:
    assert quant.VRAM_CAP_GIB <= 24.0 - 3.0 - 0.5
