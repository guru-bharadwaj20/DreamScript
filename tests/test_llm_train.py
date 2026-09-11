"""Phase 12.2.2 / 12.2.5 - packing, the isolation mask, supervised-only loss and the LR schedule.

CPU only: a 2-layer random Qwen2 stands in for the 7B model, which is enough to prove that a
packed row scores exactly what its examples score apart.
"""

from __future__ import annotations

import pytest
import torch

from src.llm import train
from src.llm.train import Example


def _examples() -> list[Example]:
    return [
        Example(list(range(1, 1 + p)), list(range(50, 50 + c)))
        for p, c in [(30, 20), (10, 5), (60, 40), (25, 25), (7, 3)]
    ]


def test_pack_respects_capacity_and_uses_every_example_once() -> None:
    ex = _examples()
    bins = train.pack(ex, 64)
    assert sorted(i for b in bins for i in b) == list(range(len(ex)))
    for b in bins:
        if len(b) > 1:
            assert sum(len(ex[i]) for i in b) <= 64


def test_an_example_longer_than_the_bin_gets_its_own_row_and_is_not_cut() -> None:
    ex = _examples()
    bins = train.pack(ex, 64)
    assert [2] in bins  # the 100-token example
    stats = train.pack_stats(ex, bins, 64)
    assert stats["truncated"] == 0 and stats["own_row"] == 1
    row = train.collate([ex[2]])
    assert row["input_ids"].shape[1] == 100


def test_collate_masks_prompt_labels_and_restarts_positions() -> None:
    a, b = Example([1, 2, 3], [4, 5]), Example([6, 7], [8])
    row = train.collate([a, b])
    assert row["input_ids"].tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert row["labels"].tolist() == [[-100, -100, -100, 4, 5, -100, -100, 8]]
    assert row["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 2]]
    allowed = row["allowed"][0, 0]
    assert not allowed[5, 4]  # the second example cannot see the first
    assert allowed[4, 0] and not allowed[0, 4]  # causal inside an example


def test_cosine_lr_warms_up_then_decays_to_the_floor() -> None:
    peak = 2e-4
    lrs = [train.cosine_lr(s, 100, 10, peak) for s in range(101)]
    assert lrs[0] == pytest.approx(peak / 10)
    assert max(lrs) == pytest.approx(peak)
    assert lrs[100] == pytest.approx(0.1 * peak)
    assert all(x >= y - 1e-12 for x, y in zip(lrs[10:], lrs[11:], strict=False))


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="sdpa",
    )
    return Qwen2ForCausalLM(cfg).eval()


def test_packed_row_loss_equals_the_examples_scored_apart(tiny_model) -> None:
    ex = [Example([3, 9, 11, 4], [20, 21, 22]), Example([5, 6], [30, 31, 32, 33])]
    with torch.no_grad():
        packed, n = train.selected_loss(tiny_model, train.collate(ex))
        parts = [train.selected_loss(tiny_model, train.collate([e])) for e in ex]
    apart = sum(float(v) * k for v, k in parts) / sum(k for _, k in parts)
    assert n == sum(k for _, k in parts) == 3 - 1 + 1 + 4 - 1 + 1
    assert float(packed) == pytest.approx(apart, rel=1e-4)


def test_the_isolation_mask_is_load_bearing(tiny_model) -> None:
    """Mutation: a plain causal mask over the same packed row must change the loss."""
    ex = [Example([3, 9, 11, 4], [20, 21, 22]), Example([5, 6], [30, 31, 32, 33])]
    row = train.collate(ex)
    leaky = dict(row, allowed=torch.tril(torch.ones_like(row["allowed"])))
    with torch.no_grad():
        good, _ = train.selected_loss(tiny_model, row)
        bad, _ = train.selected_loss(tiny_model, leaky)
    assert abs(float(good) - float(bad)) > 1e-4


def test_selected_loss_matches_the_full_logits_loss(tiny_model) -> None:
    ex = Example([3, 9, 11, 4], [20, 21, 22])
    row = train.collate([ex])
    with torch.no_grad():
        ours, _ = train.selected_loss(tiny_model, row)
        ref = tiny_model(input_ids=row["input_ids"], labels=row["labels"]).loss
    assert float(ours) == pytest.approx(float(ref), rel=1e-5)


def test_train_loop_runs_logs_and_lowers_loss(tiny_model, tmp_path) -> None:
    """The whole loop on CPU: packing, accumulation, cosine LR, val loss, summary."""
    import copy
    import json

    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        copy.deepcopy(tiny_model),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    ex = [Example([1, 2, 3, i % 7 + 4], [10 + i % 5, 20, 30, 40]) for i in range(40)]
    model.train()
    summary = train.train(
        model,
        ex,
        ex[:8],
        run_dir=tmp_path,
        lr=5e-2,
        epochs=3.0,
        max_len=32,
        tokens_per_step=64,
        eval_every=5,
        log_every=1000,
    )
    lines = [json.loads(x) for x in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert lines[0]["step"] == 0 and summary["total_steps"] == lines[-1]["step"]
    assert summary["best_val_loss"] < summary["val_loss_start"]
    assert (tmp_path / "summary.json").is_file()
