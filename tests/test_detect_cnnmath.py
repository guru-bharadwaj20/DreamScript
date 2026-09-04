"""Phase 9.2.2 / 9.2.3 - the arithmetic, checked on cases whose answer is known independently."""

from __future__ import annotations

import pytest

from src.detect import cnnmath
from src.detect.scratchcnn import Spec

# -- the check that makes the document worth reading -----------------------------------------


def test_the_hand_computed_total_equals_what_torch_reports():
    """If these ever disagree, `docs/cnn_math.md` is fiction."""
    check = cnnmath.check_against_torch(Spec())
    assert check["agree"], check


@pytest.mark.parametrize(
    "spec",
    [
        Spec(kernel=5),
        Spec(batchnorm=False),
        Spec(pool="avg"),
        Spec(stages=((8, 1), (16, 1), (32, 1), (64, 1))),
        Spec(hidden=64),
    ],
)
def test_the_arithmetic_holds_for_every_configuration_9_2_5_will_ablate(spec):
    assert cnnmath.check_against_torch(spec)["agree"]


# -- parameter counting ----------------------------------------------------------------------


def test_a_conv_layer_is_out_times_in_times_k_squared():
    """conv 1->16 3x3 with BatchNorm following: 16*1*9 = 144, no bias."""
    rows = cnnmath.walk(Spec())
    assert rows[0]["params"] == 144


def test_the_bias_comes_back_when_batchnorm_is_off():
    rows = cnnmath.walk(Spec(batchnorm=False))
    assert rows[0]["params"] == 16 * 1 * 9 + 16


def test_batchnorm_costs_two_parameters_per_channel():
    """Scale and shift are parameters; the running mean and variance are buffers and are not."""
    rows = cnnmath.walk(Spec())
    norm = next(r for r in rows if r["layer"].startswith("batchnorm"))
    assert norm["params"] == 2 * 16


def test_pooling_and_relu_have_no_parameters():
    for row in cnnmath.walk(Spec()):
        if row["layer"].startswith(("maxpool", "avgpool", "relu", "flatten", "dropout")):
            assert row["params"] == 0


def test_a_five_by_five_kernel_costs_25_over_9_of_a_three_by_three():
    three = cnnmath.totals(cnnmath.walk(Spec(kernel=3)), Spec(kernel=3))["conv"]
    five = cnnmath.totals(cnnmath.walk(Spec(kernel=5)), Spec(kernel=5))["conv"]
    assert five / three == pytest.approx(25 / 9, rel=1e-6)


# -- output sizes ---------------------------------------------------------------------------


def test_same_padding_keeps_the_map_size_through_a_convolution():
    rows = cnnmath.walk(Spec())
    conv = rows[0]
    assert conv["out"][1:] == conv["in"][1:]


def test_each_pool_halves_the_map():
    sides = [r["out"][1] for r in cnnmath.walk(Spec()) if r["layer"].startswith("maxpool")]
    assert sides == [32, 16, 8, 4]


def test_the_flatten_width_is_channels_times_area():
    rows = cnnmath.walk(Spec())
    flatten = next(r for r in rows if r["layer"] == "flatten")
    assert flatten["out"] == [128 * 4 * 4]


# -- receptive field -------------------------------------------------------------------------


def test_the_field_starts_at_one_pixel_and_grows_by_k_minus_one_times_the_jump():
    rows = cnnmath.walk(Spec())
    # first conv: 1 + (3-1)*1 = 3; second conv: 3 + 2*1 = 5; first pool: 5 + 1*1 = 6
    assert [rows[0]["receptive_field"], rows[3]["receptive_field"]] == [3, 5]
    pool = next(r for r in rows if r["layer"].startswith("maxpool"))
    assert pool["receptive_field"] == 6


def test_the_jump_doubles_at_every_pool_and_nowhere_else():
    jumps = [r["jump"] for r in cnnmath.walk(Spec()) if r["layer"].startswith("maxpool")]
    assert jumps == [2, 4, 8, 16]


def test_the_last_convolution_does_not_see_the_whole_crop():
    """The finding 9.2.3 exists to surface: 52 of 64 pixels, so the FC layer does the
    integrating that a shape - a global property - requires."""
    summary = cnnmath.totals(cnnmath.walk(Spec()), Spec())
    assert summary["final_receptive_field"] == 52
    assert summary["covers_input"] is False
    assert summary["field_shortfall"] == 12


def test_a_wider_kernel_widens_the_field():
    def field(k):
        spec = Spec(kernel=k)
        return cnnmath.totals(cnnmath.walk(spec), spec)["final_receptive_field"]

    assert field(7) > field(5) > field(3)


def test_two_stacked_three_by_threes_match_one_five_by_five():
    """The substitution the architecture is built on, verified rather than asserted."""
    stacked = cnnmath.walk(Spec(kernel=3))[3]["receptive_field"]  # after the second conv
    single = cnnmath.walk(Spec(kernel=5))[0]["receptive_field"]
    assert stacked == single == 5


# -- where the parameters live ---------------------------------------------------------------


def test_the_fully_connected_head_holds_most_of_the_parameters():
    summary = cnnmath.totals(cnnmath.walk(Spec()), Spec())
    assert summary["fc_share"] > 0.65
    assert summary["fully_connected"] > summary["conv"]


def test_the_document_names_both_findings(tmp_path):
    path = cnnmath.write_doc(Spec(), tmp_path / "cnn_math.md")
    text = path.read_text(encoding="utf-8")
    assert "372,183" in text
    assert "they agree" in text
    assert "52x52" in text
