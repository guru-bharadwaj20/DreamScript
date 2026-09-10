"""Tests for 11.1.5 - `src.rl.env`: the Gym-shaped API, determinism, and action masking.

`gymnasium` is not installed here, so there is no `check_env` to lean on and the protocol is
tested directly. The row's phrase is "env passes API check", and `api_check` is that check as
code; `test_api_check_passes_on_the_fixture_pool` runs it, while the rest of the file pins the
individual guarantees so a failure says which one broke.
"""

from __future__ import annotations

import random

import pytest

from src.rl import actions as A
from src.rl.abstraction import BOUND, key_index
from src.rl.env import Box, DiagramTraversalEnv, Discrete, api_check, env_determinism
from src.rl.reward import RewardConfig
from src.rl.state import DiagramGraph
from tests.test_rl_episode import DISCONNECTED, SELF_LOOP, SINGLE
from tests.test_rl_state import CYCLE, DIAMOND, LINE, ir

GARBLED = ir(
    [("a", ""), ("b", "  "), ("c", "→§±"), ("d", "x" * 300)], [("a", "b"), ("b", "c"), ("c", "d")]
)
UNRESOLVED = dict(
    ir([("a",), ("b",)], [("a", "b")]),
    unresolved_edges=[{"src": "a", "dst": None, "reason": "no-target"}],
)

#: The pool. Every diagram carries a distinct id, because `reset(options={"diagram_id": ...})`
#: resolves by id and a pool with duplicates would make that lookup ambiguous - which is exactly
#: what the shared `ir()` default produced the first time this file was run.
FIXTURES = [
    dict(doc, id=name)
    for name, doc in (
        ("line", LINE),
        ("diamond", DIAMOND),
        ("cycle", CYCLE),
        ("disconnected", DISCONNECTED),
        ("self-loop", SELF_LOOP),
        ("single", SINGLE),
        ("garbled-ocr", GARBLED),
        ("unresolved-edge", UNRESOLVED),
    )
]


@pytest.fixture
def env():
    return DiagramTraversalEnv([DiagramGraph.from_ir(d) for d in FIXTURES], order="sequential")


def rollout(env, rng, avoid_terminate=True):
    """Play one episode to its terminal, returning the actions taken."""
    actions = []
    while True:
        legal = [a for a, ok in enumerate(env.action_mask()) if ok]
        choices = [a for a in legal if a != A.TERMINATE] if avoid_terminate else legal
        action = rng.choice(choices or legal)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        actions.append(action)
        if terminated or truncated:
            return actions


# -- construction -----------------------------------------------------------------------


def test_env_accepts_raw_ir_dicts_and_graphs():
    mixed = DiagramTraversalEnv([LINE, DiagramGraph.from_ir(DIAMOND)])
    assert len(mixed.graphs) == 2
    assert all(isinstance(g, DiagramGraph) for g in mixed.graphs)


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError):
        DiagramTraversalEnv([])


def test_bad_order_is_rejected():
    with pytest.raises(ValueError):
        DiagramTraversalEnv([LINE], order="sideways")


def test_using_the_env_before_reset_raises(env):
    with pytest.raises(RuntimeError):
        env.step(0)
    with pytest.raises(RuntimeError):
        env.action_mask()


# -- the spaces -------------------------------------------------------------------------


def test_action_space_is_the_action_count(env):
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == A.N_ACTIONS


def test_discrete_contains_only_valid_actions(env):
    assert env.action_space.contains(0)
    assert env.action_space.contains(A.N_ACTIONS - 1)
    assert not env.action_space.contains(A.N_ACTIONS)
    assert not env.action_space.contains(-1)
    assert not env.action_space.contains("terminate")


def test_discrete_sample_is_always_in_range(env):
    rng = random.Random(0)
    assert all(env.action_space.contains(env.action_space.sample(rng)) for _ in range(50))


def test_observation_space_is_a_unit_box(env):
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (28,)
    assert (env.observation_space.low, env.observation_space.high) == (0.0, 1.0)


def test_box_rejects_wrong_width_and_out_of_range(env):
    assert not env.observation_space.contains([0.0] * 27)
    assert not env.observation_space.contains([2.0] * 28)
    assert env.observation_space.contains([0.5] * 28)


# -- reset ------------------------------------------------------------------------------


def test_reset_returns_observation_and_info(env):
    result = env.reset(seed=0)
    assert isinstance(result, tuple) and len(result) == 2
    obs, info = result
    assert len(obs) == env.observation_space.shape[0]
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_reset_info_carries_the_mask_and_the_structured_state(env):
    _obs, info = env.reset(seed=0)
    assert len(info["action_mask"]) == A.N_ACTIONS
    assert info["n_visited"] == 1 or info["n_nodes"] == 0
    assert info["n_emitted"] == 0
    assert "state" in info and "abstract_state" in info
    assert 0 <= key_index(info["abstract_state"]) < BOUND


def test_reset_by_index_selects_the_diagram(env):
    for index in range(len(FIXTURES)):
        _obs, info = env.reset(options={"index": index})
        assert info["diagram_id"] == env.graphs[index].diagram_id


def test_reset_by_diagram_id_selects_the_diagram(env):
    wanted = env.graphs[2].diagram_id
    _obs, info = env.reset(options={"diagram_id": wanted})
    assert info["diagram_id"] == wanted


def test_reset_by_unknown_diagram_id_raises(env):
    with pytest.raises(KeyError):
        env.reset(options={"diagram_id": "no-such-diagram"})


def test_sequential_order_walks_the_pool(env):
    env.reset(seed=0)
    seen = [env.reset()[1]["diagram_id"] for _ in range(len(FIXTURES))]
    assert len(set(seen)) == len(FIXTURES)


def test_reset_clears_the_previous_episode(env):
    env.reset(options={"index": 0})
    env.step(A.EMIT_NODE)
    _obs, info = env.reset(options={"index": 0})
    assert info["n_emitted"] == 0


# -- step -------------------------------------------------------------------------------


def test_step_returns_the_five_tuple(env):
    env.reset(seed=0, options={"index": 0})
    result = env.step(A.EMIT_NODE)
    assert isinstance(result, tuple) and len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert len(obs) == env.observation_space.shape[0]
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_terminated_and_truncated_are_never_both_true(env):
    rng = random.Random(0)
    for index in range(len(FIXTURES)):
        env.reset(seed=0, options={"index": index})
        while True:
            legal = [a for a, ok in enumerate(env.action_mask()) if ok]
            _o, _r, terminated, truncated, _i = env.step(rng.choice(legal))
            assert not (terminated and truncated)
            if terminated or truncated:
                break


def test_terminal_step_carries_the_reward_breakdown(env):
    env.reset(options={"index": 5})  # SINGLE
    _obs, _reward, terminated, _truncated, info = env.step(A.EMIT_NODE)
    assert terminated
    assert "terminal" in info and "stopped_by" in info
    assert info["terminal"]["terminal_reward"] == info["terminal"]["terminal_reward"]


def test_step_after_termination_raises(env):
    env.reset(options={"index": 0})
    while True:
        _o, _r, terminated, truncated, _i = env.step(A.TERMINATE)
        if terminated or truncated:
            break
    with pytest.raises(RuntimeError):
        env.step(A.TERMINATE)


def test_out_of_range_action_raises(env):
    env.reset(options={"index": 0})
    with pytest.raises(ValueError):
        env.step(A.N_ACTIONS)


# -- action masking, the part 11.2 depends on --------------------------------------------


def test_illegal_action_is_refused_and_flagged(env):
    env.reset(options={"index": 0})
    env.step(A.EMIT_NODE)
    assert not env.action_mask()[A.EMIT_NODE]
    _obs, _reward, _t, _tr, info = env.step(A.EMIT_NODE)
    assert info["legal"] is False


def test_a_refused_action_does_not_change_the_traversal(env):
    env.reset(options={"index": 0})
    env.step(A.EMIT_NODE)
    before = env.state
    env.step(A.EMIT_NODE)
    after = env.state
    assert (after.current, after.visited, after.emitted, after.loop_marked, after.stack) == (
        before.current,
        before.visited,
        before.emitted,
        before.loop_marked,
        before.stack,
    )


def test_a_refused_action_is_penalised(env):
    env.reset(options={"index": 0})
    _o, legal_reward, _t, _tr, _i = env.step(A.EMIT_NODE)
    _o, illegal_reward, _t, _tr, info = env.step(A.EMIT_NODE)
    assert info["legal"] is False
    assert illegal_reward < legal_reward


def test_every_legal_action_is_accepted(env):
    rng = random.Random(0)
    for index in range(len(FIXTURES)):
        env.reset(seed=0, options={"index": index})
        while True:
            legal = [a for a, ok in enumerate(env.action_mask()) if ok]
            assert legal, "no state may be a dead end"
            _o, _r, terminated, truncated, info = env.step(rng.choice(legal))
            assert info["legal"] is True
            if terminated or truncated:
                break


def test_terminate_is_always_masked_in(env):
    rng = random.Random(2)
    for index in range(len(FIXTURES)):
        env.reset(seed=0, options={"index": index})
        while True:
            assert env.action_mask()[A.TERMINATE]
            legal = [a for a, ok in enumerate(env.action_mask()) if ok if a != A.TERMINATE]
            if not legal:
                break
            _o, _r, terminated, truncated, _i = env.step(rng.choice(legal))
            if terminated or truncated:
                break


# -- determinism ------------------------------------------------------------------------


def test_same_seed_gives_an_identical_trajectory(env):
    assert env_determinism(env, seed=0, episodes=len(FIXTURES))


def test_repeated_seeded_rollouts_match(env):
    def run():
        rng = random.Random(11)
        env.reset(seed=11, options={"index": 1})
        return rollout(env, rng)

    assert run() == run()


def test_observations_are_reproducible(env):
    first, _ = env.reset(seed=5, options={"index": 3})
    second, _ = env.reset(seed=5, options={"index": 3})
    assert first == second


def test_random_order_is_seeded(env):
    pool = DiagramTraversalEnv([DiagramGraph.from_ir(d) for d in FIXTURES], order="random")

    def sample():
        pool.reset(seed=7)
        return [pool.reset()[1]["diagram_id"] for _ in range(6)]

    assert sample() == sample()


# -- render, and the extras ---------------------------------------------------------------


def test_render_before_reset_is_none(env):
    assert env.render() is None


def test_render_returns_text_naming_the_diagram(env):
    _obs, info = env.reset(options={"index": 0})
    text = env.render()
    assert isinstance(text, str)
    assert info["diagram_id"] in text
    assert "legal:" in text


def test_render_marks_the_current_node(env):
    env.reset(options={"index": 0})
    assert ">" in env.render()


def test_human_mode_prints(env, capsys):
    printer = DiagramTraversalEnv([DiagramGraph.from_ir(LINE)], render_mode="human")
    printer.reset(options={"index": 0})
    printer.render()
    assert capsys.readouterr().out.strip()


def test_close_clears_the_episode(env):
    env.reset(options={"index": 0})
    env.close()
    with pytest.raises(RuntimeError):
        env.action_mask()


def test_abstract_state_is_always_bounded(env):
    rng = random.Random(0)
    for index in range(len(FIXTURES)):
        env.reset(seed=0, options={"index": index})
        while True:
            assert 0 <= key_index(env.abstract_state()) < BOUND
            legal = [a for a, ok in enumerate(env.action_mask()) if ok]
            _o, _r, terminated, truncated, _i = env.step(rng.choice(legal))
            if terminated or truncated:
                break


def test_emitted_code_parses_at_every_step(env):
    import ast

    rng = random.Random(0)
    for index in range(len(FIXTURES)):
        env.reset(seed=0, options={"index": index})
        while True:
            ast.parse(env.emitted_code())
            legal = [a for a, ok in enumerate(env.action_mask()) if ok]
            _o, _r, terminated, truncated, _i = env.step(rng.choice(legal))
            if terminated or truncated:
                break
        ast.parse(env.emitted_code())


# -- robustness on the shapes real diagrams actually have ---------------------------------


@pytest.mark.parametrize("index", range(len(FIXTURES)))
def test_every_fixture_plays_to_termination_without_crashing(index):
    """Cycles, disconnected components, self-loops, unresolved edges and garbled OCR text."""
    env = DiagramTraversalEnv([DiagramGraph.from_ir(d) for d in FIXTURES], order="sequential")
    rng = random.Random(index)
    for seed in range(5):
        env.reset(seed=seed, options={"index": index})
        rollout(env, rng)
        env.render()
        env.emitted_code()


def test_an_empty_diagram_can_only_terminate():
    env = DiagramTraversalEnv([{"nodes": [], "edges": []}])
    _obs, info = env.reset(seed=0)
    assert [a for a, ok in enumerate(info["action_mask"]) if ok] == [A.TERMINATE]
    _o, _r, terminated, _tr, _i = env.step(A.TERMINATE)
    assert terminated


def test_max_steps_overrides_the_cap():
    env = DiagramTraversalEnv([DiagramGraph.from_ir(CYCLE)], max_steps=4)
    _obs, info = env.reset(seed=0)
    assert info["step_cap"] == 4


def test_a_policy_that_never_terminates_is_truncated():
    """The step cap is the only guarantee against an agent that refuses to stop."""
    env = DiagramTraversalEnv([DiagramGraph.from_ir(CYCLE)], max_steps=6)
    env.reset(seed=0)
    rng = random.Random(0)
    truncated_or_done = False
    for _ in range(50):
        legal = [a for a, ok in enumerate(env.action_mask()) if ok and a != A.TERMINATE]
        if not legal:
            break
        _o, _r, terminated, truncated, _i = env.step(rng.choice(legal))
        if terminated or truncated:
            truncated_or_done = True
            break
    assert truncated_or_done


def test_a_custom_reward_config_is_used():
    env = DiagramTraversalEnv([DiagramGraph.from_ir(LINE)], config=RewardConfig(step_cost=-5.0))
    env.reset(seed=0)
    _obs, reward, _t, _tr, _i = env.step(A.EMIT_NODE)
    assert reward == pytest.approx(-5.0)


# -- the API check itself -----------------------------------------------------------------


def test_api_check_passes_on_the_fixture_pool(env):
    report = api_check(env, seeds=(0, 1, 2), episodes=len(FIXTURES))
    assert report["ok"] is True, report["checks"]
    assert report["observation_width"] == 28
    assert report["n_actions"] == A.N_ACTIONS
    assert report["deterministic_under_same_seed"] is True


def test_api_check_probes_every_guarantee(env):
    report = api_check(env, seeds=(0, 1), episodes=len(FIXTURES))
    for name, counts in report["checks"].items():
        assert counts["probes"] > 0, f"{name} was never probed"
        assert counts["failures"] == 0, name
