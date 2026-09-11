"""Phase 11.2.5 - the DQN, and the batched machinery that makes it cheap.

The tests pin the things that would make a DQN number quietly wrong rather than merely bad: an
argmax or bootstrap that can pick an illegal action, a stacked ensemble that is not really K
independent networks (shared optimiser state, a clip over all members at once, padded input
columns that learn), a captured CUDA graph that does not compute the eager update, and a cached
reward that disagrees with the reward. Learning outcomes on the corpus are not asserted - they are
in `reports/rl_dqn.md` - because a test that pins a learning result fails when the result is
honestly negative.
"""

from __future__ import annotations

import random

import pytest
import torch

from src.rl import actions as A
from src.rl import dqn as D
from src.rl import qlearning as Q
from src.rl.episode import Episode
from src.rl.reward import terminal_reward
from src.rl.state import DiagramGraph

# ------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------


def _node(node_id: str, x: float, y: float, role: str = "process") -> dict:
    return {
        "id": node_id,
        "shape": "process",
        "bbox": [x, y, 10.0, 10.0],
        "text": node_id,
        "semantic_role": role,
        "confidence": 1.0,
    }


def _edge(edge_id: str, src: str, dst: str) -> dict:
    return {"id": edge_id, "src": src, "dst": dst, "directed": True, "label": ""}


LINE = {
    "id": "line",
    "nodes": [_node("a", 0, 0, "start"), _node("b", 0, 10), _node("c", 0, 20, "end")],
    "edges": [_edge("e1", "a", "b"), _edge("e2", "b", "c")],
    "meta": {"source": "test"},
}
DIAMOND = {
    "id": "diamond",
    "nodes": [
        _node("a", 0, 0, "decision"),
        _node("b", 0, 10),
        _node("c", 20, 10),
        _node("d", 0, 20),
    ],
    "edges": [_edge("e1", "a", "b"), _edge("e2", "a", "c"), _edge("e3", "b", "d")]
    + [_edge("e4", "c", "d")],
    "meta": {"source": "test"},
}
LOOP = {
    "id": "loop",
    "nodes": [_node("a", 0, 0, "start"), _node("b", 0, 10), _node("c", 0, 20)],
    "edges": [_edge("e1", "a", "b"), _edge("e2", "b", "c"), _edge("e3", "c", "b")],
    "meta": {"source": "test"},
}
SPLIT = {
    "id": "split",
    "nodes": [_node("a", 0, 0), _node("b", 0, 10), _node("c", 40, 0), _node("d", 40, 10)],
    "edges": [_edge("e1", "a", "b"), _edge("e2", "c", "d")],
    "meta": {"source": "test"},
}
RAW = [LINE, DIAMOND, LOOP, SPLIT]


@pytest.fixture(scope="module")
def graphs() -> list[DiagramGraph]:
    return [DiagramGraph.from_ir(d) for d in RAW]


def _cpu(**kwargs) -> D.DQNConfig:
    base = {"device": "cpu", "hidden": 16, "batch": 8, "buffer": 500, "warmup": 16, "n_envs": 4}
    return D.DQNConfig(**{**base, **kwargs})


# ------------------------------------------------------------------------------------------
# masking
# ------------------------------------------------------------------------------------------


def test_masked_argmax_never_picks_an_illegal_action() -> None:
    q = torch.tensor([[9.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    mask = torch.tensor([[False, True, False, False, False, False, False, False, True]])
    assert D.masked_argmax(q, mask).item() == 1
    stacked = torch.stack([q, q])
    assert D.masked_argmax(stacked, torch.stack([mask, mask])).tolist() == [[1], [1]]


def test_bootstrap_reads_only_the_legal_actions_of_the_next_state() -> None:
    """An illegal action with a huge Q-value in s' must not leak into the target."""
    cfg = _cpu(gamma=1.0, batch=1, buffer=4)
    learner = D.Learner([cfg], torch.device("cpu"), capture=False)
    with torch.no_grad():
        for p in learner.target.parameters():
            p.zero_()
        learner.target.b3[0, 0] = 1000.0  # follow-edge-0: illegal in the next state below
        learner.target.b3[0, A.TERMINATE] = 2.0
        for p in learner.online.parameters():
            p.zero_()
        learner.online.b3[0, 0] = 1000.0
        learner.online.b3[0, A.TERMINATE] = 2.0
    mask = [[[False] * A.N_ACTIONS]]
    mask[0][0][A.TERMINATE] = True
    import numpy as np

    obs = np.zeros((1, 1, D.WIDE), dtype=np.float32)
    learner.replay.add(
        obs,
        np.array([[A.EMIT_NODE]]),
        np.array([[0.5]], dtype=np.float32),
        obs,
        np.array(mask),
        np.array([[0.0]], dtype=np.float32),
    )
    # q_taken is 0 (EMIT_NODE output is zeroed); the masked target is 0.5 + 1.0 * 2.0 = 2.5.
    learner.update(torch.zeros((1, 1), dtype=torch.long))
    assert learner.td_out.item() == pytest.approx(2.5)


def test_a_terminal_transition_does_not_bootstrap() -> None:
    import numpy as np

    cfg = _cpu(gamma=1.0, batch=1, buffer=4)
    learner = D.Learner([cfg], torch.device("cpu"), capture=False)
    with torch.no_grad():
        for p in list(learner.target.parameters()) + list(learner.online.parameters()):
            p.zero_()
        learner.target.b3[0, :] = 50.0
    obs = np.zeros((1, 1, D.WIDE), dtype=np.float32)
    learner.replay.add(
        obs,
        np.array([[A.TERMINATE]]),
        np.array([[-1.25]], dtype=np.float32),
        obs,
        np.ones((1, 1, A.N_ACTIONS), dtype=bool),
        np.array([[1.0]], dtype=np.float32),
    )
    learner.update(torch.zeros((1, 1), dtype=torch.long))
    assert learner.td_out.item() == pytest.approx(1.25)


# ------------------------------------------------------------------------------------------
# the ensemble is K independent networks
# ------------------------------------------------------------------------------------------


def test_ensemble_member_is_the_qnet_its_seed_would_build() -> None:
    cfgs = [_cpu(seed=3, inputs="features"), _cpu(seed=5, inputs="features_mask")]
    ens = D.Ensemble(cfgs)
    x = torch.rand(2, 6, D.WIDE)
    x[0, :, D.FEATURE_WIDTH :] = 0.0
    out = ens(x)
    for k, cfg in enumerate(cfgs):
        torch.manual_seed(cfg.seed)
        net = D.QNet(cfg.width, cfg.hidden)
        assert torch.allclose(out[k], net(x[k, :, : cfg.width]), atol=1e-6)
        assert torch.allclose(ens.member(k)(x[k, :, : cfg.width]), out[k], atol=1e-6)


def test_stacked_adam_is_k_independent_adams_with_their_own_lr_and_clip() -> None:
    torch.set_default_dtype(torch.float64)
    try:
        cfgs = [_cpu(seed=0, lr=1e-2, grad_clip=0.05), _cpu(seed=1, lr=3e-3, grad_clip=1e9)]
        ens = D.Ensemble(cfgs)
        opt = D.StackedAdam(list(ens.parameters()), [c.lr for c in cfgs], 0.05, torch.device("cpu"))
        opt.clip = torch.tensor([0.05, 1e9])
        nets, opts = [], []
        for cfg in cfgs:
            torch.manual_seed(cfg.seed)
            nets.append(D.QNet(cfg.width, cfg.hidden))
            opts.append(torch.optim.Adam(nets[-1].parameters(), lr=cfg.lr))
        for _ in range(15):
            x = torch.rand(2, 5, D.WIDE)
            x[:, :, D.FEATURE_WIDTH :] = 0.0
            opt.zero_grad()
            ens(x).pow(2).mean(dim=(1, 2)).sum().backward()
            opt.advance()
            opt.step()
            for k, (cfg, net, o) in enumerate(zip(cfgs, nets, opts, strict=True)):
                o.zero_grad()
                net(x[k, :, : cfg.width]).pow(2).mean().backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                o.step()
        for k, net in enumerate(nets):
            member = ens.member(k).double()
            for a, b in zip(member.parameters(), net.parameters(), strict=True):
                assert torch.allclose(a, b, atol=1e-6)
    finally:
        torch.set_default_dtype(torch.float32)


def test_padded_input_columns_never_learn(graphs) -> None:
    """A "features" member's mask columns stay exactly zero through real training."""
    cfgs = [_cpu(episodes=60, seed=0), _cpu(episodes=60, seed=1, inputs="features_mask")]
    learner_seen: list[D.Learner] = []
    original = D.Learner.update

    def spy(self, idx=None):
        if self not in learner_seen:
            learner_seen.append(self)
        return original(self, idx)

    D.Learner.update = spy
    try:
        agents = D.train_many(graphs, cfgs)
    finally:
        D.Learner.update = original
    assert learner_seen, "training never reached an update"
    w1 = learner_seen[0].online.w1
    assert w1[0, :, D.FEATURE_WIDTH :].abs().max().item() == 0.0
    assert w1[1, :, D.FEATURE_WIDTH :].abs().max().item() > 0.0
    assert [a.net.body[0].weight.shape[1] for a in agents] == [D.FEATURE_WIDTH, D.WIDE]


def test_members_must_share_the_non_member_fields() -> None:
    with pytest.raises(ValueError, match="must share"):
        D.train_many([DiagramGraph.from_ir(LINE)], [_cpu(batch=8), _cpu(batch=16)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_captured_update_is_bit_identical_to_the_eager_update() -> None:
    import numpy as np

    device = torch.device("cuda")
    cfgs = [D.DQNConfig(seed=0, buffer=300, batch=32), D.DQNConfig(seed=1, buffer=300, batch=32)]
    cfgs[1].lr, cfgs[1].gamma, cfgs[1].inputs = 3e-4, 0.95, "features_mask"
    learners = [D.Learner(cfgs, device, capture=c) for c in (True, False)]
    rng = np.random.default_rng(0)
    shape = (2, 300)
    obs = rng.random(shape + (D.WIDE,), dtype=np.float32)
    nxt = rng.random(shape + (D.WIDE,), dtype=np.float32)
    mask = rng.random(shape + (A.N_ACTIONS,)) < 0.5
    mask[..., A.TERMINATE] = True
    act = rng.integers(0, A.N_ACTIONS, shape)
    rew = rng.normal(size=shape).astype(np.float32)
    done = (rng.random(shape) < 0.1).astype(np.float32)
    for learner in learners:
        learner.replay.add(obs, act, rew, nxt, mask, done)
    generator = torch.Generator(device=device).manual_seed(1)
    for step in range(60):
        idx = torch.randint(0, 300, (2, 32), device=device, generator=generator)
        for learner in learners:
            learner.update(idx)
            if step % 20 == 19:
                learner.sync_target()
    for a, b in zip(learners[0].online.parameters(), learners[1].online.parameters(), strict=True):
        assert torch.equal(a, b)


# ------------------------------------------------------------------------------------------
# the loop agrees with the reward and the episode
# ------------------------------------------------------------------------------------------


def test_terminal_cache_returns_exactly_the_terminal_reward(graphs) -> None:
    cache = D.TerminalCache(D.DEFAULT)
    rng = random.Random(0)
    for _ in range(40):
        graph = rng.choice(graphs)
        episode = Episode(graph)
        while not episode.done():
            episode.apply(rng.choice(episode.legal_actions()))
        expected = terminal_reward(graph, episode.state, episode.truncated())[0]
        assert cache(graph, episode) == expected
    assert cache.hits > 0


def test_epsilon_schedule_matches_the_tabular_one() -> None:
    dqn_cfg = D.DQNConfig(episodes=1000)
    q_cfg = Q.TrainConfig(episodes=1000)
    for episode in (0, 1, 250, 999):
        assert D.epsilon_at(dqn_cfg, episode) == pytest.approx(Q.epsilon_at(q_cfg, episode))


def test_trained_agent_plays_legally_and_its_arm_is_a_permutation(graphs) -> None:
    agent = D.train(graphs, _cpu(episodes=80, seed=2))
    assert len(agent.history["reward"]) == 80
    for raw, graph in zip(RAW, graphs, strict=True):
        episode = agent.rollout(graph)
        assert episode.done()
        assert all(outcome.legal for outcome in episode.history)
        order = agent.arm(raw)
        assert sorted(order) == sorted(n["id"] for n in raw["nodes"])


def test_batched_rollout_equals_one_at_a_time(graphs) -> None:
    agent = D.train(graphs, _cpu(episodes=40, seed=4))
    batched = agent.rollout_many(graphs)
    for graph, episode in zip(graphs, batched, strict=True):
        assert episode.state == agent.rollout(graph).state


def test_save_and_load_round_trip(tmp_path, graphs) -> None:
    agent = D.train(graphs, _cpu(episodes=30, seed=5, inputs="features_mask"))
    path = tmp_path / "agent.pt"
    agent.save(path)
    again = D.DQNAgent.load(path, device="cpu")
    assert again.cfg.inputs == "features_mask"
    for graph in graphs:
        assert again.rollout(graph).state == agent.rollout(graph).state


# ------------------------------------------------------------------------------------------
# the hypothesis instruments
# ------------------------------------------------------------------------------------------


def test_aliasing_counts_only_keys_whose_gold_actions_disagree(graphs) -> None:
    report = D.aliasing(graphs)
    assert report["gold_states"] > 0
    for rep in ("abstract_key", "features", "features_mask"):
        block = report[rep]
        assert 0 <= block["gold_states_on_conflicting_key"] <= report["gold_states"]
    # the vector is at least as fine as the key on every state gold visits
    assert report["features"]["distinct_keys"] >= report["abstract_key"]["distinct_keys"]


def test_table_policy_reproduces_gold_where_its_key_is_unambiguous(graphs) -> None:
    table = D.TablePolicy([DiagramGraph.from_ir(LINE)])
    line = DiagramGraph.from_ir(LINE)
    assert table.rollout(line).state.emit_sequence == Q.gold_player(line).state.emit_sequence


def test_split_is_disjoint_deterministic_and_covers_everything() -> None:
    diagrams = [{"id": f"d{i}"} for i in range(50)]
    fake = list(range(50))
    (tr_d, tr_g), (te_d, te_g) = D.split(diagrams, fake)
    assert set(tr_g).isdisjoint(te_g) and len(tr_g) + len(te_g) == 50 and len(te_g) == 10
    assert D.split(diagrams, fake)[1][1] == te_g


def test_init_starts_every_member_from_the_given_networks(graphs) -> None:
    first = D.train_many(graphs, [_cpu(episodes=20, seed=0, warmup=10**6)])
    x = torch.rand(3, D.FEATURE_WIDTH)
    again = D.train_many(
        graphs, [_cpu(episodes=20, seed=9, warmup=10**6)], init=[first[0].net]
    )  # warmup never reached, so no update moves the loaded weights
    assert torch.allclose(again[0].net(x), first[0].net(x))
