"""Phase 11.2.5 - a DQN over 11.1.1's 28-wide state vector, and the test of 11.2.1's hypothesis.

    python -m src.rl.dqn --stage throughput            # the throughput table
    python -m src.rl.dqn --stage prelim --out P.json   # aliasing, imitation ceilings, references
    python -m src.rl.dqn --stage sweep --out S.json    # the whole grid as one ensemble
    python -m src.rl.dqn --stage member --seed 0 [--heldout] --inputs I --lr L --gamma G --out M
    python -m src.rl.dqn --stage merge --parts P.json S.json M0.json ...   # dqn.json + figure
    python -m src.rl.dqn --quick                       # every stage at toy size, for the tests

Stages are separate processes on purpose; "Throughput" below says why.

## The question this row is actually asked

11.2.1's tabular agent plateaus at 0.2729 edge F1 against an in-action-space gold ceiling of
0.3648, and it left one hypothesis for this row: that 11.1.6's abstract key aliases states the
policy needs to tell apart, so no table over that key can represent a good policy. A function
approximator over `StateEncoder.features` is the natural test, because the vector keeps the
continuous fractions, the step count and per-successor flags the key discretises away.

**The hypothesis was built on a misread number, and that is measured here before any training.**
The "25.2% of raw states share an abstract key" quoted in 11.2.1 is not an aliasing rate: 11.1.6
measured that 25.2% of *raw* keys are shared by two or more *diagrams*. The aliasing that matters
for control is narrower - two states on one key whose correct actions differ - and `aliasing`
measures exactly that, over every state gold play visits. `imitation_ceiling` then asks the
sharper question directly: take the best policy each representation can express (the majority
gold action per key for the table, a supervised MLP for the vector), and play it through the same
harness. If the table can express nearly gold, the tabular failure is a learning failure and a
function approximator is not, by itself, the fix.

## The agent

Double DQN with a hard-synced target network, uniform replay and a Huber loss, written in plain
torch - `gymnasium` is absent (11.1.5) and nothing here needs it. Two choices are load-bearing:

    masked argmax, masked bootstrap   the greedy action and the Double-DQN `argmax_a' Q(s', a')`
                                      are both taken over the legal actions only, for the reason
                                      11.2.1 gives: `terminate` is legal everywhere, so an
                                      unmasked max leaks the value of illegal follow slots.
    truncation is terminal            `terminal_reward` pays the whole semantic score and the
                                      -2.0 loop penalty *at* the cap, so the return is complete
                                      there; bootstrapping past it would add a continuation the
                                      reward already priced. Same convention as 11.2.1, so the
                                      two agents optimise the identical objective.

Epsilon decays geometrically per *episode* from 1.0 to 0.05, identical to 11.2.1's schedule, and
the budget is counted in episodes for the same reason: equal budgets, or the comparison is about
budget. The reward, the episode, the evaluation harness (`qlearning.play`) and the completion rule
for edge F1 (`qlearning.completed_order`, IR document order) are all 11.2.1's, imported rather
than copied, so a difference between the two agents is a difference in the learner.

## Throughput: where the time goes, and what was chosen

A 28-wide input makes the network nearly free and the Python environment expensive. Profiled
before scaling, the first eager loop spent **9.5 ms per update** on a 256-row batch whatever the
batch or member count - kernel-launch overhead, not arithmetic - so the loop is built around that:

    CUDA-graph update       the whole update (gather, Double-DQN target, backward, Adam) is one
                            captured graph: **7.43 ms -> 1.25 ms per update** for 3 members at
                            batch 256. The capture pass is undone and replayed, and a device
                            synchronise precedes the side-stream warm-up; both were bugs found by
                            measurement (a first version silently dropped the first update and
                            diverged from eager; a second crashed 2 of 6 concurrent runs with an
                            illegal memory access). A test pins capture == eager bit for bit.
    stacked members         seeds or sweep cells as one `Ensemble` of `(K, out, in)` weights run
                            with `baddbmm`, each member with its own replay, target, envs,
                            epsilon, learning rate and gamma. Adam is element-wise, so
                            `StackedAdam` is K independent Adams with a per-member clip (tested
                            against `torch.optim.Adam` + `clip_grad_norm_`).
    cached terminal reward  `terminal_reward` is pure and costs an `ast` walk; memoised per final
                            state it hit 77% of calls in the sweep.

Measured, replay ratio held at 64 samples per transition (`--stage throughput`, 3,000 episodes):

    members  n_envs  batch   transitions/s   env ms/transition   update ms   peak alloc MB
       1       16     256        1,538             0.49            0.90          106
       1       32     256        1,766             0.41            0.82          122
       1       64     256        1,948             0.35            0.86          138
       1      128     256        2,176             0.30            0.88          154
       3       64     256        3,861             0.16            1.40          332
       3      128     256        3,452             0.19            1.39          349
       8       64     256        5,689             0.14            1.43          689
       8      128     256        5,347             0.14            1.72          705

Transitions/s flattens at `n_envs` 64 once members are stacked, so 64 is the default; batch 1024
was faster (3,105 t/s) only by taking 4x fewer gradient steps, which changes the algorithm and was
not adopted. After capture the Python environment is the bottleneck (sweep: 428 s env against 124
s update). **Stacking was therefore not always the faster choice, and the measurement decided**:
at 15,000 episodes three single-member processes ran **10.2k transitions/s together (+1,077 MiB
device memory) against 6,961 for one 3-member ensemble (+499 MiB)**, because the env loop is
CPU-bound and parallelises over processes. The 8-cell sweep ran as one ensemble (553 s, 559 MB
allocated); the six 60k-episode seed runs ran as six single-member processes. Everything stayed
under 3 GB.

The measured learning results are in `reports/rl_dqn.md` and `experiments/rl/dqn.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import nn

from src.rl import actions as A
from src.rl.abstraction import abstract, key_index
from src.rl.episode import Episode
from src.rl.qlearning import RUNS, completed_order, play
from src.rl.reward import DEFAULT, RewardConfig, step_reward, terminal_reward
from src.rl.state import DiagramGraph, StateEncoder
from src.utils.config import ROOT

OUT = RUNS / "dqn.json"
CHECKPOINTS = RUNS / "dqn"
FIGURE = ROOT / "reports" / "figures" / "p11_dqn.png"

FEATURE_WIDTH = 28
#: The ensemble always carries the widest input; a "features" member sees the mask columns as 0.
WIDE = FEATURE_WIDTH + A.N_ACTIONS

#: Seeds for the headline comparison. conventions.md section 7 asks for at least three.
SEEDS: tuple[int, ...] = (0, 1, 2)

#: The sweep grid. Coarse, like 11.2.1's: it is here to show the answer does not hinge on one cell.
LRS: tuple[float, ...] = (1e-3, 3e-4)
GAMMAS: tuple[float, ...] = (0.95, 1.0)
INPUTS: tuple[str, ...] = ("features", "features_mask")

#: The tabular comparators. `tabular` is 11.2.1's selected cell; `tabular_best` combines the
#: best settings 11.2.2-11.2.3 found one at a time (alpha 0.2, gamma 0.99, epsilon floor 0.01),
#: so the DQN is compared against the strongest unshaped table on record, not only the first.
TABULAR_CELLS: dict[str, dict[str, float]] = {
    "tabular": {"alpha": 0.4, "gamma": 1.0, "epsilon_min": 0.05},
    "tabular_best": {"alpha": 0.2, "gamma": 0.99, "epsilon_min": 0.01},
}

#: Size buckets for "large graphs where tabular fails" - the plan's own framing of the row.
SIZE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("n<=5", 0, 5),
    ("6-12", 6, 12),
    ("13-25", 13, 25),
    ("n>25", 26, 10**9),
)


# ------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------


@dataclass
class DQNConfig:
    """Everything the DQN loop reads for one member."""

    episodes: int = 60_000
    gamma: float = 1.0
    lr: float = 1e-3
    hidden: int = 256
    #: "features" = the 28-wide `StateEncoder.features`; "features_mask" appends the 9 legal
    #: flags, which the environment already exposes in `info["action_mask"]`.
    inputs: str = "features"
    batch: int = 256
    buffer: int = 200_000
    warmup: int = 5_000
    #: environment transitions (per member) per gradient step
    train_every: int = 4
    target_every: int = 1_000
    n_envs: int = 64
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    grad_clip: float = 10.0
    seed: int = 0
    device: str = "auto"

    @property
    def width(self) -> int:
        return FEATURE_WIDTH + (A.N_ACTIONS if self.inputs == "features_mask" else 0)


#: Fields members of one ensemble may differ in. Everything else must match.
PER_MEMBER = ("seed", "lr", "gamma", "inputs", "episodes")


def epsilon_at(cfg: DQNConfig, episode: int) -> float:
    """Geometric decay over the episode budget - the same curve as `qlearning.epsilon_at`."""
    if cfg.episodes <= 1 or cfg.epsilon <= cfg.epsilon_min:
        return cfg.epsilon_min
    ratio = (cfg.epsilon_min / cfg.epsilon) ** (min(episode, cfg.episodes - 1) / (cfg.episodes - 1))
    return max(cfg.epsilon_min, cfg.epsilon * ratio)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def observe(encoder: StateEncoder, episode: Episode, inputs: str) -> list[float]:
    """The network's input for one state: the 28 features, optionally with the legal mask."""
    vector = list(encoder.features(episode.state))
    if inputs == "features_mask":
        vector += [float(m) for m in episode.mask()]
    return vector


# ------------------------------------------------------------------------------------------
# the network and the agent
# ------------------------------------------------------------------------------------------


class QNet(nn.Module):
    """A two-hidden-layer MLP, `width -> hidden -> hidden -> 9`."""

    def __init__(self, width: int, hidden: int = 256, n_actions: int = A.N_ACTIONS) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax over legal actions only, on the last axis. `terminate` is always legal."""
    return q.masked_fill(~mask, float("-inf")).argmax(dim=-1)


class Ensemble(nn.Module):
    """K independent `QNet`s with stacked weights: `forward((K, N, WIDE)) -> (K, N, 9)`.

    Member k is initialised as `QNet(width_k, hidden)` under `torch.manual_seed(seed_k)` and then
    padded to `WIDE` with zero input columns, so a "features" member computes exactly the function
    its own 28-wide `QNet` would, and its padded columns receive zero gradient forever.
    """

    def __init__(self, cfgs: Sequence[DQNConfig]) -> None:
        super().__init__()
        hidden = cfgs[0].hidden
        stacks: dict[str, list[torch.Tensor]] = defaultdict(list)
        for cfg in cfgs:
            torch.manual_seed(cfg.seed)
            net = QNet(cfg.width, hidden)
            layers = [net.body[0], net.body[2], net.body[4]]
            w1 = torch.zeros(hidden, WIDE)
            w1[:, : cfg.width] = layers[0].weight.data
            stacks["w1"].append(w1)
            stacks["b1"].append(layers[0].bias.data)
            stacks["w2"].append(layers[1].weight.data)
            stacks["b2"].append(layers[1].bias.data)
            stacks["w3"].append(layers[2].weight.data)
            stacks["b3"].append(layers[2].bias.data)
        for name, tensors in stacks.items():
            setattr(self, name, nn.Parameter(torch.stack(tensors)))
        self.widths = [cfg.width for cfg in cfgs]
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.baddbmm(self.b1[:, None, :], x, self.w1.transpose(1, 2)))
        h = torch.relu(torch.baddbmm(self.b2[:, None, :], h, self.w2.transpose(1, 2)))
        return torch.baddbmm(self.b3[:, None, :], h, self.w3.transpose(1, 2))

    @torch.no_grad()
    def load_member(self, k: int, net: QNet) -> None:
        """Overwrite member k with `net`'s weights (the inverse of `member`)."""
        width = self.widths[k]
        self.w1[k].zero_()
        self.w1[k, :, :width].copy_(net.body[0].weight)
        self.b1[k].copy_(net.body[0].bias)
        self.w2[k].copy_(net.body[2].weight)
        self.b2[k].copy_(net.body[2].bias)
        self.w3[k].copy_(net.body[4].weight)
        self.b3[k].copy_(net.body[4].bias)

    def member(self, k: int) -> QNet:
        """Member k as a standalone `QNet` at its own input width, on the ensemble's device."""
        width = self.widths[k]
        net = QNet(width, self.hidden).to(self.w1.device)
        with torch.no_grad():
            net.body[0].weight.copy_(self.w1[k, :, :width])
            net.body[0].bias.copy_(self.b1[k])
            net.body[2].weight.copy_(self.w2[k])
            net.body[2].bias.copy_(self.b2[k])
            net.body[4].weight.copy_(self.w3[k])
            net.body[4].bias.copy_(self.b3[k])
        return net


class StackedAdam:
    """Adam over stacked `(K, ...)` parameters with a per-member learning rate and grad clip.

    Adam is element-wise, so this is exactly K independent `torch.optim.Adam`s (default betas and
    eps); what the stock optimiser cannot do is give each member its own learning rate, and
    `clip_grad_norm_` would clip on the norm of all K members together.
    """

    def __init__(self, params, lrs: Sequence[float], clip: float, device: torch.device) -> None:
        self.params = list(params)
        self.lr = torch.tensor(lrs, dtype=torch.float32, device=device)
        self.clip = clip
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        #: bias corrections live in device tensors so `step` can run inside a captured CUDA graph
        self.c1 = torch.ones((), device=device)
        self.c2 = torch.ones((), device=device)

    def advance(self) -> None:
        """Increment the step count. Called outside `step` so `step` has no Python-side state."""
        self.t += 1
        self.c1.fill_(1 - self.b1**self.t)
        self.c2.fill_(1 - self.b2**self.t)

    @torch.no_grad()
    def step(self) -> None:
        sq = sum(p.grad.pow(2).flatten(1).sum(1) for p in self.params)
        scale = (self.clip / (sq.sqrt() + 1e-6)).clamp(max=1.0)
        c1, c2 = self.c1, self.c2
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            shape = (-1,) + (1,) * (p.dim() - 1)
            g = p.grad * scale.view(shape)
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = (v / c2).sqrt_().add_(self.eps)
            p.sub_(self.lr.view(shape) * (m / c1) / denom)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


class DQNAgent:
    """A trained network plus the policies that read it. Mirrors `TabularAgent`'s surface."""

    def __init__(self, cfg: DQNConfig | None = None, net: QNet | None = None) -> None:
        self.cfg = cfg or DQNConfig()
        self.device = resolve_device(self.cfg.device)
        if net is None:
            torch.manual_seed(self.cfg.seed)
            net = QNet(self.cfg.width, self.cfg.hidden)
        self.net = net.to(self.device)
        self.history: dict[str, Any] = {}
        self.report: dict[str, Any] = {}

    @torch.no_grad()
    def greedy_batch(self, rows: list[list[float]], masks: list[list[bool]]) -> list[int]:
        x = torch.tensor(rows, dtype=torch.float32, device=self.device)
        m = torch.tensor(masks, dtype=torch.bool, device=self.device)
        return masked_argmax(self.net(x), m).tolist()

    def rollout(self, graph: DiagramGraph, seed: int = 0, cap: int | None = None) -> Episode:
        """Play one diagram greedily to the end. `seed` is accepted for `TabularAgent` parity."""
        return self.rollout_many([graph], cap=cap)[0]

    def rollout_many(self, graphs: Sequence[DiagramGraph], cap: int | None = None) -> list[Episode]:
        """Greedy play on many diagrams at once: one forward per step across every live episode."""
        episodes = [Episode(g, cap=cap) for g in graphs]
        encoders = [StateEncoder(g) for g in graphs]
        live = [i for i, e in enumerate(episodes) if not e.done()]
        while live:
            rows = [observe(encoders[i], episodes[i], self.cfg.inputs) for i in live]
            masks = [episodes[i].mask() for i in live]
            for i, action in zip(live, self.greedy_batch(rows, masks), strict=True):
                episodes[i].apply(action)
            live = [i for i in live if not episodes[i].done()]
        return episodes

    def order(self, graph: DiagramGraph) -> list[str]:
        return completed_order(graph, self.rollout(graph))

    def arm(self, diagram: dict) -> list[str]:
        """`src.rl.baselines.register_arm`-shaped adapter: raw IR dict -> emission order."""
        return self.order(DiagramGraph.from_ir(diagram))

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str = "auto") -> DQNAgent:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        cfg = DQNConfig(**{**blob["cfg"], "device": device})
        net = QNet(cfg.width, cfg.hidden)
        net.load_state_dict(blob["state_dict"])
        return cls(cfg, net)


# ------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------


class Replay:
    """K uniform ring buffers, stacked and held on the training device. All members add together."""

    def __init__(self, members: int, capacity: int, device: torch.device) -> None:
        self.k = members
        self.capacity = capacity
        shape = (members, capacity)
        self.obs = torch.zeros(shape + (WIDE,), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros(shape + (WIDE,), dtype=torch.float32, device=device)
        self.next_mask = torch.zeros(shape + (A.N_ACTIONS,), dtype=torch.bool, device=device)
        self.action = torch.zeros(shape, dtype=torch.long, device=device)
        self.reward = torch.zeros(shape, dtype=torch.float32, device=device)
        self.done = torch.zeros(shape, dtype=torch.float32, device=device)
        self.device = device
        self.size = 0
        self.cursor = 0

    def add(self, obs, action, reward, next_obs, next_mask, done) -> None:
        """Arrays shaped `(K, N, ...)`, one host-to-device copy per field."""
        n = action.shape[1]
        idx = torch.arange(self.cursor, self.cursor + n, device=self.device) % self.capacity
        dev = self.device
        self.obs[:, idx] = torch.from_numpy(obs).to(dev)
        self.next_obs[:, idx] = torch.from_numpy(next_obs).to(dev)
        self.next_mask[:, idx] = torch.from_numpy(next_mask).to(dev)
        self.action[:, idx] = torch.from_numpy(action).to(dev)
        self.reward[:, idx] = torch.from_numpy(reward).to(dev)
        self.done[:, idx] = torch.from_numpy(done).to(dev)
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def gather(self, idx: torch.Tensor):
        """The batch at `(K, B)` indices. Pure indexing, so it can sit inside a CUDA graph."""
        rows = torch.arange(self.k, device=self.device)[:, None]
        return (
            self.obs[rows, idx],
            self.action[rows, idx],
            self.reward[rows, idx],
            self.next_obs[rows, idx],
            self.next_mask[rows, idx],
            self.done[rows, idx],
        )


class TerminalCache:
    """`terminal_reward` memoised on everything it reads. Pure function, so identical results."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg
        self.memo: dict[tuple, float] = {}
        self.hits = 0
        self.calls = 0

    def __call__(self, graph: DiagramGraph, episode: Episode) -> float:
        state = episode.state
        key = (
            id(graph),
            state.emit_sequence,
            state.visited,
            state.loop_marked,
            state.duplicate_emissions,
            episode.truncated(),
        )
        self.calls += 1
        if key in self.memo:
            self.hits += 1
            return self.memo[key]
        value = terminal_reward(graph, state, episode.truncated(), self.cfg)[0]
        self.memo[key] = value
        return value


class _Member:
    """One member's environments, rng, schedule and bookkeeping."""

    def __init__(self, cfg: DQNConfig, graphs: Sequence[DiagramGraph], mask_input: bool) -> None:
        self.cfg = cfg
        self.graphs = graphs
        self.rng = random.Random(cfg.seed)
        self.mask_input = mask_input
        self.started = 0
        self.slots: list[list[Any]] = [self.new_slot() for _ in range(cfg.n_envs)]
        self.rewards: list[float] = []
        self.emitted: list[float] = []
        self.full: list[float] = []
        self.lengths: list[int] = []
        self.stopped: Counter = Counter()
        self.td: list[float] = []
        self.curve: list[list[float]] = []
        self.snapshot: QNet | None = None
        self.next_eval = 0

    def new_slot(self) -> list[Any]:
        graph = self.graphs[self.rng.randrange(len(self.graphs))]
        eps = epsilon_at(self.cfg, self.started)
        self.started += 1
        episode = Episode(graph)
        encoder = StateEncoder(graph)
        # [episode, encoder, epsilon, return, length, current observation, current mask]
        slot = [episode, encoder, eps, 0.0, 0, None, None]
        self.refresh(slot)
        return slot

    def refresh(self, slot: list[Any]) -> None:
        episode, encoder = slot[0], slot[1]
        mask = episode.mask()
        row = np.zeros(WIDE, dtype=np.float32)
        row[:FEATURE_WIDTH] = encoder.features(episode.state)
        if self.mask_input:
            row[FEATURE_WIDTH:] = mask
        slot[5], slot[6] = row, mask

    @property
    def finished(self) -> bool:
        return len(self.rewards) >= self.cfg.episodes


def train_many(
    graphs: Sequence[DiagramGraph],
    cfgs: Sequence[DQNConfig],
    reward_cfg: RewardConfig = DEFAULT,
    progress: bool = False,
    eval_every: int = 0,
    capture: bool = True,
    init: Sequence[QNet] | None = None,
) -> list[DQNAgent]:
    """Train K Double-DQN members in one process and return one `DQNAgent` per config.

    Members share `hidden`, `batch`, `buffer`, `warmup`, `train_every`, `target_every` and
    `n_envs`, and may differ in `PER_MEMBER`. Each member's network is snapshotted the moment it
    finishes its own episode budget, so a member with a smaller budget is not trained further
    while the others finish. `eval_every` > 0 records greedy terminal reward on `graphs` every
    that many episodes - a learning curve on the objective, never on edge F1.
    """
    if not graphs:
        raise ValueError("train needs at least one diagram")
    if not cfgs:
        raise ValueError("train_many needs at least one config")
    base = cfgs[0]
    for cfg in cfgs[1:]:
        for name in asdict(base):
            if name not in PER_MEMBER and getattr(cfg, name) != getattr(base, name):
                raise ValueError(f"ensemble members must share {name!r}")
    device = resolve_device(base.device)
    k_members = len(cfgs)
    learner = Learner(cfgs, device, capture=capture)
    if init is not None:
        # continue from earlier networks (11.2.6's stages); replay and epsilon start afresh
        for k, net in enumerate(init):
            learner.online.load_member(k, net)
            learner.target.load_member(k, net)
    online, replay = learner.online, learner.replay
    members = [_Member(c, graphs, c.inputs == "features_mask") for c in cfgs]
    terminal = TerminalCache(reward_cfg)
    n = base.n_envs

    transitions = 0
    updates = 0
    since_update = 0
    started_at = time.perf_counter()
    env_seconds = 0.0
    update_seconds = 0.0

    while not all(m.finished for m in members):
        t0 = time.perf_counter()
        obs = np.stack([np.stack([s[5] for s in m.slots]) for m in members])
        masks = np.array([[s[6] for s in m.slots] for m in members], dtype=bool)
        with torch.no_grad():
            q = online(torch.from_numpy(obs).to(device))
            greedy = masked_argmax(q, torch.from_numpy(masks).to(device)).cpu().numpy()

        actions = np.zeros((k_members, n), dtype=np.int64)
        rewards = np.zeros((k_members, n), dtype=np.float32)
        next_obs = np.zeros_like(obs)
        next_masks = np.zeros_like(masks)
        dones = np.zeros((k_members, n), dtype=np.float32)
        for k, member in enumerate(members):
            for j, slot in enumerate(member.slots):
                episode = slot[0]
                if member.rng.random() < slot[2]:
                    action = member.rng.choice([a for a in range(A.N_ACTIONS) if masks[k, j, a]])
                else:
                    action = int(greedy[k, j])
                outcome = episode.apply(action)
                reward = step_reward(episode.graph, outcome, reward_cfg)
                done = episode.done()
                if done:
                    reward += terminal(episode.graph, episode)
                member.refresh(slot)
                actions[k, j] = action
                rewards[k, j] = reward
                next_obs[k, j] = slot[5]
                next_masks[k, j] = slot[6]
                dones[k, j] = float(done)
                slot[3] += reward
                slot[4] += 1
                if done:
                    if not member.finished:
                        graph = episode.graph
                        member.rewards.append(slot[3])
                        member.emitted.append(episode.state.n_emitted() / max(1, graph.n_nodes))
                        member.full.append(float(episode.full_coverage()))
                        member.lengths.append(slot[4])
                        member.stopped[episode.stopped_by or "none"] += 1
                    member.slots[j] = member.new_slot()
        replay.add(obs, actions, rewards, next_obs, next_masks, dones)
        transitions += n
        since_update += n
        t1 = time.perf_counter()
        env_seconds += t1 - t0

        if replay.size >= base.warmup:
            while since_update >= base.train_every:
                since_update -= base.train_every
                learner.update()
                updates += 1
                if updates % 50 == 0:
                    for member, value in zip(members, learner.td_out.tolist(), strict=True):
                        if not member.finished:
                            member.td.append(value)
                if updates % base.target_every == 0:
                    learner.sync_target()
            if device.type == "cuda":
                torch.cuda.synchronize()
        else:
            since_update = 0
        update_seconds += time.perf_counter() - t1

        for k, member in enumerate(members):
            if eval_every and not member.finished and len(member.rewards) >= member.next_eval:
                agent = DQNAgent(member.cfg, online.member(k))
                member.curve.append([len(member.rewards), greedy_reward(agent, graphs, reward_cfg)])
                member.next_eval += eval_every
            if member.finished and member.snapshot is None:
                member.snapshot = online.member(k)
                member.transitions = transitions  # type: ignore[attr-defined]
                member.updates = updates  # type: ignore[attr-defined]
                if progress:
                    print(
                        f"  member {k} ({member.cfg.inputs} lr={member.cfg.lr} "
                        f"gamma={member.cfg.gamma} seed={member.cfg.seed}) done: "
                        f"reward tail {statistics.fmean(member.rewards[-2000:]):.3f}  "
                        f"{time.perf_counter() - started_at:.0f}s",
                        file=sys.stderr,
                    )
        if progress and updates and updates % 20_000 < (n // base.train_every + 1):
            print(
                f"  {transitions} transitions/member, {updates} updates, "
                f"episodes {[len(m.rewards) for m in members]}, "
                f"{time.perf_counter() - started_at:.0f}s",
                file=sys.stderr,
            )

    seconds = time.perf_counter() - started_at
    agents: list[DQNAgent] = []
    for member in members:
        agent = DQNAgent(member.cfg, member.snapshot)
        agent.history = {
            "reward": member.rewards,
            "emitted_share": member.emitted,
            "full_coverage": member.full,
            "length": member.lengths,
            "td_abs": member.td,
            "greedy_curve": member.curve,
        }
        agent.report = convergence(agent)
        agent.report.update(
            transitions=member.transitions,  # type: ignore[attr-defined]
            updates=member.updates,  # type: ignore[attr-defined]
            stopped_by=dict(member.stopped),
        )
        agents.append(agent)
    timing = {
        "members": k_members,
        "seconds": round(seconds, 1),
        "env_seconds": round(env_seconds, 1),
        "update_seconds": round(update_seconds, 1),
        "transitions_per_member": transitions,
        "updates": updates,
        "transitions_per_second_all_members": round(transitions * k_members / seconds, 1),
        "terminal_cache_hit_rate": round(terminal.hits / max(1, terminal.calls), 4),
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            if device.type == "cuda"
            else None
        ),
    }
    for agent in agents:
        agent.report["timing"] = timing
    return agents


class Learner:
    """The GPU half of training: stacked online/target networks, replay, optimiser, one update.

    `update()` draws a `(K, batch)` index outside the step, then runs the step - captured as a
    CUDA graph after its first call when `capture` and the device is CUDA, eager otherwise. The
    two paths compute the same update; `tests/test_rl_dqn.py` pins that on identical batches.
    """

    def __init__(self, cfgs: Sequence[DQNConfig], device: torch.device, capture: bool = True):
        base = cfgs[0]
        self.device = device
        self.k = len(cfgs)
        self.batch = base.batch
        self.online = Ensemble(cfgs).to(device)
        self.target = Ensemble(cfgs).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = StackedAdam(
            list(self.online.parameters()), [c.lr for c in cfgs], base.grad_clip, device
        )
        self.gammas = torch.tensor([c.gamma for c in cfgs], dtype=torch.float32, device=device)
        self.gammas = self.gammas[:, None]
        self.replay = Replay(self.k, base.buffer, device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(base.seed * 7919 + self.k)
        self.idx = torch.zeros((self.k, base.batch), dtype=torch.long, device=device)
        self.td_out = torch.zeros(self.k, device=device)
        self.capture = capture and device.type == "cuda"
        self.runner = None
        self.graphs: list[Any] = []

    def step(self) -> None:
        """One Double-DQN step for every member. Static tensors only, so it can be captured."""
        b_obs, b_act, b_rew, b_nxt, b_nmask, b_done = self.replay.gather(self.idx)
        with torch.no_grad():
            choice = masked_argmax(self.online(b_nxt), b_nmask)
            bootstrap = self.target(b_nxt).gather(2, choice[..., None]).squeeze(2)
            y = b_rew + self.gammas * (1.0 - b_done) * bootstrap
        q_taken = self.online(b_obs).gather(2, b_act[..., None]).squeeze(2)
        per_member = nn.functional.smooth_l1_loss(q_taken, y, reduction="none").mean(1)
        per_member.sum().backward()
        self.optim.step()
        with torch.no_grad():
            self.td_out.copy_((q_taken - y).abs().mean(1))

    def update(self, idx: torch.Tensor | None = None) -> None:
        """Sample (or take) a batch index and apply one update."""
        if idx is None:
            self.idx.random_(0, self.replay.size, generator=self.generator)
        else:
            self.idx.copy_(idx)
        self.optim.advance()
        if self.runner is None:
            self.runner = _capture(self.step, self.optim, self.capture, self.graphs)
        else:
            self.runner()

    @torch.no_grad()
    def sync_target(self) -> None:
        """Hard target update, in place so a captured graph keeps pointing at the same tensors."""
        for dst, src in zip(self.target.parameters(), self.online.parameters(), strict=True):
            dst.copy_(src)


def _capture(update_step, optim: StackedAdam, capture: bool, holder: list[Any]):
    """Run the first update and return a callable that repeats it.

    On CUDA the step is captured as one `torch.cuda.CUDAGraph`, which replays every kernel of the
    forward, backward and optimiser without a Python round trip per op - for a network this small
    the launches, not the arithmetic, are the cost. The warm-up iterations the capture API needs
    and the capture pass itself are run and then **undone** (parameters and Adam moments restored),
    so the first replay is the one and only first update and capturing does not change learning.
    On CPU the step simply runs eagerly.
    """

    def eager() -> None:
        optim.zero_grad()
        update_step()

    if not capture:
        eager()
        return eager

    state = optim.params + optim.m + optim.v
    saved = [t.detach().clone() for t in state]

    def restore() -> None:
        torch.cuda.synchronize()
        with torch.no_grad():
            for live, copy in zip(state, saved, strict=True):
                live.copy_(copy)
        torch.cuda.synchronize()

    # The warm-up stream must not start reading `idx` and the step counters while the default
    # stream is still writing them. Without this barrier 2 of 6 concurrent 60k-episode runs died
    # with "illegal memory access" right after their first update; with it, none have.
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            eager()
    torch.cuda.current_stream().wait_stream(side)
    restore()
    optim.zero_grad()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update_step()
    # The capture pass is not trusted to be the first update: measured, the parameters after it
    # were unchanged while its TD read came from the warm-up weights, so the capture's own
    # effects are discarded too and the first real update is the first replay.
    restore()
    graph.replay()
    holder.append(graph)
    return graph.replay


def train(
    graphs: Sequence[DiagramGraph],
    cfg: DQNConfig | None = None,
    reward_cfg: RewardConfig = DEFAULT,
    progress: bool = False,
    eval_every: int = 0,
) -> DQNAgent:
    """One member. `train_many` with K = 1."""
    return train_many(graphs, [cfg or DQNConfig()], reward_cfg, progress, eval_every)[0]


def greedy_reward(agent: DQNAgent, graphs: Sequence[DiagramGraph], cfg=DEFAULT) -> float:
    """Mean terminal reward of greedy play over `graphs` - the training objective, cheaply."""
    values = []
    for episode in agent.rollout_many(graphs):
        values.append(terminal_reward(episode.graph, episode.state, episode.truncated(), cfg)[0])
    return round(statistics.fmean(values), 4)


def convergence(agent: DQNAgent, tail: float = 0.1) -> dict[str, Any]:
    """The same head/tail read as `qlearning.convergence`, over sampled updates for the TD."""
    td = agent.history.get("td_abs", [])
    reward = agent.history.get("reward", [])
    out: dict[str, Any] = {"episodes": len(reward)}
    if td:
        k = max(1, int(len(td) * tail))
        head, foot = statistics.fmean(td[:k]), statistics.fmean(td[-k:])
        out.update(
            td_head=round(head, 6),
            td_tail=round(foot, 6),
            td_tail_ratio=round(foot / head, 6) if head else None,
        )
    if reward:
        k = max(1, int(len(reward) * tail))
        out.update(
            reward_head=round(statistics.fmean(reward[:k]), 6),
            reward_tail=round(statistics.fmean(reward[-k:]), 6),
        )
    return out


def throughput(
    graphs: Sequence[DiagramGraph],
    grid: Sequence[tuple[int, int, int]] = (
        (1, 16, 256),
        (1, 32, 256),
        (1, 64, 256),
        (1, 128, 256),
        (1, 64, 1024),
        (3, 32, 256),
        (3, 64, 256),
        (3, 128, 256),
        (8, 64, 256),
        (8, 128, 256),
    ),
    episodes: int = 3000,
) -> list[dict[str, Any]]:
    """Measured cost of `(members, n_envs, batch)` at a fixed replay ratio of 64 samples/transition.

    `train_every` is `batch / 64`, so every cell draws the same number of samples per transition
    and the table compares cost for the same work. Warm-up is kept short and epsilon is held at
    0.3 so episodes are representative of mid-training length rather than of the first random
    few; the numbers are wall-clock on whatever else this machine is running.
    """
    rows = []
    for members, n_envs, batch in grid:
        cfgs = [
            DQNConfig(
                episodes=episodes,
                n_envs=n_envs,
                batch=batch,
                train_every=max(1, batch // 64),
                warmup=2000,
                epsilon=0.3,
                epsilon_min=0.3,
                seed=s,
            )
            for s in range(members)
        ]
        timing = train_many(graphs, cfgs)[0].report["timing"]
        total = timing["transitions_per_member"] * members
        rows.append(
            {
                "members": members,
                "n_envs": n_envs,
                "batch": batch,
                "transitions_per_s": round(total / timing["seconds"], 1),
                "samples_per_s": round(total * 64 / timing["seconds"], 1),
                "env_ms_per_transition": round(1e3 * timing["env_seconds"] / total, 4),
                "update_ms_per_update": round(
                    1e3 * timing["update_seconds"] / max(1, timing["updates"]), 4
                ),
                "terminal_cache_hit_rate": timing["terminal_cache_hit_rate"],
                "peak_vram_mb": timing["peak_vram_mb"],
            }
        )
    return rows


# ------------------------------------------------------------------------------------------
# the hypothesis, measured before training
# ------------------------------------------------------------------------------------------


def _gold_states(graphs: Sequence[DiagramGraph]):
    """Yield `(graph, episode, encoder, gold_action)` for every state gold play passes through."""
    for graph in graphs:
        episode = Episode(graph)
        encoder = StateEncoder(graph)
        for action in episode.gold_actions():
            if episode.done():
                break
            yield graph, episode, encoder, action
            episode.apply(action)


def aliasing(graphs: Sequence[DiagramGraph]) -> dict[str, Any]:
    """How often a representation puts two gold states with *different* gold actions on one key.

    This is the aliasing a control problem actually suffers - not how many raw states share a key
    (that is the compression the abstraction exists for), but how many share one while needing
    different actions. Measured for the tabular key, the 28-wide vector, and the vector plus mask.
    """
    reps = ("abstract_key", "features", "features_mask")
    owners: dict[str, dict[Any, set[int]]] = {r: defaultdict(set) for r in reps}
    seen: list[tuple[Any, Any, Any]] = []
    for graph, episode, encoder, action in _gold_states(graphs):
        feats = encoder.features(episode.state)
        keys = (
            key_index(abstract(graph, episode.state)),
            feats,
            feats + tuple(episode.mask()),
        )
        for rep, key in zip(reps, keys, strict=True):
            owners[rep][key].add(action)
        seen.append(keys)
    out: dict[str, Any] = {"gold_states": len(seen)}
    for position, rep in enumerate(reps):
        conflicted = sum(1 for keys in seen if len(owners[rep][keys[position]]) > 1)
        out[rep] = {
            "distinct_keys": len(owners[rep]),
            "gold_states_on_conflicting_key": conflicted,
            "rate": round(conflicted / max(1, len(seen)), 4),
        }
    return out


class TablePolicy:
    """The best policy expressible over the abstract key: the majority gold action per key."""

    def __init__(self, graphs: Sequence[DiagramGraph]) -> None:
        votes: dict[int, Counter] = defaultdict(Counter)
        for graph, episode, _, action in _gold_states(graphs):
            votes[key_index(abstract(graph, episode.state))][action] += 1
        self.ranked = {key: [a for a, _ in c.most_common()] for key, c in votes.items()}

    def rollout(self, graph: DiagramGraph) -> Episode:
        """Unseen keys terminate - the table has no evidence about them, so it stops."""
        episode = Episode(graph)
        while not episode.done():
            mask = episode.mask()
            ranked = self.ranked.get(key_index(abstract(graph, episode.state)), [])
            legal = [a for a in ranked if mask[a]]
            episode.apply(legal[0] if legal else A.TERMINATE)
        return episode


def imitate_mlp(
    graphs: Sequence[DiagramGraph], inputs: str = "features", epochs: int = 60, seed: int = 0
) -> DQNAgent:
    """A supervised `QNet` on gold (state, action) pairs: what the vector can express, no RL.

    Same network, same masked argmax as the DQN, so the ceiling is a ceiling for that network.
    """
    agent = DQNAgent(DQNConfig(inputs=inputs, seed=seed))
    rows, masks, targets = [], [], []
    for _, episode, encoder, action in _gold_states(graphs):
        rows.append(observe(encoder, episode, inputs))
        masks.append(episode.mask())
        targets.append(action)
    x = torch.tensor(rows, dtype=torch.float32, device=agent.device)
    m = torch.tensor(masks, dtype=torch.bool, device=agent.device)
    y = torch.tensor(targets, dtype=torch.long, device=agent.device)
    optim = torch.optim.Adam(agent.net.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=generator).to(agent.device)
        for start in range(0, len(y), 256):
            idx = perm[start : start + 256]
            logits = agent.net(x[idx]).masked_fill(~m[idx], -1e9)
            loss = nn.functional.cross_entropy(logits, y[idx])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    with torch.no_grad():
        agent.report["train_accuracy"] = round(
            float((masked_argmax(agent.net(x), m) == y).float().mean()), 4
        )
    return agent


def batched_player(agent: DQNAgent, graphs: Sequence[DiagramGraph]):
    """`play` calls one graph at a time; run every rollout in one batch and look each one up."""
    finished = {id(g): e for g, e in zip(graphs, agent.rollout_many(graphs), strict=True)}
    return lambda graph: finished[id(graph)] if id(graph) in finished else agent.rollout(graph)


def imitation_ceiling(
    train_graphs: Sequence[DiagramGraph],
    eval_graphs: Sequence[DiagramGraph],
    eval_diagrams: Sequence[dict],
    seed: int = 0,
) -> dict[str, Any]:
    """Fit each representation's best policy on `train_graphs` and play it on `eval_graphs`."""
    table = TablePolicy(train_graphs)
    mlp = imitate_mlp(train_graphs, "features", seed=seed)
    out = {
        "abstract_key_majority": play(table.rollout, eval_graphs, eval_diagrams),
        "features_mlp": play(batched_player(mlp, eval_graphs), eval_graphs, eval_diagrams),
    }
    out["abstract_key_majority"]["keys"] = len(table.ranked)
    out["features_mlp"]["train_accuracy"] = mlp.report["train_accuracy"]
    return out


def evaluate(
    agent: DQNAgent, graphs: Sequence[DiagramGraph], diagrams: Sequence[dict] | None = None
) -> dict[str, Any]:
    """`qlearning.play` with the DQN's greedy rollout - the harness every Phase 11 arm shares."""
    return play(batched_player(agent, graphs), graphs, diagrams)


def by_size(player, graphs: Sequence[DiagramGraph], diagrams: Sequence[dict]) -> dict[str, Any]:
    """`play` split by node count: the plan's "large graphs where tabular fails" as a table."""
    out: dict[str, dict[str, Any]] = {}
    for name, low, high in SIZE_BUCKETS:
        keep = [i for i, g in enumerate(graphs) if low <= g.n_nodes <= high]
        if keep:
            out[name] = play(player, [graphs[i] for i in keep], [diagrams[i] for i in keep])
    return out


# ------------------------------------------------------------------------------------------
# the study
# ------------------------------------------------------------------------------------------


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sample SD across seeds for the headline metrics."""
    keys = ("mean_terminal_reward", "full_coverage", "mean_emitted_share", "mean_edge_f1")
    out: dict[str, Any] = {}
    for key in keys:
        values = [r[key] for r in rows if key in r]
        if values:
            out[key] = {
                "mean": round(statistics.fmean(values), 4),
                "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "values": values,
            }
    return out


def split(
    diagrams: Sequence[dict], graphs: Sequence[DiagramGraph], seed: int = 0, share: float = 0.2
):
    """A fixed diagram-level held-out split, identical for every agent evaluated on it."""
    order = list(range(len(graphs)))
    random.Random(f"heldout/{seed}").shuffle(order)
    cut = int(len(order) * share)
    test, rest = sorted(order[:cut]), sorted(order[cut:])
    return (
        ([diagrams[i] for i in rest], [graphs[i] for i in rest]),
        ([diagrams[i] for i in test], [graphs[i] for i in test]),
    )


def _trail(values: Sequence[float], window: int = 1000, points: int = 200) -> list[list[float]]:
    """A trailing mean thinned to `points` samples, so the JSON carries a curve, not 60k floats."""
    if not values:
        return []
    csum = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64))])
    out = []
    for end in np.unique(np.linspace(1, len(values), num=min(points, len(values)), dtype=int)):
        start = max(0, end - window)
        out.append([int(end), round(float((csum[end] - csum[start]) / (end - start)), 4)])
    return out


def _sets(limit: int | None = None) -> dict[str, Any]:
    """The evaluation sets every stage uses, built identically in every process."""
    from src.rl.qlearning import training_set

    raw_all, graphs_all = training_set(limit, ambiguous=False)
    raw_amb, graphs_amb = training_set(limit, ambiguous=True)
    amb_ids = {g.diagram_id for g in graphs_amb}
    return {
        "raw_all": raw_all,
        "graphs_all": graphs_all,
        "raw_amb": raw_amb,
        "amb_graphs": [g for g in graphs_all if g.diagram_id in amb_ids],
        **dict(zip(("train", "test"), split(raw_all, graphs_all), strict=True)),
    }


def stage_prelim(limit: int | None = None) -> dict[str, Any]:
    """References, the aliasing count and both imitation ceilings. No RL."""
    from src.rl.qlearning import references

    sets = _sets(limit)
    (_, tr_g), (te_raw, te_g) = sets["train"], sets["test"]
    return {
        "train_diagrams": len(sets["graphs_all"]),
        "heldout_split": {"train": len(tr_g), "test": len(te_g)},
        "references": {
            "all": references(sets["graphs_all"], sets["raw_all"]),
            "ambiguous": references(sets["amb_graphs"], sets["raw_amb"]),
        },
        "aliasing": aliasing(sets["graphs_all"]),
        "imitation_ceiling": {
            "fit_and_play_all": imitation_ceiling(
                sets["graphs_all"], sets["graphs_all"], sets["raw_all"]
            ),
            "fit_train_play_heldout": imitation_ceiling(tr_g, te_g, te_raw),
        },
    }


def stage_sweep(
    base: DQNConfig, episodes: int, limit: int | None = None, progress: bool = False
) -> dict[str, Any]:
    """Every cell of the grid as one ensemble, ranked on greedy terminal reward - never on F1."""
    graphs = _sets(limit)["graphs_all"]
    cells = [
        replace(base, episodes=episodes, lr=lr, gamma=gamma, inputs=inputs)
        for inputs in INPUTS
        for lr in LRS
        for gamma in GAMMAS
    ]
    agents = train_many(graphs, cells, progress=progress)
    rows = []
    for cell, agent in zip(cells, agents, strict=True):
        rows.append(
            {
                "inputs": cell.inputs,
                "lr": cell.lr,
                "gamma": cell.gamma,
                "greedy_terminal_reward": greedy_reward(agent, graphs),
                "convergence": {k: v for k, v in agent.report.items() if k != "timing"},
            }
        )
    top = max(rows, key=lambda r: r["greedy_terminal_reward"])
    return {
        "sweep": rows,
        "sweep_timing": agents[0].report["timing"],
        "best_cell": {"inputs": top["inputs"], "lr": top["lr"], "gamma": top["gamma"]},
    }


def stage_member(
    cfg: DQNConfig,
    heldout: bool = False,
    limit: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """One seed: the DQN at `cfg` and 11.2.1's tabular agent, trained and scored the same way.

    `heldout=False` trains on all 993 and scores on the 993 and the ambiguous 684 - 11.2.1's
    protocol, so the like-for-like comparison. `heldout=True` trains both on the 80% split and
    scores both on the unseen 20%, the question a function approximator is supposed to answer
    better than a table.
    """
    from src.rl.qlearning import TrainConfig, gold_player
    from src.rl.qlearning import train as q_train

    sets = _sets(limit)
    if heldout:
        _, train_graphs = sets["train"]
        scopes = {"heldout": sets["test"]}
    else:
        train_graphs = sets["graphs_all"]
        scopes = {
            "all": (sets["raw_all"], sets["graphs_all"]),
            "ambiguous": (sets["raw_amb"], sets["amb_graphs"]),
        }
    started = time.perf_counter()
    every = 0 if heldout else max(1, cfg.episodes // 10)
    agent = train_many(train_graphs, [cfg], progress=progress, eval_every=every)[0]
    dqn_seconds = time.perf_counter() - started
    tag = "heldout-" if heldout else ""
    agent.save(CHECKPOINTS / f"p11-dqn-{tag}{cfg.inputs}-s{cfg.seed}.pt")
    tabulars: dict[str, Any] = {}
    tab_seconds: dict[str, float] = {}
    for name, tcfg in TABULAR_CELLS.items():
        started = time.perf_counter()
        tabulars[name] = q_train(
            train_graphs, TrainConfig(**tcfg, episodes=cfg.episodes, seed=cfg.seed)
        )
        tab_seconds[name] = round(time.perf_counter() - started, 1)
    tab = tabulars["tabular"]

    out: dict[str, Any] = {"seed": cfg.seed, "heldout": heldout, "config": asdict(cfg)}
    for scope, (raw, graphs) in scopes.items():
        players = {"dqn": batched_player(agent, graphs)}
        players.update({name: t.rollout for name, t in tabulars.items()})
        for name, player in players.items():
            out.setdefault(name, {})[scope] = play(player, graphs, raw)
        if scope in ("all", "heldout"):
            out["by_size"] = {
                "dqn": by_size(players["dqn"], graphs, raw),
                "tabular": by_size(tab.rollout, graphs, raw),
                "tabular_best": by_size(tabulars["tabular_best"].rollout, graphs, raw),
                "gold": by_size(gold_player, graphs, raw),
            }
            out["gold"] = play(gold_player, graphs, raw)
    out["curves"] = {
        "dqn_reward": _trail(agent.history["reward"]),
        "dqn_greedy": agent.history["greedy_curve"],
        "dqn_td": _trail(agent.history["td_abs"], window=20),
        "tabular_reward": _trail(tab.history["reward"]),
        "tabular_best_reward": _trail(tabulars["tabular_best"].history["reward"]),
    }
    out["convergence"] = {
        "dqn": {k: v for k, v in agent.report.items() if k != "timing"},
        **{name: t.report for name, t in tabulars.items()},
    }
    out["timing"] = {
        **agent.report["timing"],
        "dqn_wall_seconds": round(dqn_seconds, 1),
        "tabular_wall_seconds": tab_seconds,
    }
    return out


LEARNERS: tuple[str, ...] = ("dqn", "tabular", "tabular_best")


def merge(parts: dict[str, Any], members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Assemble `experiments/rl/dqn.json` from the stage outputs, with seed means and SDs."""
    result = dict(parts)
    main_runs = sorted((m for m in members if not m["heldout"]), key=lambda m: m["seed"])
    held_runs = sorted((m for m in members if m["heldout"]), key=lambda m: m["seed"])
    result["seeds"] = [m["seed"] for m in main_runs]
    result["config"] = main_runs[0]["config"]
    result["summary"] = {
        name: {
            scope: summarise([m[name][scope] for m in main_runs]) for scope in ("all", "ambiguous")
        }
        for name in LEARNERS
    }
    if held_runs:
        result["heldout"] = {
            "seeds": [m["seed"] for m in held_runs],
            "gold": held_runs[0]["gold"],
            "summary": {
                name: summarise([m[name]["heldout"] for m in held_runs]) for name in LEARNERS
            },
            "by_size_seed0": held_runs[0]["by_size"],
        }
    result["by_size_seed0"] = main_runs[0]["by_size"]
    result["curves"] = {f"s{m['seed']}": m["curves"] for m in main_runs}
    result["convergence"] = {f"s{m['seed']}": m["convergence"] for m in main_runs}
    result["timing"] = {f"s{m['seed']}": m["timing"] for m in main_runs}
    result["timing"].update({f"heldout_s{m['seed']}": m["timing"] for m in held_runs})
    result["per_seed"] = {
        f"{'heldout_' if m['heldout'] else ''}s{m['seed']}": {name: m[name] for name in LEARNERS}
        for m in main_runs + held_runs
    }
    return result


def run_quick(limit: int = 40, episodes: int = 300) -> dict[str, Any]:
    """Every stage at toy size in one process, for the tests."""
    base = DQNConfig(episodes=episodes, warmup=200, n_envs=8, batch=32, train_every=2)
    return merge(stage_prelim(limit), [stage_member(base, heldout=False, limit=limit)])


def plot(result: dict[str, Any], path=FIGURE):
    """Three panels: learning curves, edge F1 by graph size, and the headline bars with seed SD."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"dqn": "#d62728", "tabular": "#1f77b4", "tabular_best": "#17becf", "gold": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    for seed, block in result["curves"].items():
        for kind in ("dqn", "tabular", "tabular_best"):
            points = [p for p in block.get(f"{kind}_reward", []) if p[0] >= 1000]
            if points:
                xs, ys = zip(*points, strict=True)
                label = f"{kind} behaviour return" if seed == "s0" else None
                ax.plot(xs, ys, color=colours[kind], lw=1.0, alpha=0.8, label=label)
        greedy = block.get("dqn_greedy") or []
        if greedy:
            xs, ys = zip(*greedy, strict=True)
            label = "dqn greedy terminal reward" if seed == "s0" else None
            ax.plot(xs, ys, color=colours["dqn"], lw=0, marker="o", ms=3, label=label)
    gold = result["references"]["all"]["gold"]["mean_terminal_reward"]
    ax.axhline(gold, color=colours["gold"], ls="--", lw=1.0, label="gold greedy terminal reward")
    ax.set_title(
        f"training curves, trailing mean of 1,000 episodes ({len(result['seeds'])} seeds each)",
        fontsize=10,
    )
    ax.set_xlabel("episode", fontsize=9)
    ax.set_ylabel("reward", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    names = ("tabular", "tabular_best", "dqn", "gold")
    ax = axes[1]
    sizes = result["by_size_seed0"]
    buckets = [b[0] for b in SIZE_BUCKETS if b[0] in sizes["gold"]]
    width = 0.2
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    for offset, name in zip(offsets, names, strict=True):
        values = [sizes[name][b]["mean_edge_f1"] for b in buckets]
        ax.bar(
            [i + offset for i in range(len(buckets))],
            values,
            width,
            color=colours[name],
            label=name,
        )
    for i, b in enumerate(buckets):
        floor = sizes["gold"][b]["empty_policy_edge_f1"]
        ax.hlines(
            floor,
            i - 2 * width,
            i + 2 * width,
            color="black",
            lw=1.2,
            label="empty-policy floor" if i == 0 else None,
        )
    counts = [sizes["gold"][b]["diagrams"] for b in buckets]
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(
        [f"{b} nodes\n({c} diagrams)" for b, c in zip(buckets, counts, strict=True)], fontsize=8
    )
    ax.set_ylabel("edge F1", fontsize=9)
    ax.set_title("edge F1 by diagram size (seed 0, 993 labelled)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    labels = ["edge F1", "full coverage", "emitted share"]
    keys = ["mean_edge_f1", "full_coverage", "mean_emitted_share"]
    gold_row = result["references"]["all"]["gold"]
    width = 0.26
    for offset, name in zip((-width, 0.0, width), names[:3], strict=True):
        block = result["summary"][name]["all"]
        ax.bar(
            [i + offset for i in range(3)],
            [block[k]["mean"] for k in keys],
            width,
            yerr=[block[k]["sd"] for k in keys],
            capsize=4,
            color=colours[name],
            label=f"{name}, mean ± sd over {len(result['seeds'])} seeds",
        )
    ax.scatter(
        range(3),
        [gold_row[k] for k in keys],
        marker="*",
        s=180,
        color=colours["gold"],
        zorder=3,
        label="gold play in the action space",
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("greedy play, 993 labelled diagrams", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.5 DQN extension")
    ap.add_argument("--stage", choices=("prelim", "sweep", "member", "merge", "throughput"))
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--n-envs", type=int, default=DQNConfig.n_envs)
    ap.add_argument("--inputs", default=DQNConfig.inputs)
    ap.add_argument("--lr", type=float, default=DQNConfig.lr)
    ap.add_argument("--gamma", type=float, default=DQNConfig.gamma)
    ap.add_argument("--out", default=None, help="where a stage writes its JSON part")
    ap.add_argument("--parts", nargs="*", default=[], help="stage JSON files for --stage merge")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        print(json.dumps(run_quick()["summary"], indent=2))
        return 0
    from pathlib import Path

    from src.rl.qlearning import training_set

    base = DQNConfig(
        episodes=args.episodes,
        n_envs=args.n_envs,
        inputs=args.inputs,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
    )
    if args.stage == "throughput":
        _, graphs = training_set(args.limit, ambiguous=False)
        out: dict[str, Any] = {"throughput": throughput(graphs)}
    elif args.stage == "prelim":
        out = stage_prelim(args.limit)
    elif args.stage == "sweep":
        out = stage_sweep(base, args.episodes, args.limit, progress=True)
    elif args.stage == "member":
        out = stage_member(base, heldout=args.heldout, limit=args.limit, progress=True)
    elif args.stage == "merge":
        parts: dict[str, Any] = {}
        members = []
        for name in args.parts:
            blob = json.loads(Path(name).read_text(encoding="utf-8"))
            if "seed" in blob and "heldout" in blob:
                members.append(blob)
            else:
                parts.update(blob)
        out = merge(parts, members)
        RUNS.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        out["figure"] = str(plot(out).relative_to(ROOT))
    else:
        ap.error("choose --stage or --quick")
        return 2
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for key in ("curves", "per_seed", "references"):
        out.pop(key, None)
    print(json.dumps(out, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
