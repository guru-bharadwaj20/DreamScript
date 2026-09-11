"""Phase 11.2.1 - tabular Q-learning over 11.1.6's abstract state, with the alpha/gamma sweep
and the epsilon decay, plus the shared training loop 11.2.2-11.2.6 reuse.

    python -m src.rl.qlearning              # train at the swept best, evaluate, write the JSON
    python -m src.rl.qlearning --sweep      # the alpha x gamma grid
    python -m src.rl.qlearning --quick      # a short run, for the tests

## What "converged Q-table" can and cannot mean here

The plan's definition of done is "converged Q-table". Two facts about this environment decide
what that is allowed to claim.

**The ceiling is 63.06%, not 100%.** 11.1.4 measured it: the action space has no jump, so a
traversal cannot cross into a second connected component, and 44.1% of the labelled corpus has
more than one. Gold play - a replay of 7.3.3's DFS in this action space - reaches full coverage
on 63.06% of diagrams and that is the ceiling *any* policy in this action space can reach. Every
coverage number below is reported against 63.06%. A number near 100% would be a bug, not a
result.

**The state is abstract, so the optimal policy is not representable.** 11.1.6's key is a 9-factor
product over 128,000 cells, of which 1,766 are ever reached, and 25.2% of raw states share a key
with another raw state. The Q-table is therefore learning a policy over an aliased state, and
"converged" means the *table* stops moving, not that the policy is optimal. Both are reported:
the mean |TD error| over the last 10% of episodes, and the share of reached keys whose greedy
action stopped changing.

## The training loop

`train` drives `src.rl.episode.Episode` directly rather than `DiagramTraversalEnv`. The env is a
thin wrapper, but it calls `StateEncoder.features` on every `step` to build a 28-vector no
tabular agent ever reads, and it recomputes the abstract key in `_info` as well. Skipping both is
worth 3.4x on wall clock and changes no number - `test_rl_qlearning.py` asserts the two paths
produce identical trajectories under a fixed policy. 11.2.5's DQN uses the env, because it is the
consumer the 28-vector exists for.

The update is off-policy, over the legal actions only:

    Q[s,a] <- Q[s,a] + alpha * (r + gamma * max_{a' legal in s'} Q[s',a'] - Q[s,a])

Masking the bootstrap matters more than it usually does: `terminate` is legal in every state
(11.1.2), so an unmasked max would let the value of an illegal `follow-edge-4` leak backwards
through states where that edge does not exist, and those states are the majority - the mask
leaves 3.29 legal actions of 9.

## Results

Trained on the 684-diagram ambiguous set (11.2.7's definition), evaluated greedily on the same
set. The measured numbers live in `experiments/rl/qlearning.json`; the summary the plan row
quotes is reproduced by `python -m src.rl.qlearning`.

The honest headline is in `evaluate`: this agent is scored by the same `baselines.score` edge F1
the four non-learned arms are scored by, on the same set, so the comparison in 11.2.7 is not a
new metric invented for the learner.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.rl import actions as A
from src.rl.abstraction import BOUND, abstract, key_index
from src.rl.episode import Episode
from src.rl.reward import DEFAULT, RewardConfig, step_reward, terminal_reward
from src.rl.state import DiagramGraph
from src.utils.config import ROOT

RUNS = ROOT / "experiments" / "rl"
OUT = RUNS / "qlearning.json"

#: 11.1.4's measured ceiling: the share of diagrams on which *gold play* reaches full coverage.
#: There is no jump action, so a traversal cannot enter a second connected component. Every
#: coverage figure in Phase 11.2 is reported against this number and never against 100%.
COVERAGE_CEILING = 0.6306

#: The alpha x gamma grid. Coarse on purpose: the sweep is here to show the answer does not hinge
#: on the hyper-parameters, not to squeeze the last thousandth out of one.
ALPHAS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
GAMMAS: tuple[float, ...] = (0.9, 0.95, 0.99, 1.0)


# ------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Everything the loop reads. One object so 11.2.2-11.2.6 vary one field and share the rest."""

    algo: str = "q"  #: "q" (off-policy, max) or "sarsa" (on-policy, the action actually taken)
    alpha: float = 0.2
    gamma: float = 0.95
    episodes: int = 20_000
    #: epsilon decays geometrically from `epsilon` to `epsilon_min` across the run.
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    explore: str = "egreedy"  #: "egreedy" | "softmax" | "ucb" - see `src.rl.exploration`
    temperature: float = 1.0
    ucb_c: float = 1.0
    seed: int = 0
    #: 11.2.4. A callable `(graph, state) -> float` used as a potential; the loop adds
    #: `gamma * phi(s') - phi(s)` to the reward. Potential-based, so the optimal policy is
    #: provably unchanged - which is the point 11.2.4 has to check rather than assume.
    potential: Callable[[DiagramGraph, Any], float] | None = field(default=None, repr=False)
    #: 11.2.4's *other* hook: an unconstrained per-step bonus `(graph, outcome, state) -> float`,
    #: added to the reward with no potential structure and therefore **no invariance guarantee**.
    #: It exists so that 11.2.4 can measure a shaping exploit rather than argue one is impossible;
    #: nothing else in Phase 11 sets it.
    step_bonus: Callable[[DiagramGraph, Any, Any], float] | None = field(default=None, repr=False)
    #: 11.2.8's hook: called as `on_episode(agent, index)` after each episode. Diagnostics need
    #: the table *during* training (policy churn is a difference between two snapshots), and
    #: chunking the run instead would restart the epsilon schedule at every chunk boundary and
    #: quietly change the thing being measured.
    on_episode: Callable[[Any, int], None] | None = field(default=None, repr=False)
    reward: RewardConfig = field(default_factory=lambda: DEFAULT, repr=False)

    def to_dict(self) -> dict[str, Any]:
        out = {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
            if k not in {"potential", "reward", "step_bonus", "on_episode"}
        }
        for hook in ("potential", "step_bonus", "on_episode"):
            value = getattr(self, hook)
            out[hook] = None if value is None else getattr(value, "__name__", "custom")
        return out


def epsilon_at(cfg: TrainConfig, episode: int) -> float:
    """Geometric decay from `epsilon` to `epsilon_min` over `episodes`. Never below the floor."""
    if cfg.episodes <= 1 or cfg.epsilon <= cfg.epsilon_min:
        return cfg.epsilon_min
    ratio = (cfg.epsilon_min / cfg.epsilon) ** (episode / max(1, cfg.episodes - 1))
    return max(cfg.epsilon_min, cfg.epsilon * ratio)


# ------------------------------------------------------------------------------------------
# the agent
# ------------------------------------------------------------------------------------------


def completed_order(graph: DiagramGraph, episode: Episode) -> list[str]:
    """A finished episode's emission order as node ids, completed to a permutation.

    11.2.7's arms must return every node exactly once, and a policy in this action space cannot
    reach a second connected component - so a rollout that stops short of full coverage is not a
    permutation and cannot enter the table at all. Something has to fill the tail, and **which
    filler is chosen decides the result**, so the choice is stated here rather than buried.

    The tail is **IR document order** - the order the nodes happen to appear in the file.
    `gold_order` was tried first and rejected on measurement: it is 7.3.3's DFS, the very
    traversal the `dfs` arm scores 0.6247 with, so a policy that emitted *nothing* inherited an
    edge F1 of 0.60 and the table would have been scoring the filler rather than the policy.
    Document order carries no flow information, which is what a filler must not carry. `play`
    measures the empty-policy control explicitly (`empty_policy_edge_f1`) and every learned
    number is to be read against it, not against 0.
    """
    emitted = list(episode.state.emit_sequence)
    seen = set(emitted)
    tail = [i for i in range(graph.n_nodes) if i not in seen]
    return [graph.node_ids[i] for i in emitted + tail]


class TabularAgent:
    """A Q-table over `src.rl.abstraction`'s bounded key space, and the policies that read it.

    The table is dense `float32` at the full bound - 128,000 x 9 x 4 B = 4.6 MB - rather than a
    dict. 11.1.6 measured only 1,766 keys reached, so a dict would be 70x smaller, but the array
    makes `argmax` over a masked row a single vectorised call and removes a branch from the
    innermost loop. 4.6 MB is not a constraint; the inner loop is.
    """

    def __init__(self, cfg: TrainConfig | None = None) -> None:
        self.cfg = cfg or TrainConfig()
        self.q = np.zeros((BOUND, A.N_ACTIONS), dtype=np.float32)
        self.counts = np.zeros((BOUND, A.N_ACTIONS), dtype=np.int32)
        self.history: dict[str, list[float]] = {}
        self.report: dict[str, Any] = {}

    # -- action selection -------------------------------------------------------------------

    def greedy(self, key: int, mask: Sequence[bool], rng: random.Random) -> int:
        """Argmax over the legal actions, ties broken at random rather than by index.

        Deterministic tie-breaking would make `terminate` (action 8) lose every tie against a
        follow-edge, or win every one, depending on the direction - and on an untouched row every
        action ties at 0.0, so the tie rule *is* the initial policy.
        """
        legal = [a for a in range(A.N_ACTIONS) if mask[a]]
        row = self.q[key]
        best = max(row[a] for a in legal)
        top = [a for a in legal if row[a] >= best - 1e-12]
        return top[0] if len(top) == 1 else rng.choice(top)

    def select(self, key: int, mask: Sequence[bool], rng: random.Random, eps: float) -> int:
        """Behaviour policy. `explore` picks the rule; `src.rl.exploration` owns the other two."""
        if self.cfg.explore == "egreedy":
            legal = [a for a in range(A.N_ACTIONS) if mask[a]]
            if rng.random() < eps:
                return rng.choice(legal)
            return self.greedy(key, mask, rng)
        from src.rl.exploration import select as explore_select

        return explore_select(self, key, mask, rng, eps)

    # -- using the learnt table -------------------------------------------------------------

    def rollout(self, graph: DiagramGraph, seed: int = 0, cap: int | None = None) -> Episode:
        """Play one diagram greedily to the end and return the finished `Episode`."""
        rng = random.Random(seed)
        episode = Episode(graph, cap=cap)
        while not episode.done():
            key = key_index(abstract(graph, episode.state))
            episode.apply(self.greedy(key, episode.mask(), rng))
        return episode

    def order(self, graph: DiagramGraph, seed: int = 0) -> list[str]:
        """The greedy rollout's emission order, completed to a permutation by `completed_order`."""
        return completed_order(graph, self.rollout(graph, seed=seed))

    def arm(self, diagram: dict) -> list[str]:
        """`src.rl.baselines.register_arm`-shaped adapter: raw IR dict -> emission order."""
        return self.order(DiagramGraph.from_ir(diagram))


# ------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------


def train(
    graphs: Sequence[DiagramGraph],
    cfg: TrainConfig | None = None,
    agent: TabularAgent | None = None,
    progress: bool = False,
) -> TabularAgent:
    """Run `cfg.episodes` episodes over `graphs` and return the trained agent.

    Handles both algorithms; `cfg.algo == "sarsa"` bootstraps off the action the behaviour policy
    actually chose next, which is the whole of the 11.2.2 difference and is why it lives here
    rather than in a duplicated loop.
    """
    cfg = cfg or (agent.cfg if agent is not None else TrainConfig())
    agent = agent or TabularAgent(cfg)
    agent.cfg = cfg
    rng = random.Random(cfg.seed)
    if not graphs:
        raise ValueError("train needs at least one diagram")

    rewards: list[float] = []
    td_errors: list[float] = []
    coverage: list[float] = []
    emitted: list[float] = []
    lengths: list[float] = []
    epsilons: list[float] = []
    truncations: list[float] = []
    q = agent.q

    for ep in range(cfg.episodes):
        eps = epsilon_at(cfg, ep)
        graph = graphs[rng.randrange(len(graphs))]
        episode = Episode(graph)
        phi_prev = 0.0 if cfg.potential is None else cfg.potential(graph, episode.state)
        key = key_index(abstract(graph, episode.state))
        mask = episode.mask()
        action = agent.select(key, mask, rng, eps)
        total = 0.0
        abs_td = 0.0
        steps = 0

        while True:
            outcome = episode.apply(action)
            reward = step_reward(graph, outcome, cfg.reward)
            if cfg.step_bonus is not None:
                reward += cfg.step_bonus(graph, outcome, episode.state)
            terminated = episode.terminated()
            truncated = episode.truncated() and not terminated
            done = terminated or truncated
            if done:
                final, _ = terminal_reward(graph, episode.state, truncated, cfg.reward)
                reward = reward + final
            if cfg.potential is not None:
                phi_next = 0.0 if done else cfg.potential(graph, episode.state)
                reward = reward + cfg.gamma * phi_next - phi_prev
                phi_prev = phi_next
            total += reward
            steps += 1

            next_key = -1
            next_action = -1
            if done:
                target = reward
            else:
                next_key = key_index(abstract(graph, episode.state))
                next_mask = episode.mask()
                next_action = agent.select(next_key, next_mask, rng, eps)
                if cfg.algo == "sarsa":
                    bootstrap = float(q[next_key, next_action])
                else:
                    legal = [a for a in range(A.N_ACTIONS) if next_mask[a]]
                    bootstrap = float(max(q[next_key, a] for a in legal))
                target = reward + cfg.gamma * bootstrap

            delta = target - float(q[key, action])
            q[key, action] += cfg.alpha * delta
            agent.counts[key, action] += 1
            abs_td += abs(delta)

            if done:
                break
            key, action = next_key, next_action

        if cfg.on_episode is not None:
            cfg.on_episode(agent, ep)
        rewards.append(total)
        td_errors.append(abs_td / max(1, steps))
        coverage.append(1.0 if episode.full_coverage() else 0.0)
        emitted.append(episode.state.n_emitted() / max(1, graph.n_nodes))
        lengths.append(steps)
        epsilons.append(eps)
        truncations.append(1.0 if episode.truncated() else 0.0)
        if progress and (ep + 1) % 2000 == 0:
            print(
                f"  episode {ep + 1}/{cfg.episodes}  eps={eps:.3f}  "
                f"reward={statistics.fmean(rewards[-2000:]):.3f}  "
                f"|td|={statistics.fmean(td_errors[-2000:]):.4f}",
                file=sys.stderr,
            )

    agent.history = {
        "reward": rewards,
        "td_error": td_errors,
        "full_coverage": coverage,
        "emitted_share": emitted,
        "length": lengths,
        "epsilon": epsilons,
        #: 1.0 when the behaviour policy ran into the step cap - 11.2.2's cliff, during training.
        "truncated": truncations,
    }
    agent.report = convergence(agent)
    return agent


def convergence(agent: TabularAgent, tail: float = 0.1) -> dict[str, Any]:
    """Did the *table* stop moving? Two numbers, both about the table and neither about optimality.

    `td_tail_ratio` is the mean |TD error| over the last `tail` of the run against the first, and
    `reached_keys` is how much of 11.1.6's 128,000-cell bound was ever touched. A ratio near 1
    means the table is still moving; a small number of reached keys is the expected result and is
    the reason the bound is a guarantee rather than a size estimate.
    """
    td = agent.history.get("td_error", [])
    reward = agent.history.get("reward", [])
    n = len(td)
    if n == 0:
        return {}
    k = max(1, int(n * tail))
    head, foot = statistics.fmean(td[:k]), statistics.fmean(td[-k:])
    return {
        "episodes": n,
        "td_head": round(head, 6),
        "td_tail": round(foot, 6),
        "td_tail_ratio": round(foot / head, 6) if head else None,
        "reward_head": round(statistics.fmean(reward[:k]), 6),
        "reward_tail": round(statistics.fmean(reward[-k:]), 6),
        "reached_keys": int((agent.counts.sum(axis=1) > 0).sum()),
        "bound": BOUND,
        "updates": int(agent.counts.sum()),
    }


# ------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------


def gold_coverage(graphs: Sequence[DiagramGraph]) -> float:
    """The ceiling, recomputed on *this* set rather than quoted from 11.1.4's corpus figure."""
    hits = 0
    for graph in graphs:
        episode = Episode(graph)
        for action in episode.gold_actions():
            if episode.done():
                break
            episode.apply(action)
        hits += int(episode.full_coverage())
    return hits / max(1, len(graphs))


def play(
    player: Callable[[DiagramGraph], Episode],
    graphs: Sequence[DiagramGraph],
    diagrams: Sequence[dict] | None = None,
    cfg: RewardConfig = DEFAULT,
) -> dict[str, Any]:
    """Score any policy over `graphs`. Reward, coverage against the ceiling, and 11.2.7's edge F1.

    `player` is anything that finishes an `Episode` on one graph: the agent's greedy rollout, gold
    play, or a random walk. It is one function so the learner and its references are never scored
    by two different harnesses - which is the failure mode that makes a learned result look good.

    `diagrams` are the raw IR dicts matching `graphs`; when given, the same `baselines.score` that
    scores topological / DFS / BFS / reading is applied to the completed order.
    """
    rewards: list[float] = []
    full: list[float] = []
    emitted: list[float] = []
    lengths: list[float] = []
    stopped: dict[str, int] = {}
    f1: list[float] = []
    f1_empty: list[float] = []

    by_id = {str(d.get("id")): d for d in (diagrams or [])}

    for graph in graphs:
        episode = player(graph)
        _, breakdown = terminal_reward(graph, episode.state, episode.truncated(), cfg)
        rewards.append(breakdown["terminal_reward"])
        full.append(1.0 if episode.full_coverage() else 0.0)
        emitted.append(episode.state.n_emitted() / max(1, graph.n_nodes))
        lengths.append(episode.state.steps)
        stopped[episode.stopped_by or "none"] = stopped.get(episode.stopped_by or "none", 0) + 1
        raw = by_id.get(graph.diagram_id)
        if raw is not None:
            from src.rl.baselines import score

            f1.append(score(raw, completed_order(graph, episode))["edge_f1"])
            f1_empty.append(score(raw, list(graph.node_ids))["edge_f1"])

    ceiling = gold_coverage(graphs)
    mean_full = statistics.fmean(full)
    out = {
        "diagrams": len(graphs),
        "mean_terminal_reward": round(statistics.fmean(rewards), 4),
        "full_coverage": round(mean_full, 4),
        "gold_full_coverage_ceiling": round(ceiling, 4),
        "coverage_vs_ceiling": round(mean_full / ceiling, 4) if ceiling else None,
        "mean_emitted_share": round(statistics.fmean(emitted), 4),
        "mean_episode_length": round(statistics.fmean(lengths), 2),
        "stopped_by": stopped,
    }
    if f1:
        out["mean_edge_f1"] = round(statistics.fmean(f1), 4)
        #: what a policy that emits nothing scores - the floor every learned F1 is read against.
        out["empty_policy_edge_f1"] = round(statistics.fmean(f1_empty), 4)
        out["scored_diagrams"] = len(f1)
    return out


def evaluate(
    agent: TabularAgent,
    graphs: Sequence[DiagramGraph],
    diagrams: Sequence[dict] | None = None,
    cfg: RewardConfig = DEFAULT,
) -> dict[str, Any]:
    """`play` with the agent's greedy rollout as the player."""
    return play(lambda g: agent.rollout(g), graphs, diagrams, cfg)


def gold_player(graph: DiagramGraph) -> Episode:
    """Gold play in this action space: the reference, and the *real* ceiling on the F1 metric."""
    episode = Episode(graph)
    for action in episode.gold_actions():
        if episode.done():
            break
        episode.apply(action)
    return episode


def random_player(graph: DiagramGraph, seed: int = 0) -> Episode:
    """Uniform over the legal actions. 11.1.4 measured it at 2.68% coverage; it is the floor."""
    rng = random.Random(seed)
    episode = Episode(graph)
    while not episode.done():
        episode.apply(rng.choice(episode.legal_actions()))
    return episode


def references(
    graphs: Sequence[DiagramGraph], diagrams: Sequence[dict] | None = None
) -> dict[str, Any]:
    """Gold and random through the same harness, so the learned row has something to sit between.

    **The gold row is the number that reframes 11.2.7.** The `dfs` arm scores 0.6247 there, but it
    is a whole-graph permutation: it walks into a second connected component because it never
    passes through this action space. Gold play *inside* the action space cannot, and its edge F1
    is the honest upper bound on what any policy trained here can reach.
    """
    return {
        "gold": play(gold_player, graphs, diagrams),
        "random": play(random_player, graphs, diagrams),
    }


def sweep(
    graphs: Sequence[DiagramGraph],
    diagrams: Sequence[dict] | None = None,
    alphas: Sequence[float] = ALPHAS,
    gammas: Sequence[float] = GAMMAS,
    episodes: int = 4000,
    seed: int = 0,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """The alpha x gamma grid, each cell trained from scratch on the same seed.

    Cells are not equally cheap: a badly tuned one wanders to the `4n + 8` step cap on every
    episode, so it costs several times what a converging cell costs. That is itself a signal and
    the wall-clock is reported per cell rather than hidden behind a total.
    """
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for gamma in gammas:
            started = time.perf_counter()
            cfg = TrainConfig(alpha=alpha, gamma=gamma, episodes=episodes, seed=seed)
            agent = train(graphs, cfg)
            row = {"alpha": alpha, "gamma": gamma}
            row.update(evaluate(agent, graphs, diagrams))
            row["convergence"] = agent.report
            row["seconds"] = round(time.perf_counter() - started, 1)
            rows.append(row)
            if progress:
                print(
                    f"  cell alpha={alpha} gamma={gamma}  "
                    f"reward={row['mean_terminal_reward']}  "
                    f"coverage={row['full_coverage']}  {row['seconds']}s",
                    file=sys.stderr,
                )
    return rows


# ------------------------------------------------------------------------------------------
# the corpus, and the entry point
# ------------------------------------------------------------------------------------------


def training_set(
    limit: int | None = None, ambiguous: bool = True
) -> tuple[list[dict], list[DiagramGraph]]:
    """11.2.7's ambiguous set (684 of 993 labelled diagrams) as raw IR and as compiled graphs."""
    from src.parse.sequences import labelled_diagrams
    from src.rl.baselines import ambiguous_set

    raw = labelled_diagrams(limit)
    if ambiguous:
        raw = ambiguous_set(raw)
    return raw, [DiagramGraph.from_ir(d) for d in raw]


def run(
    episodes: int = 20_000,
    limit: int | None = None,
    write: bool = False,
    do_sweep: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Sweep, train at the best cell, evaluate against gold and random. The 11.2.1 measurement.

    Trains on all 993 labelled diagrams and evaluates on two sets: the same 993, and 11.2.7's
    684-diagram ambiguous subset. Both are reported because they answer different questions - the
    ambiguous set is where 11.2.7's table lives, and it is also the hardest slice of the corpus
    (610 of its 684 pages are multi-component hdbpmn), so an agent judged only there is judged
    where the action space itself is weakest.

    The sweep is ranked on `mean_terminal_reward` and not on edge F1: the reward is what the
    agent optimises, and picking the cell by the transfer metric would be selecting on the test.
    """
    raw_all, graphs_all = training_set(limit, ambiguous=False)
    raw_amb, graphs_amb = training_set(limit, ambiguous=True)
    result: dict[str, Any] = {
        "train_diagrams": len(graphs_all),
        "coverage_ceiling_corpus": COVERAGE_CEILING,
        "references": {
            "all": references(graphs_all, raw_all),
            "ambiguous": references(graphs_amb, raw_amb),
        },
    }
    best = TrainConfig(episodes=episodes, seed=seed)
    if do_sweep:
        rows = sweep(
            graphs_all, raw_all, episodes=max(2000, episodes // 5), seed=seed, progress=write
        )
        result["sweep"] = rows
        top = max(rows, key=lambda r: r["mean_terminal_reward"])
        best = TrainConfig(alpha=top["alpha"], gamma=top["gamma"], episodes=episodes, seed=seed)
        result["best_cell"] = {"alpha": top["alpha"], "gamma": top["gamma"]}
    agent = train(graphs_all, best, progress=write)
    result["config"] = best.to_dict()
    result["convergence"] = agent.report
    result["evaluation"] = {
        "all": evaluate(agent, graphs_all, raw_all),
        "ambiguous": evaluate(agent, graphs_amb, raw_amb),
    }
    result["epsilon_schedule"] = {
        "start": best.epsilon,
        "floor": best.epsilon_min,
        "at_10pct": round(epsilon_at(best, best.episodes // 10), 4),
        "at_50pct": round(epsilon_at(best, best.episodes // 2), 4),
    }
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["path"] = str(OUT.relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.1 tabular Q-learning")
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="only the alpha x gamma grid")
    ap.add_argument("--quick", action="store_true", help="600 episodes, no sweep")
    args = ap.parse_args(argv)
    if args.quick:
        print(json.dumps(run(600, args.limit, do_sweep=False, seed=args.seed), indent=2))
        return 0
    if args.sweep:
        raw, graphs = training_set(args.limit)
        print(json.dumps(sweep(graphs, raw, episodes=args.episodes, seed=args.seed), indent=2))
        return 0
    print(json.dumps(run(args.episodes, args.limit, write=args.write, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
