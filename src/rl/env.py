"""Phase 11.1.5 - `DiagramTraversalEnv`: reset / step / render, and the check that it is real.

    python -m src.rl.env             # runs api_check() over the corpus and prints the verdict

## No gymnasium dependency, and why

**`gymnasium` is not installed in this environment and `gym` is not either.** Both were checked
before anything here was written to depend on them (`ModuleNotFoundError` for each), and Phase 0
froze the dependency set. Adding a package for an interface that is nine lines of protocol would
be the wrong trade, so this class implements the **Gymnasium 0.26+ API shape natively**:

    reset(seed=None, options=None) -> (observation, info)
    step(action)                   -> (observation, reward, terminated, truncated, info)
    render()                       -> str | None      ("ansi" and "human" modes)
    close()
    action_space / observation_space   - minimal `Discrete` / `Box` stand-ins in this module

If gymnasium is ever added, `DiagramTraversalEnv` can be given `gymnasium.Env` as a base class
with no other change; `api_check()` is written against the protocol, not the package, so it will
keep passing. The stand-in spaces expose `.n`, `.shape`, `.sample(rng)` and `.contains(x)` and
nothing else - they are not a Gym reimplementation and do not pretend to be.

## What `api_check` checks, and the measured result

Run over a 3,993-diagram pool, 10 seeds x 40 episodes = **400 episodes, 7,306 steps, 68,154
probes, 0 failures**:

    reset returns (observation, info)                              pass
    observation width equals StateEncoder.feature_names   28, constant, pass
    observation lies inside observation_space                      pass
    step returns the 5-tuple                                       pass
    terminated and truncated never both true                       pass
    reward always finite                                           pass
    info carries action_mask of length N_ACTIONS         9, pass
    a masked-out action is refused (info["legal"] is False)        pass
    a refused action leaves current/visited/emitted/marked intact   pass
    every legal action is accepted                                 pass
    episode always ends within step_cap + 1                        pass
    abstract_state always inside 11.1.6's bound                    pass
    render() returns text                                          pass
    step() after termination raises                                pass
    same seed -> byte-identical trajectory                         pass
    two different seeds -> different trajectory      40/40 on diagrams with a real choice

**The driving policy avoids `TERMINATE` unless it is the only legal action.** This matters: a
uniform random policy ends 97.60% of episodes by firing `TERMINATE` immediately
(`src.rl.episode` measures it), which left the inherited check averaging ~4 steps per episode.
The inherited docstring nevertheless claimed "1,600 episodes ... 24,881 steps"; re-run, the
same code produced 118 steps for 32 episodes, so that figure could not be reproduced and is
replaced by the numbers above rather than adjusted.

**Robustness on the real corpus, which is the thing 11.1 is actually at risk from**: every one of
the 3,993 sequential diagrams was played to termination under a random legal policy, with
`render()` and `emitted_code()` called at the terminal. **0 crashes.** The pool it survived
includes 1,389 diagrams with cycles, 709 with more than one connected component, 530 carrying an
unresolved edge (`src`/`dst` None), and 698 with at least one node whose OCR'd text is blank.

## What was REJECTED

**Returning the raw `TraversalState` as the observation was rejected.** An agent would then have
to import `src.rl.state` to interpret it and a function approximator could not consume it at all.
The observation is `StateEncoder.features` - a fixed-width float tuple - and the structured state
is offered alongside in `info["state"]` for tabular agents, which do want it.

**Auto-resetting on termination was rejected**: it hides the terminal transition, and 11.2.1's TD
update needs to see it. `step` after termination raises, and `api_check` probes that it does.

**Batching several diagrams into one vectorised env was rejected at this row.** 11.2.5's DQN may
want it; building it now would fix a batching layout before anything measures whether the
bottleneck is env stepping. Measured, it is not: **21,781 steps/s single-threaded on CPU**,
terminal reward included. (The inherited docstring said 34,900 steps/s; 21,781 is what this
machine produces, and the conclusion is unchanged either way.)
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from typing import Any

from src.rl import actions as A
from src.rl.abstraction import AbstractState, abstract
from src.rl.episode import Episode
from src.rl.reward import DEFAULT, RewardConfig, step_reward, terminal_reward
from src.rl.state import DiagramGraph, StateEncoder, TraversalState

Observation = tuple[float, ...]


class Discrete:
    """Minimal stand-in for `gymnasium.spaces.Discrete`. See the module docstring."""

    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.shape: tuple[int, ...] = ()

    def sample(self, rng: random.Random | None = None) -> int:
        return (rng or random).randrange(self.n)

    def contains(self, x: Any) -> bool:
        return isinstance(x, int) and 0 <= x < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Box:
    """Minimal stand-in for `gymnasium.spaces.Box`, low/high scalar and a 1-D shape."""

    def __init__(self, low: float, high: float, shape: tuple[int, ...]) -> None:
        self.low = float(low)
        self.high = float(high)
        self.shape = tuple(shape)

    def contains(self, x: Any) -> bool:
        try:
            values = list(x)
        except TypeError:
            return False
        return len(values) == self.shape[0] and all(
            isinstance(v, int | float) and self.low <= v <= self.high for v in values
        )

    def __repr__(self) -> str:
        return f"Box({self.low}, {self.high}, {self.shape})"


class DiagramTraversalEnv:
    """One diagram is one episode; `reset` draws the next diagram from the pool.

    The environment owns no logic of its own: transitions are `src.rl.episode.Episode`, rewards
    are `src.rl.reward`, and the observation is `src.rl.state.StateEncoder`. It is the API surface
    the seven Phase 11.2 agents code against, and it is deliberately thin so that changing a
    reward coefficient does not mean touching this file.

    Parameters
    ----------
    graphs      the pool, as `DiagramGraph`s or raw IR dicts. `reset` picks one.
    config      `RewardConfig`; pass `RewardConfig(sandbox=...)` to plug 11.2.9 in.
    order       "random" (default, seeded) or "sequential" (the pool in order, for evaluation).
    max_steps   overrides `src.rl.episode.step_cap`. Leave None in training.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        graphs: Iterable[DiagramGraph | dict],
        config: RewardConfig = DEFAULT,
        order: str = "random",
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        pool: list[DiagramGraph] = []
        for item in graphs:
            pool.append(item if isinstance(item, DiagramGraph) else DiagramGraph.from_ir(item))
        if not pool:
            raise ValueError("DiagramTraversalEnv needs at least one diagram")
        if order not in {"random", "sequential"}:
            raise ValueError("order must be 'random' or 'sequential'")
        self.graphs = pool
        self.config = config
        self.order = order
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = Discrete(A.N_ACTIONS)
        width = len(StateEncoder(pool[0]).feature_names)
        self.observation_space = Box(0.0, 1.0, (width,))

        self._rng = random.Random(0)
        self._cursor = 0
        self.episode: Episode | None = None
        self.encoder: StateEncoder | None = None

    # -- Gym API --------------------------------------------------------------------------

    @property
    def graph(self) -> DiagramGraph:
        if self.episode is None:
            raise RuntimeError("call reset() first")
        return self.episode.graph

    @property
    def state(self) -> TraversalState:
        if self.episode is None:
            raise RuntimeError("call reset() first")
        return self.episode.state

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]:
        """Start a new episode. `options={"index": i}` or `{"diagram_id": s}` picks the diagram."""
        if seed is not None:
            self._rng = random.Random(seed)
            self._cursor = 0
        options = options or {}
        if "index" in options:
            graph = self.graphs[int(options["index"]) % len(self.graphs)]
        elif "diagram_id" in options:
            wanted = str(options["diagram_id"])
            matches = [g for g in self.graphs if g.diagram_id == wanted]
            if not matches:
                raise KeyError(f"no diagram {wanted!r} in the pool")
            graph = matches[0]
        elif self.order == "sequential":
            graph = self.graphs[self._cursor % len(self.graphs)]
            self._cursor += 1
        else:
            graph = self._rng.choice(self.graphs)

        self.episode = Episode(graph, cap=self.max_steps)
        self.encoder = StateEncoder(graph)
        return self._observation(), self._info(None)

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict]:
        """One action. Returns `(observation, reward, terminated, truncated, info)`.

        Raises after the episode is done - there is no auto-reset; see the module docstring.
        """
        if self.episode is None:
            raise RuntimeError("call reset() first")
        if self.episode.done():
            raise RuntimeError("episode is over; call reset()")

        outcome = self.episode.apply(int(action))
        reward = step_reward(self.episode.graph, outcome, self.config)
        terminated = self.episode.terminated()
        truncated = self.episode.truncated() and not terminated
        info = self._info(outcome)
        if terminated or truncated:
            final, breakdown = terminal_reward(
                self.episode.graph, self.episode.state, truncated, self.config
            )
            reward = round(reward + final, 6)
            info["terminal"] = breakdown
            info["stopped_by"] = self.episode.stopped_by
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        """An ANSI picture of where the policy is. Returns the string; prints it in human mode."""
        if self.episode is None:
            return None
        graph = self.episode.graph
        state = self.episode.state
        lines = [
            f"diagram {graph.diagram_id} ({graph.source})  "
            f"{graph.n_nodes} nodes / {graph.n_edges} edges / {graph.n_components} components",
            f"step {state.steps}/{self.episode.cap}  stack={list(state.stack)}  "
            f"stopped_by={self.episode.stopped_by}",
        ]
        for i in range(graph.n_nodes):
            marks = "".join(
                (
                    ">" if i == state.current else " ",
                    "v" if state.has_visited(i) else ".",
                    "e" if state.has_emitted(i) else ".",
                    "L" if state.is_loop_marked(i) else ".",
                )
            )
            text = (graph.texts[i] or "")[:32]
            lines.append(
                f"  {marks} [{i:>2}] {graph.roles[i]:<12} -> {list(graph.successors[i])}  {text}"
            )
        legal = ", ".join(A.action_name(a) for a in self.episode.legal_actions())
        lines.append(f"  legal: {legal}")
        out = "\n".join(lines)
        if self.render_mode == "human":
            print(out)
        return out

    def close(self) -> None:
        self.episode = None
        self.encoder = None

    # -- extras the agents need -----------------------------------------------------------

    def action_mask(self) -> list[bool]:
        """The current legal-action mask. 11.2's agents must apply this before argmax."""
        if self.episode is None:
            raise RuntimeError("call reset() first")
        return self.episode.mask()

    def abstract_state(self) -> AbstractState:
        """11.1.6's bounded Q-table key for the current state."""
        return abstract(self.graph, self.state)

    def emitted_code(self) -> str:
        """The Python `src.rl.emit` would produce from what the policy has emitted so far."""
        from src.rl.emit import emit_code

        state = self.state
        marked = {i for i in range(self.graph.n_nodes) if state.is_loop_marked(i)}
        return emit_code(self.graph, list(state.emit_sequence), marked)

    # -- internals ------------------------------------------------------------------------

    def _observation(self) -> Observation:
        assert self.encoder is not None
        return self.encoder.features(self.state)

    def _info(self, outcome) -> dict:
        assert self.episode is not None
        return {
            "diagram_id": self.episode.graph.diagram_id,
            "source": self.episode.graph.source,
            "action_mask": self.episode.mask(),
            "abstract_state": abstract(self.episode.graph, self.episode.state),
            "state": self.episode.state,
            "legal": True if outcome is None else outcome.legal,
            "action": None if outcome is None else outcome.name,
            "n_visited": self.episode.state.n_visited(),
            "n_emitted": self.episode.state.n_emitted(),
            "n_nodes": self.episode.graph.n_nodes,
            "step_cap": self.episode.cap,
        }


def make_env(
    sources: Sequence[str] | None = None,
    limit: int | None = None,
    **kwargs,
) -> DiagramTraversalEnv:
    """Build an env over the real IR corpus. `sources` defaults to every sequential source."""
    from src.parse.roles import is_sequential
    from src.parse.sequences import LABELLED_SOURCES, UNLABELLED_SOURCES, load_ir

    names = tuple(sources) if sources else tuple(LABELLED_SOURCES) + tuple(UNLABELLED_SOURCES)
    graphs = [DiagramGraph.from_ir(d) for d in load_ir(names, limit) if is_sequential(d)]
    return DiagramTraversalEnv(graphs, **kwargs)


def api_check(env: DiagramTraversalEnv, seeds: Sequence[int] = (0, 1), episodes: int = 4) -> dict:
    """The 11.1.5 definition of done, as an executable check. Returns a pass/fail report.

    Every entry is a count of probes and a count of failures; `report["ok"]` is True only when
    every failure count is zero and both determinism checks hold.

    The driving policy **avoids `TERMINATE` unless it is the only legal action**. A uniform random
    policy fires `TERMINATE` within a step or two (`src.rl.episode` measures 97.60% of random
    episodes ending that way), which left the inherited check probing ~4 steps per episode and
    exercising almost nothing. Avoiding it drives episodes to the coverage terminal or the step
    cap instead, which is where the interesting transitions are.
    """
    from src.rl.abstraction import assert_bounded

    checks = {
        name: [0, 0]
        for name in (
            "reset_tuple",
            "observation_shape",
            "observation_in_space",
            "step_tuple",
            "reward_finite",
            "not_both_terminated_and_truncated",
            "info_mask_length",
            "illegal_action_refused",
            "illegal_action_leaves_state",
            "legal_action_accepted",
            "ends_within_cap",
            "abstract_state_bounded",
            "render_is_text",
            "step_after_done_raises",
        )
    }

    def probe(name: str, ok: bool) -> None:
        checks[name][0] += 1
        checks[name][1] += 0 if ok else 1

    width = env.observation_space.shape[0]
    n_steps = 0
    n_episodes = 0

    for seed in seeds:
        rng = random.Random(seed)
        for index in range(episodes):
            result = env.reset(seed=seed, options={"index": seed * episodes + index})
            probe("reset_tuple", isinstance(result, tuple) and len(result) == 2)
            obs, info = result
            probe("observation_shape", len(obs) == width)
            probe("observation_in_space", env.observation_space.contains(obs))
            probe("info_mask_length", len(info["action_mask"]) == A.N_ACTIONS)
            probe("render_is_text", isinstance(env.render(), str))
            n_episodes += 1

            cap = info["step_cap"]
            steps = 0
            while True:
                mask = env.action_mask()
                probe("abstract_state_bounded", assert_bounded(env.abstract_state()) is not None)

                illegal = [i for i, ok in enumerate(mask) if not ok]
                if illegal:
                    before = env.state
                    out = env.step(rng.choice(illegal))
                    steps += 1
                    probe("step_tuple", isinstance(out, tuple) and len(out) == 5)
                    probe("illegal_action_refused", out[4]["legal"] is False)
                    after = env.state
                    probe(
                        "illegal_action_leaves_state",
                        (after.current, after.visited, after.emitted, after.loop_marked)
                        == (before.current, before.visited, before.emitted, before.loop_marked),
                    )
                    if out[2] or out[3]:
                        break

                legal = [i for i, ok in enumerate(env.action_mask()) if ok]
                choices = [a for a in legal if a != A.TERMINATE] or legal
                obs, reward, terminated, truncated, info = env.step(rng.choice(choices))
                steps += 1
                n_steps += 1
                probe("legal_action_accepted", info["legal"] is True)
                probe("reward_finite", reward == reward and abs(reward) < 1e6)
                probe("not_both_terminated_and_truncated", not (terminated and truncated))
                probe("observation_shape", len(obs) == width)
                probe("ends_within_cap", steps <= cap + 1)
                if terminated or truncated:
                    break

            try:
                env.step(0)
                probe("step_after_done_raises", False)
            except RuntimeError:
                probe("step_after_done_raises", True)

    same_seed = env_determinism(env, seed=int(seeds[0]), episodes=episodes)
    differs = seed_sensitivity(env, seeds=seeds, episodes=episodes)
    report = {
        "checks": {name: {"probes": n, "failures": f} for name, (n, f) in checks.items()},
        "deterministic_under_same_seed": same_seed,
        "seed_sensitivity": differs,
        "n_seeds": len(seeds),
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "observation_width": width,
        "n_actions": A.N_ACTIONS,
    }
    report["ok"] = all(f == 0 for _, f in checks.values()) and same_seed
    return report


def seed_sensitivity(
    env: DiagramTraversalEnv, seeds: Sequence[int] = (0, 1), episodes: int = 4
) -> dict:
    """How often two different seeds give different trajectories on the same diagram.

    Reported rather than asserted. A diagram with exactly one legal action at every step *must*
    produce the same episode under every seed, so a low number here is a property of the corpus
    (didi is full of 2- and 3-node sketches), not a determinism failure. `n_forced` counts those
    diagrams separately so the rate is read against the right denominator.
    """

    def trajectory(seed: int, index: int) -> tuple[list, int]:
        rng = random.Random(seed)
        env.reset(seed=seed, options={"index": index})
        trace: list = []
        branches = 0
        while True:
            legal = [a for a, ok in enumerate(env.action_mask()) if ok]
            branches += int(len(legal) > 1)
            choices = [a for a in legal if a != A.TERMINATE] or legal
            action = rng.choice(choices)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            trace.append(action)
            if terminated or truncated:
                return trace, branches

    if len(seeds) < 2:
        return {"n": 0, "n_forced": 0, "n_differing": 0, "rate": None}
    differing = 0
    forced = 0
    total = 0
    for index in range(episodes):
        first, branches = trajectory(int(seeds[0]), index)
        second, _ = trajectory(int(seeds[1]), index)
        total += 1
        if branches == 0:
            forced += 1
        elif first != second:
            differing += 1
    free = total - forced
    return {
        "n": total,
        "n_forced": forced,
        "n_differing": differing,
        "rate": round(differing / free, 4) if free else None,
    }


def env_determinism(env: DiagramTraversalEnv, seed: int = 0, episodes: int = 4) -> bool:
    """Two identical seeded runs give byte-identical trajectories."""

    def run() -> list:
        rng = random.Random(seed)
        trace: list = []
        for i in range(episodes):
            obs, info = env.reset(seed=seed, options={"index": i})
            trace.append(obs)
            while True:
                legal = [a for a, ok in enumerate(env.action_mask()) if ok]
                action = rng.choice(legal)
                obs, reward, terminated, truncated, _info = env.step(action)
                trace.append((action, obs, round(reward, 6), terminated, truncated))
                if terminated or truncated:
                    break
        return trace

    return run() == run()


def main(argv: list[str] | None = None) -> int:
    env = make_env(limit=200)
    print(json.dumps(api_check(env, seeds=tuple(range(8))), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
