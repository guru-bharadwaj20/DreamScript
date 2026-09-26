/**
 * Phase 16.2.6 - the layout, against the three shapes a hand-drawn page actually produces.
 *
 * A layered layout is easy to test on a tree and a tree is the one input this will never get. 10.1
 * produces graphs that **loop** (a flowchart returns to a decision), that are **disconnected** (a
 * missed arrow orphans a subgraph), and that contain **self loops** (`a000` in a real payload has
 * `src === dst`). Each of those breaks a naive implementation in its own way: a topological sort of
 * a cyclic graph does not terminate, ranking every component from zero stacks unrelated nodes in
 * one line, and a self loop drawn as an edge has zero length and disappears.
 */

import { describe, expect, it } from "vitest";

import type { Diagram, Edge, Node } from "./api";
import { layout } from "./layout";

const node = (id: string, extra: Partial<Node> = {}): Node => ({
  id,
  shape: "rectangle",
  text: id,
  confidence: 1,
  ...extra,
});

const edge = (id: string, src: string, dst: string, extra: Partial<Edge> = {}): Edge => ({
  id,
  src,
  dst,
  directed: true,
  confidence: 1,
  ...extra,
});

const ir = (nodes: Node[], edges: Edge[]): Diagram => ({ nodes, edges });

const rankOf = (model: NonNullable<ReturnType<typeof layout>>, id: string) =>
  model.nodes.find((p) => p.node.id === id)!.rank;

describe("layout", () => {
  it("puts a chain in a column", () => {
    const model = layout(ir([node("a"), node("b"), node("c")], [edge("1", "a", "b"), edge("2", "b", "c")]))!;
    expect(rankOf(model, "a")).toBe(0);
    expect(rankOf(model, "b")).toBe(1);
    expect(rankOf(model, "c")).toBe(2);
    // One node per rank means one column, so every box has the same centre.
    const centres = new Set(model.nodes.map((p) => p.x + p.width / 2));
    expect(centres.size).toBe(1);
  });

  it("forks a branch onto one rank", () => {
    const model = layout(
      ir(
        [node("q"), node("yes"), node("no")],
        [edge("1", "q", "yes"), edge("2", "q", "no")],
      ),
    )!;
    expect(rankOf(model, "yes")).toBe(1);
    expect(rankOf(model, "no")).toBe(1);
    expect(model.nodes.find((p) => p.node.id === "yes")!.x).not.toBe(
      model.nodes.find((p) => p.node.id === "no")!.x,
    );
  });

  it("takes the longest path when a node has parents at two depths", () => {
    // `d` has parents at rank 0 and rank 2. At rank 1 its edge from `c` would run up the page.
    const model = layout(
      ir(
        [node("a"), node("b"), node("c"), node("d")],
        [edge("1", "a", "b"), edge("2", "b", "c"), edge("3", "c", "d"), edge("4", "a", "d")],
      ),
    )!;
    expect(rankOf(model, "d")).toBe(3);
  });

  it("terminates on a cycle and marks the edge that closes it", () => {
    // The case that hangs a naive topological sort. A flowchart that loops is the normal case.
    const model = layout(
      ir(
        [node("a"), node("b"), node("c")],
        [edge("1", "a", "b"), edge("2", "b", "c"), edge("3", "c", "a")],
      ),
    )!;
    expect(model.nodes).toHaveLength(3);
    const back = model.edges.filter((r) => r.back);
    expect(back).toHaveLength(1);
    expect(back[0].edge.id).toBe("3");
    // And the forward chain is still ranked properly rather than collapsed.
    expect(rankOf(model, "c")).toBe(2);
  });

  it("survives a graph that is entirely a cycle", () => {
    const model = layout(ir([node("a"), node("b")], [edge("1", "a", "b"), edge("2", "b", "a")]))!;
    expect(model.nodes).toHaveLength(2);
    expect(model.edges.filter((r) => r.back)).toHaveLength(1);
  });

  it("does not stack disconnected components on one rank", () => {
    // Two separate chains. Both start at rank 0, and both must have their *own* second rank rather
    // than every node piling into one line.
    const model = layout(
      ir(
        [node("a"), node("b"), node("x"), node("y")],
        [edge("1", "a", "b"), edge("2", "x", "y")],
      ),
    )!;
    expect(rankOf(model, "a")).toBe(0);
    expect(rankOf(model, "x")).toBe(0);
    expect(rankOf(model, "b")).toBe(1);
    expect(rankOf(model, "y")).toBe(1);
  });

  it("keeps a self loop as a self loop rather than a zero-length edge", () => {
    // Drawn as an edge it is invisible and still counts in the total, which sends a person looking
    // for an arrow that is not on screen.
    const model = layout(ir([node("a")], [edge("1", "a", "a")]))!;
    expect(model.edges).toHaveLength(1);
    expect(model.edges[0].selfLoop).toBe(true);
    expect(model.edges[0].back).toBe(false);
    expect(rankOf(model, "a")).toBe(0);
  });

  it("counts the nodes no arrow touches", () => {
    const model = layout(ir([node("a"), node("b"), node("lonely")], [edge("1", "a", "b")]))!;
    expect(model.isolated).toBe(1);
  });

  it("drops an edge whose endpoint is missing rather than crashing on it", () => {
    // 10.1 emits dangling edges: `src` or `dst` null when an arrow was found and its target was
    // not. The graph must draw what it has.
    const model = layout(
      ir([node("a")], [edge("1", "a", "ghost"), { ...edge("2", "a", "a"), dst: null }]),
    )!;
    expect(model.edges).toHaveLength(0);
    expect(model.nodes).toHaveLength(1);
  });

  it("carries the reading order onto each node", () => {
    const model = layout(ir([node("a"), node("b")], [edge("1", "a", "b")]), ["b", "a"])!;
    expect(model.nodes.find((p) => p.node.id === "b")!.step).toBe(1);
    expect(model.nodes.find((p) => p.node.id === "a")!.step).toBe(2);
  });

  it("gives a node outside the traversal no step rather than step zero", () => {
    const model = layout(ir([node("a"), node("b")], []), ["a"])!;
    expect(model.nodes.find((p) => p.node.id === "b")!.step).toBe(0);
  });

  it("is null for an empty diagram", () => {
    expect(layout(null)).toBeNull();
    expect(layout(ir([], []))).toBeNull();
  });

  it("sizes the canvas to the widest rank", () => {
    const wide = layout(
      ir(
        [node("q"), node("a"), node("b"), node("c")],
        [edge("1", "q", "a"), edge("2", "q", "b"), edge("3", "q", "c")],
      ),
    )!;
    const narrow = layout(ir([node("a"), node("b")], [edge("1", "a", "b")]))!;
    expect(wide.width).toBeGreaterThan(narrow.width);
    expect(narrow.height).toBeGreaterThan(0);
  });

  it("orders a rank by its parents rather than by detection order", () => {
    // Detection order is scan order on the photograph. Without the median pass, `l` and `r` come
    // back in the order they were found and the two edges cross for no reason.
    const model = layout(
      ir(
        [node("p1"), node("p2"), node("r"), node("l")],
        [edge("1", "p1", "l"), edge("2", "p2", "r")],
      ),
    )!;
    const p1 = model.nodes.find((p) => p.node.id === "p1")!;
    const p2 = model.nodes.find((p) => p.node.id === "p2")!;
    const l = model.nodes.find((p) => p.node.id === "l")!;
    const r = model.nodes.find((p) => p.node.id === "r")!;
    // Whichever way round the parents landed, each child sits on the same side as its parent.
    expect(Math.sign(l.x - r.x)).toBe(Math.sign(p1.x - p2.x));
  });
});
