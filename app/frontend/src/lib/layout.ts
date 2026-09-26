/**
 * Phase 16.2.6 - laying the IR out as a graph, in about a hundred and fifty lines.
 *
 * ## Why not the detector's own coordinates
 *
 * The obvious layout is the one the overlay uses: put every node where its bbox says it is. That is
 * right *on the photograph* and wrong as a graph, for the reason this view exists at all. A
 * hand-drawn page is crooked, its boxes overlap, its arrows cross, and a reading of it that is
 * structurally correct can look like a mess - which leaves a person unable to tell a bad reading
 * from a bad drawing. Laid out by structure, the same IR shows its shape: a chain is a column, a
 * branch forks, a cycle closes.
 *
 * Both views are kept because they answer different questions. The photograph answers "did it see
 * my page"; this answers "what did it understand".
 *
 * ## Why not d3 or dagre
 *
 * `dagre` is the right tool and is 200 kB. This app's whole bundle is 190 kB and 16.3.1 caches it
 * for offline use. What is needed here is a layered layout of a graph that is almost always under
 * twenty nodes - a topological sort into ranks, a median heuristic to order each rank, and straight
 * edges. The expensive part of Sugiyama is crossing minimisation on large graphs; at this size one
 * ordering pass is enough and the difference is not visible.
 *
 * ## The three things this handles that a naive layered layout does not
 *
 *   cycles        a hand-drawn flowchart loops, and a topological sort of a cyclic graph does not
 *                 terminate. Back edges are found by a depth-first search and removed for ranking,
 *                 then drawn anyway - as a curve, so a loop reads as a loop
 *   disconnected  10.1 produces islands, because a missed arrow orphans a subgraph. Each component
 *                 is ranked from its own root rather than all of them from rank 0, which would
 *                 stack unrelated nodes in one line
 *   self loops    `a000` in a real payload has `src === dst`. Drawn as a small circle on the node
 *                 rather than as an edge of length zero, which is an invisible artefact that still
 *                 counts in the edge total
 */

import type { Diagram, Edge, Node } from "./api";

export interface Placed {
  node: Node;
  x: number;
  y: number;
  width: number;
  height: number;
  rank: number;
  /** Index in 10.1's traversal, 1-based; 0 when it is not in the reading order. */
  step: number;
}

export interface Routed {
  edge: Edge;
  from: Placed;
  to: Placed;
  /** A back edge closes a cycle and is drawn curved so it reads as a loop. */
  back: boolean;
  selfLoop: boolean;
}

export interface Layout {
  nodes: Placed[];
  edges: Routed[];
  width: number;
  height: number;
  /** Nodes with no edges at all, which is worth saying on screen rather than drawing silently. */
  isolated: number;
}

const NODE_W = 132;
const NODE_H = 46;
const GAP_X = 26;
const GAP_Y = 54;
const PAD = 20;

/**
 * Find the edges that close a cycle, by depth-first search.
 *
 * An edge to a node currently on the stack is a back edge. Removing exactly those makes the graph
 * acyclic without deleting any node, which matters because a hand-drawn flowchart is usually one
 * loop away from a tree and dropping a node would change what the person is looking at.
 */
function backEdges(nodes: Node[], edges: Edge[]): Set<string> {
  const out = new Map<string, { id: string; to: string }[]>();
  for (const node of nodes) out.set(node.id, []);
  for (const edge of edges) {
    if (!edge.src || !edge.dst || edge.src === edge.dst) continue;
    out.get(edge.src)?.push({ id: edge.id, to: edge.dst });
  }

  const back = new Set<string>();
  const state = new Map<string, 0 | 1 | 2>();
  const walk = (id: string) => {
    state.set(id, 1);
    for (const { id: edgeId, to } of out.get(id) ?? []) {
      const seen = state.get(to) ?? 0;
      if (seen === 1) back.add(edgeId);
      else if (seen === 0) walk(to);
    }
    state.set(id, 2);
  };
  for (const node of nodes) if (!state.get(node.id)) walk(node.id);
  return back;
}

/** Rank every node: longest path from a source, per connected component. */
function rank(nodes: Node[], edges: Edge[], back: Set<string>): Map<string, number> {
  const forward = edges.filter(
    (e) => e.src && e.dst && e.src !== e.dst && !back.has(e.id),
  );
  const incoming = new Map<string, number>();
  const out = new Map<string, string[]>();
  for (const node of nodes) {
    incoming.set(node.id, 0);
    out.set(node.id, []);
  }
  for (const edge of forward) {
    out.get(edge.src!)?.push(edge.dst!);
    incoming.set(edge.dst!, (incoming.get(edge.dst!) ?? 0) + 1);
  }

  const ranks = new Map<string, number>();
  // Kahn's algorithm, taking the longest path rather than the first: a node whose two parents are
  // at ranks 0 and 3 belongs at 4, not at 1, or its edge runs backwards up the page.
  const queue = nodes.filter((n) => (incoming.get(n.id) ?? 0) === 0).map((n) => n.id);
  for (const id of queue) ranks.set(id, 0);
  while (queue.length) {
    const id = queue.shift() as string;
    for (const next of out.get(id) ?? []) {
      ranks.set(next, Math.max(ranks.get(next) ?? 0, (ranks.get(id) ?? 0) + 1));
      const left = (incoming.get(next) ?? 1) - 1;
      incoming.set(next, left);
      if (left === 0) queue.push(next);
    }
  }
  // Anything still unranked is inside a cycle whose every member had an incoming edge. Ranked
  // after its earliest neighbour so it lands somewhere sensible rather than at 0.
  for (const node of nodes) {
    if (ranks.has(node.id)) continue;
    const parents = forward.filter((e) => e.dst === node.id).map((e) => ranks.get(e.src!) ?? 0);
    ranks.set(node.id, parents.length ? Math.max(...parents) + 1 : 0);
  }
  return ranks;
}

/**
 * Order the nodes within each rank by the median position of their parents.
 *
 * One pass, downward. Sugiyama's crossing minimisation iterates; at twenty nodes the second pass
 * moves nothing a person would notice, and the first pass is what turns a rank ordered by
 * detection order - which is scan order on the photograph - into one where an edge mostly goes
 * straight down.
 */
function order(byRank: Map<number, Node[]>, edges: Edge[], ranks: Map<string, number>): void {
  const position = new Map<string, number>();
  const ranksAscending = [...byRank.keys()].sort((a, b) => a - b);
  for (const r of ranksAscending) {
    const row = byRank.get(r)!;
    if (r === ranksAscending[0]) {
      row.forEach((node, i) => position.set(node.id, i));
      continue;
    }
    const median = new Map<string, number>();
    for (const node of row) {
      const parents = edges
        .filter((e) => e.dst === node.id && e.src && ranks.get(e.src) === r - 1)
        .map((e) => position.get(e.src!) ?? 0)
        .sort((a, b) => a - b);
      median.set(
        node.id,
        parents.length ? parents[Math.floor(parents.length / 2)] : Number.MAX_SAFE_INTEGER,
      );
    }
    row.sort((a, b) => (median.get(a.id) ?? 0) - (median.get(b.id) ?? 0));
    row.forEach((node, i) => position.set(node.id, i));
  }
}

export function layout(ir: Diagram | null, traversal: string[] = []): Layout | null {
  const nodes = ir?.nodes ?? [];
  if (!nodes.length) return null;
  const edges = ir?.edges ?? [];

  const back = backEdges(nodes, edges);
  const ranks = rank(nodes, edges, back);

  const byRank = new Map<number, Node[]>();
  for (const node of nodes) {
    const r = ranks.get(node.id) ?? 0;
    if (!byRank.has(r)) byRank.set(r, []);
    byRank.get(r)!.push(node);
  }
  order(byRank, edges, ranks);

  const widest = Math.max(...[...byRank.values()].map((row) => row.length));
  const width = PAD * 2 + widest * NODE_W + (widest - 1) * GAP_X;
  const step = new Map(traversal.map((id, i) => [id, i + 1]));

  const placed: Placed[] = [];
  const ranksAscending = [...byRank.keys()].sort((a, b) => a - b);
  ranksAscending.forEach((r, row) => {
    const items = byRank.get(r)!;
    const rowWidth = items.length * NODE_W + (items.length - 1) * GAP_X;
    const left = (width - rowWidth) / 2;
    items.forEach((node, i) => {
      placed.push({
        node,
        x: left + i * (NODE_W + GAP_X),
        y: PAD + row * (NODE_H + GAP_Y),
        width: NODE_W,
        height: NODE_H,
        rank: r,
        step: step.get(node.id) ?? 0,
      });
    });
  });

  const index = new Map(placed.map((p) => [p.node.id, p]));
  const routed: Routed[] = [];
  const touched = new Set<string>();
  for (const edge of edges) {
    if (!edge.src || !edge.dst) continue;
    const from = index.get(edge.src);
    const to = index.get(edge.dst);
    if (!from || !to) continue;
    touched.add(edge.src);
    touched.add(edge.dst);
    routed.push({ edge, from, to, back: back.has(edge.id), selfLoop: edge.src === edge.dst });
  }

  return {
    nodes: placed,
    edges: routed,
    width,
    height: PAD * 2 + ranksAscending.length * NODE_H + (ranksAscending.length - 1) * GAP_Y,
    isolated: placed.filter((p) => !touched.has(p.node.id)).length,
  };
}
