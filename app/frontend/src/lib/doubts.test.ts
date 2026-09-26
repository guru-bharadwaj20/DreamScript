/**
 * Phase 16.2.9 - what the reading is unsure about, and the ordering that makes it useful.
 *
 * The defect this row exists for is not a missing colour. 2.1.6 put three fields in the IR for
 * ambiguity - `unresolved_edges`, `crossed_out`, `low_conf_text` - and a real payload from this
 * repository carries four unresolved edges reading `{"edge": "traced_0000", "reason":
 * "both-ends-open"}`. Those arrows are **not in `edges`**, so nothing on any screen mentioned them.
 * The pipeline had done the honest thing four phases earlier and the client dropped it.
 */

import { describe, expect, it } from "vitest";

import type { Diagram } from "./api";
import { LOW_CONFIDENCE, doubts, summarise } from "./doubts";

const ir = (extra: Partial<Diagram> = {}): Diagram => ({ nodes: [], edges: [], ...extra });

describe("doubts", () => {
  it("surfaces unresolved edges, which are in no other list", () => {
    const list = doubts(
      ir({
        unresolved_edges: [
          { edge: "traced_0000", reason: "both-ends-open" },
          { edge: "traced_0001", reason: "target-open" },
        ],
      }),
    );
    expect(list).toHaveLength(2);
    expect(list[0].kind).toBe("unattached");
    expect(list[0].edge).toBe("traced_0000");
  });

  it("says what the tracer meant rather than repeating its vocabulary", () => {
    const [first] = doubts(ir({ unresolved_edges: [{ edge: "e", reason: "both-ends-open" }] }));
    expect(first.detail).toBe("neither end of this arrow reached a shape");
  });

  it.each(["both-ends-open", "no-source", "no-target", "ambiguous-endpoint"])(
    "has words for the reason %s, which `Diagram.record_unresolved` actually emits",
    (reason) => {
      // The four in `src/ir/model.py`. The first version of the map was written from memory and
      // had three names that no payload contains, so a real `no-source` fell through to the
      // fallback - which is correct behaviour hiding an incorrect table.
      const [first] = doubts(ir({ unresolved_edges: [{ edge: "e", reason }] }));
      expect(first.detail).not.toContain("the tracer reported");
    },
  );

  it("keeps an unknown reason rather than inventing one", () => {
    const [first] = doubts(ir({ unresolved_edges: [{ edge: "e", reason: "some-new-reason" }] }));
    expect(first.detail).toContain("some-new-reason");
  });

  it("copes with an unresolved edge that carries no reason", () => {
    const [first] = doubts(ir({ unresolved_edges: [{ edge: "e" }] }));
    expect(first.detail).toContain("did not say");
  });

  it("flags a node below the threshold and not one above it", () => {
    const list = doubts(
      ir({
        nodes: [
          { id: "a", text: "sure", confidence: 0.99 },
          { id: "b", text: "maybe", confidence: 0.2 },
        ],
      }),
    );
    expect(list).toHaveLength(1);
    expect(list[0].node).toBe("b");
    expect(list[0].title).toContain("maybe");
  });

  it("stops flagging a label a person has corrected", () => {
    // 16.2.8 sets `corrected` and raises the confidence. The label is a person's now, and leaving
    // it flagged would ask them to check their own typing.
    const list = doubts(
      ir({ nodes: [{ id: "a", text: "Receive order", confidence: 1, corrected: true }] }),
    );
    expect(list).toHaveLength(0);
  });

  it("names a self loop as one rather than as a generic uncertain arrow", () => {
    const [first] = doubts(ir({ edges: [{ id: "e", src: "a", dst: "a", confidence: 0.17 }] }));
    expect(first.title).toContain("loops back");
  });

  it("puts the routing question above everything else", () => {
    // Everything after the diagram type is type-specific, so this one invalidates the rest of the
    // reading rather than adding to it.
    const list = doubts(
      ir({
        unresolved_edges: [{ edge: "e", reason: "both-ends-open" }],
        nodes: [{ id: "a", confidence: 0.1 }],
      }),
      { needsConfirmation: true },
    );
    expect(list[0].kind).toBe("confirm");
  });

  it("ranks an unattached arrow above a doubtful word", () => {
    // A missing connection changes the program; a misread word changes a name.
    const list = doubts(
      ir({
        unresolved_edges: [{ edge: "e", reason: "target-open" }],
        nodes: [{ id: "a", text: "x", confidence: 0.05 }],
      }),
    );
    expect(list.map((d) => d.kind)).toEqual(["unattached", "low_node"]);
  });

  it("orders within a kind by worst confidence first", () => {
    const list = doubts(
      ir({
        nodes: [
          { id: "a", text: "a", confidence: 0.45 },
          { id: "b", text: "b", confidence: 0.05 },
          { id: "c", text: "c", confidence: 0.3 },
        ],
      }),
    );
    expect(list.map((d) => d.node)).toEqual(["b", "c", "a"]);
  });

  it("is a stable sort, so equally uncertain items keep the IR's order", () => {
    const list = doubts(
      ir({
        unresolved_edges: [
          { edge: "traced_0000", reason: "both-ends-open" },
          { edge: "traced_0001", reason: "both-ends-open" },
          { edge: "traced_0002", reason: "both-ends-open" },
        ],
      }),
    );
    expect(list.map((d) => d.edge)).toEqual(["traced_0000", "traced_0001", "traced_0002"]);
  });

  it("reads crossed-out and low-confidence text from the IR's own fields", () => {
    const list = doubts(
      ir({
        crossed_out: [{ ref: "n1" }],
        low_conf_text: [{ ref: "n2", confidence: 0.3 }],
      }),
    );
    expect(list.map((d) => d.kind)).toEqual(["crossed_out", "text"]);
    expect(list[0].node).toBe("n1");
    expect(list[1].confidence).toBe(0.3);
  });

  it("is empty for a clean reading", () => {
    const list = doubts(
      ir({
        nodes: [{ id: "a", text: "Start", confidence: 1 }],
        edges: [{ id: "e", src: "a", dst: "a", confidence: 0.9 }],
      }),
    );
    expect(list).toEqual([]);
    expect(summarise(list)).toEqual({ count: 0, worst: null });
  });

  it("is empty rather than a crash for a missing IR", () => {
    expect(doubts(null)).toEqual([]);
  });

  it("summarises the count and the worst kind", () => {
    const list = doubts(ir({ nodes: [{ id: "a", confidence: 0.1 }] }), { needsConfirmation: true });
    expect(summarise(list)).toEqual({ count: 2, worst: "confirm" });
  });

  it("uses one threshold, exported, so the picture and the list cannot disagree", () => {
    expect(LOW_CONFIDENCE).toBe(0.5);
    const just_under = doubts(ir({ nodes: [{ id: "a", confidence: LOW_CONFIDENCE - 0.001 }] }));
    const exactly = doubts(ir({ nodes: [{ id: "a", confidence: LOW_CONFIDENCE }] }));
    expect(just_under).toHaveLength(1);
    expect(exactly).toHaveLength(0);
  });
});
