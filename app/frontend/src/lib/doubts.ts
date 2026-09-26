/**
 * Phase 16.2.9 - everything the pipeline said it was unsure about, in one ranked list.
 *
 * ## The client was hiding declared uncertainty
 *
 * 2.1.6 put three fields in the IR for ambiguity a hand-drawn page leaves behind: `unresolved_edges`,
 * `crossed_out` and `low_conf_text`. They are *recorded rather than resolved* on purpose - Phase 10
 * repairs what it can and the rest is meant to be asked about. A real payload from this repository
 * carries four of them:
 *
 *     "unresolved_edges": [{"edge": "traced_0000", "reason": "both-ends-open"}, ...]
 *
 * Those are arrows the tracer found and could not attach to anything. They are **not in `edges`**,
 * so before this row neither the overlay nor the graph drew them and nothing on any screen said they
 * existed. The pipeline had done the honest thing four phases earlier and the last hop dropped it.
 *
 * ## Ranked by what a person should look at first, not by where it came from
 *
 *   confirm      the router was below its floor. Everything after it is type-specific, so this one
 *                invalidates the rest of the reading rather than adding to it
 *   unattached   an arrow was seen and not connected. A missing connection changes the program;
 *                a misread word changes a name
 *   crossed out  something the person scribbled out and the reader may have kept
 *   text         the pipeline's own doubt about what a word says
 *   low          a node or edge below the confidence floor, worst first
 *
 * ## The threshold is one number and it is named
 *
 * `LOW_CONFIDENCE = 0.5` is this client's, not the pipeline's: 13.5's 0.60 floor is about *routing*
 * and means something different. It is exported and used by the overlay, the graph and this list, so
 * the dashed orange box on the picture and the row in this panel are the same claim.
 */

import type { Diagram, Edge, Node } from "./api";

/** Below this, a detection is worth looking at rather than trusting. */
export const LOW_CONFIDENCE = 0.5;

export type DoubtKind = "confirm" | "unattached" | "crossed_out" | "text" | "low_node" | "low_edge";

export interface Doubt {
  kind: DoubtKind;
  /** What to say. One sentence, in the language of the drawing. */
  title: string;
  detail: string;
  /** The node this points at, when tapping it should select something. */
  node?: string;
  edge?: string;
  confidence?: number;
}

const RANK: Record<DoubtKind, number> = {
  confirm: 0,
  unattached: 1,
  crossed_out: 2,
  text: 3,
  low_node: 4,
  low_edge: 5,
};

/**
 * The four reasons `Diagram.record_unresolved` writes, in words rather than in its vocabulary.
 *
 * Taken from `src/ir/model.py`, which is the only place they are produced - the first version of
 * this map was written from memory and had three of them wrong (`source-open`, `target-open`,
 * `ambiguous-target`), none of which any payload contains. The browser showed the fallback text for
 * a real `no-source` before the list was checked against the source.
 */
const REASONS: Record<string, string> = {
  "both-ends-open": "neither end of this arrow reached a shape",
  "no-source": "the tail of this arrow did not start at a shape",
  "no-target": "the head of this arrow did not reach a shape",
  "ambiguous-endpoint": "this arrow ended between shapes and the reader could not choose",
};

function say(reason: unknown): string {
  const key = String(reason ?? "");
  return REASONS[key] ?? (key ? `the tracer reported "${key}"` : "the tracer did not say why");
}

export function doubts(
  ir: Diagram | null,
  flags: { needsConfirmation?: boolean; diagramType?: string } = {},
): Doubt[] {
  const out: Doubt[] = [];

  if (flags.needsConfirmation) {
    out.push({
      kind: "confirm",
      title: "It is not sure what kind of diagram this is",
      detail:
        "Everything after the diagram type is specific to that type, so it stopped rather than " +
        "writing confident code for the wrong kind of drawing.",
    });
  }

  for (const raw of ir?.unresolved_edges ?? []) {
    const item = raw as { edge?: string; reason?: string };
    out.push({
      kind: "unattached",
      title: "An arrow was found but not connected",
      detail: say(item.reason),
      edge: item.edge,
    });
  }

  for (const raw of ir?.crossed_out ?? []) {
    const item = raw as { ref?: string; id?: string; reason?: string };
    out.push({
      kind: "crossed_out",
      title: "Something looks crossed out",
      detail:
        item.reason ??
        "The reader saw strokes over this and kept it. If it was deleted on the page, it should " +
          "not be in the code.",
      node: item.ref ?? item.id,
    });
  }

  for (const raw of ir?.low_conf_text ?? []) {
    const item = raw as { ref?: string; id?: string; confidence?: number };
    out.push({
      kind: "text",
      title: "The handwriting here was hard to read",
      detail: "The reader is not confident it got these words right.",
      node: item.ref ?? item.id,
      confidence: item.confidence,
    });
  }

  const nodes = (ir?.nodes ?? []) as Node[];
  for (const node of nodes) {
    const confidence = node.confidence ?? 1;
    // A corrected label is a person's, not the recogniser's, and must stop being flagged. 16.2.8
    // sets both when a correction lands.
    if (node.corrected === true || confidence >= LOW_CONFIDENCE) continue;
    out.push({
      kind: "low_node",
      title: node.text ? `"${String(node.text)}" may be wrong` : "A shape with no readable text",
      detail: `The reader was ${Math.round(confidence * 100)}% sure about this one.`,
      node: node.id,
      confidence,
    });
  }

  for (const edge of (ir?.edges ?? []) as Edge[]) {
    const confidence = edge.confidence ?? 1;
    if (confidence >= LOW_CONFIDENCE) continue;
    out.push({
      kind: "low_edge",
      title:
        edge.src && edge.dst && edge.src === edge.dst
          ? "An arrow that loops back to its own shape"
          : "An arrow it is unsure about",
      detail: `The reader was ${Math.round(confidence * 100)}% sure this arrow is there.`,
      edge: edge.id,
      node: edge.src ?? undefined,
      confidence,
    });
  }

  // Ranked by what to look at first, then worst-confidence first within a kind. A stable sort, so
  // two items with the same rank and no confidence keep the order the IR gave them - which for
  // unattached edges is the order they were traced.
  return out
    .map((doubt, index) => ({ doubt, index }))
    .sort((a, b) => {
      const byKind = RANK[a.doubt.kind] - RANK[b.doubt.kind];
      if (byKind !== 0) return byKind;
      const ca = a.doubt.confidence ?? 1;
      const cb = b.doubt.confidence ?? 1;
      if (ca !== cb) return ca - cb;
      return a.index - b.index;
    })
    .map(({ doubt }) => doubt);
}

/** One line for the header: how much of this reading is worth a second look. */
export function summarise(list: Doubt[]): { count: number; worst: DoubtKind | null } {
  if (!list.length) return { count: 0, worst: null };
  return { count: list.length, worst: list[0].kind };
}
