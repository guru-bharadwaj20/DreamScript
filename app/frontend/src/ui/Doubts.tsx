/**
 * Phase 16.2.9 - "where to look", as a list you can tap.
 *
 * The row asks for low-confidence items to be *visually flagged*. The overlay and the graph already
 * do that - a dashed orange box on the picture. This is the other half, and it is the half that
 * makes the flagging useful: a list, ranked, that says what is uncertain and **jumps to it**.
 *
 * A flag on a picture tells a person there is a problem somewhere in the picture. A list tells them
 * there are four, which one matters most, and puts a finger on it.
 */

import { type Doubt, doubts, summarise } from "../lib/doubts";
import type { Prediction } from "../lib/api";
import { Card, Pill } from "../ui";
import "./Doubts.css";

const TONE: Record<Doubt["kind"], "confirm" | "stopped" | "degraded"> = {
  confirm: "confirm",
  unattached: "stopped",
  crossed_out: "degraded",
  text: "degraded",
  low_node: "degraded",
  low_edge: "degraded",
};

const LABEL: Record<Doubt["kind"], string> = {
  confirm: "type",
  unattached: "arrow",
  crossed_out: "crossed out",
  text: "handwriting",
  low_node: "shape",
  low_edge: "arrow",
};

export function Doubts({
  prediction,
  onSelect,
}: {
  prediction: Prediction;
  onSelect?: (id: string | null) => void;
}) {
  const list = doubts(prediction.ir, {
    needsConfirmation: prediction.needs_confirmation,
    diagramType: prediction.diagram_type,
  });
  const { count } = summarise(list);

  if (count === 0) {
    return (
      <Card style={{ padding: "var(--sp-4)" }} className="row">
        <Pill tone="ok">nothing flagged</Pill>
        <span className="muted grow" style={{ fontSize: 13 }}>
          Every shape and arrow came back confident, and the reader recorded no ambiguity.
        </span>
      </Card>
    );
  }

  return (
    <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
      <div className="row">
        <span className="eyebrow grow">Where to look</span>
        <Pill tone="degraded">{count}</Pill>
      </div>
      <p className="dim" style={{ fontSize: 12.5 }}>
        {/* The point of the whole panel, said once: these are the reader's own doubts, not a
            guess by this app about what might be wrong. */}
        Things the reader itself was unsure about, worst first.
      </p>

      <ul className="doubts">
        {list.map((doubt, index) => {
          const target = doubt.node ?? null;
          const Row = target && onSelect ? "button" : "div";
          return (
            <li key={`${doubt.kind}-${index}`}>
              <Row
                className={`doubt${target && onSelect ? " is-tappable" : ""}`}
                {...(target && onSelect ? { onClick: () => onSelect(target) } : {})}
              >
                <Pill tone={TONE[doubt.kind]}>{LABEL[doubt.kind]}</Pill>
                <span className="grow">
                  <strong>{doubt.title}</strong>
                  <small>{doubt.detail}</small>
                </span>
                {typeof doubt.confidence === "number" ? (
                  <span className="dim num doubt-score">{doubt.confidence.toFixed(2)}</span>
                ) : null}
              </Row>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}
