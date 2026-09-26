/**
 * Phase 16.2.6 - the IR as a graph, which is the same reading without the photograph.
 *
 * SVG here, and canvas in the overlay (16.2.5), and the difference is not inconsistency. The
 * overlay redraws a hundred primitives on every pinch of a photograph; this draws twenty boxes
 * once and then wants each of them to be selectable, focusable and readable by a screen reader -
 * which is what DOM elements are and canvas pixels are not. A graph of a hand-drawn page is small
 * by construction: the largest thing 10.1 has produced in this project is a few dozen nodes.
 *
 * ## What it shows that the overlay cannot
 *
 * A structurally correct reading of a crooked page looks crooked on the page. Laid out by
 * structure, a chain becomes a column and a branch forks - so a person can see that the *shape* is
 * right even when the drawing was a mess, which is precisely the case where the overlay is least
 * reassuring.
 *
 * ## Three things drawn differently because they mean different things
 *
 *   a back edge    curved, and gold-dashed. It closes a cycle: a flowchart that loops should read
 *                  as looping rather than as an arrow that happens to point upwards
 *   a self loop    a small circle on the node's shoulder. Drawn as an edge it has zero length, is
 *                  invisible, and still counts in the edge total - which is how a person ends up
 *                  looking for an arrow that is not on screen
 *   an isolated    nothing special, but it is counted under the graph. A node no arrow touches is
 *   node           usually a missed arrow rather than a real island, and it is the single most
 *                  useful thing to tell someone about a reading that looks complete
 */

import { useMemo } from "react";

import type { Diagram } from "../lib/api";
import { type Placed, layout } from "../lib/layout";
import { LOW_CONFIDENCE } from "./Overlay";
import "./Graph.css";

export function Graph({
  ir,
  traversal = [],
  selected = null,
  onSelect,
}: {
  ir: Diagram | null;
  traversal?: string[];
  selected?: string | null;
  onSelect?: (id: string | null) => void;
}) {
  const model = useMemo(() => layout(ir, traversal), [ir, traversal]);

  if (!model) {
    return (
      <div className="graph-empty">
        <p>
          <strong>No graph was assembled.</strong>
        </p>
        <p className="muted">
          The pipeline found no shapes it could connect, so there is nothing to lay out.
        </p>
      </div>
    );
  }

  return (
    <div className="graph">
      <svg
        viewBox={`0 0 ${model.width} ${model.height}`}
        width="100%"
        role="img"
        aria-label={`${model.nodes.length} shapes and ${model.edges.length} arrows`}
      >
        <defs>
          <marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--gold)" />
          </marker>
          <marker id="tip-low" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--degraded)" />
          </marker>
        </defs>

        {model.edges.map(({ edge, from, to, back, selfLoop }) => {
          const low = (edge.confidence ?? 1) < LOW_CONFIDENCE;
          const stroke = low ? "var(--degraded)" : "var(--gold)";
          if (selfLoop) {
            const cx = from.x + from.width - 6;
            const cy = from.y - 4;
            return (
              <circle
                key={edge.id}
                className="graph-self"
                cx={cx}
                cy={cy}
                r={8}
                stroke={stroke}
                strokeDasharray={low ? "3 3" : undefined}
              >
                <title>{`${edge.src} loops to itself`}</title>
              </circle>
            );
          }
          return (
            <path
              key={edge.id}
              className={back ? "graph-edge graph-back" : "graph-edge"}
              d={path(from, to, back)}
              stroke={stroke}
              strokeDasharray={back || low ? "5 4" : undefined}
              markerEnd={`url(#${low ? "tip-low" : "tip"})`}
            >
              <title>
                {`${edge.src} → ${edge.dst}${edge.label ? ` (${edge.label})` : ""}`}
              </title>
            </path>
          );
        })}

        {model.edges
          .filter((r) => r.edge.label && !r.selfLoop)
          .map(({ edge, from, to }) => {
            const mx = (from.x + from.width / 2 + to.x + to.width / 2) / 2;
            const my = (from.y + from.height + to.y) / 2;
            return (
              <text key={`l-${edge.id}`} className="graph-label" x={mx} y={my} textAnchor="middle">
                {String(edge.label)}
              </text>
            );
          })}

        {model.nodes.map((placed) => (
          <GraphNode
            key={placed.node.id}
            placed={placed}
            selected={placed.node.id === selected}
            onSelect={onSelect}
          />
        ))}
      </svg>

      <p className="graph-note dim">
        {model.nodes.length} shape{model.nodes.length === 1 ? "" : "s"}, {model.edges.length} arrow
        {model.edges.length === 1 ? "" : "s"}
        {model.isolated > 0
          ? ` — ${model.isolated} touched by no arrow, which usually means one was missed`
          : ""}
      </p>
    </div>
  );
}

function GraphNode({
  placed,
  selected,
  onSelect,
}: {
  placed: Placed;
  selected: boolean;
  onSelect?: (id: string | null) => void;
}) {
  const { node, x, y, width, height, step } = placed;
  const low = (node.confidence ?? 1) < LOW_CONFIDENCE;
  const text = String(node.text ?? "").trim();

  return (
    <g
      className={`graph-node${selected ? " is-selected" : ""}${low ? " is-low" : ""}`}
      // A real button, not a div with a click handler: this is how the graph is reachable by
      // keyboard and announced by a screen reader, which canvas could not have been.
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={() => onSelect?.(selected ? null : node.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect?.(selected ? null : node.id);
        }
      }}
    >
      <Shape shape={String(node.shape ?? "rectangle")} x={x} y={y} w={width} h={height} />
      <text className="graph-text" x={x + width / 2} y={y + height / 2 + 4} textAnchor="middle">
        {text ? clip(text) : "—"}
      </text>
      {step > 0 ? (
        <>
          <circle className="graph-step" cx={x} cy={y} r={9} />
          <text className="graph-step-text" x={x} y={y + 4} textAnchor="middle">
            {step}
          </text>
        </>
      ) : null}
      <title>{`${node.shape ?? "shape"} ${node.id}${text ? `: ${text}` : ""}`}</title>
    </g>
  );
}

/**
 * The node's own shape, because a diamond means a decision.
 *
 * 10.1 records `shape` per node and a flowchart read as all-rectangles is a flowchart whose
 * branches are invisible. Four shapes cover every type this pipeline emits; anything else falls
 * back to a rectangle rather than to nothing.
 */
function Shape({ shape, x, y, w, h }: { shape: string; x: number; y: number; w: number; h: number }) {
  const common = { className: "graph-shape" };
  switch (shape) {
    case "diamond":
      return (
        <polygon
          {...common}
          points={`${x + w / 2},${y} ${x + w},${y + h / 2} ${x + w / 2},${y + h} ${x},${y + h / 2}`}
        />
      );
    case "ellipse":
    case "circle":
      return <ellipse {...common} cx={x + w / 2} cy={y + h / 2} rx={w / 2} ry={h / 2} />;
    case "rounded":
    case "terminator":
      return <rect {...common} x={x} y={y} width={w} height={h} rx={h / 2} />;
    default:
      return <rect {...common} x={x} y={y} width={w} height={h} rx={6} />;
  }
}

/** A straight line between two boxes, or a curve when the edge goes back up the page. */
function path(from: Placed, to: Placed, back: boolean): string {
  const x1 = from.x + from.width / 2;
  const y1 = from.y + from.height;
  const x2 = to.x + to.width / 2;
  const y2 = to.y;

  if (!back) return `M ${x1} ${y1} L ${x2} ${y2}`;
  // Out of the side and round, so a loop is visibly a loop rather than an arrow pointing the wrong
  // way through the middle of the diagram.
  const sx = from.x + from.width;
  const sy = from.y + from.height / 2;
  const ex = to.x + to.width;
  const ey = to.y + to.height / 2;
  const bulge = Math.max(40, Math.abs(sy - ey) * 0.4);
  return `M ${sx} ${sy} C ${sx + bulge} ${sy}, ${ex + bulge} ${ey}, ${ex} ${ey}`;
}

/** Long labels are cut rather than allowed to run across the diagram. */
function clip(text: string, max = 16): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}
