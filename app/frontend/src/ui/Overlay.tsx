/**
 * Phase 16.2.5 - the detection, drawn back onto the page it came from.
 *
 * This is the screen that makes the pipeline legible. A list of node ids tells a person nothing;
 * the same nodes outlined on their own handwriting tells them immediately whether the reader saw
 * what they drew - which box was missed, which arrow went to the wrong place, which word came back
 * as something else.
 *
 * ## It draws on the image the server was given, not the one the camera saw
 *
 * Every bbox in the IR is in the pixel space of the **uploaded** image, which after 16.2.3 is the
 * dewarped crop rather than the raw frame. Drawing on the raw photograph would put every box a few
 * per cent out and leave a person adjusting labels that are in fact correct.
 *
 * ## The photograph is not on the server, and that is the design
 *
 * 16.1.1 stores the result and not the picture, and the About screen promises exactly that. So the
 * overlay works from the blob this browser still holds, and on a reload or a shared link there is
 * no photograph to draw on. That case is **said** rather than papered over: the graph view (16.2.6)
 * is the same information without the picture, and the alternative - keeping every photograph a
 * stranger uploads - is a privacy position this project has not earned.
 *
 * ## Canvas, not SVG
 *
 * An SVG overlay would be less code and is the wrong tool here. A dense page is a hundred boxes and
 * polylines; as SVG that is a hundred DOM nodes re-laid-out on every pinch-zoom, on a phone. One
 * canvas redraws in a frame. What SVG would have bought - hit testing for free - is thirty lines of
 * point-in-rectangle, and the hit test has to be nearest-first anyway so that overlapping boxes
 * select the smaller one.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { Diagram, Node } from "../lib/api";
import { LOW_CONFIDENCE } from "../lib/doubts";
import "./Overlay.css";

export interface OverlayProps {
  /** The image the server was given. `null` when this session no longer holds it. */
  src: string | null;
  ir: Diagram | null;
  /** 10.1's reading order, drawn as a number on each node when shown. */
  traversal?: string[];
  selected?: string | null;
  onSelect?: (id: string | null) => void;
  /** 16.2.9 turns this on; until then the boxes are all one colour. */
  flagLowConfidence?: boolean;
}

// The threshold lives in `lib/doubts.ts` and is re-exported here for the screens that were reading
// it from this module. One number, so the dashed orange box on the picture and the row in the
// "where to look" list are the same claim rather than two thresholds that happen to agree.
export { LOW_CONFIDENCE } from "../lib/doubts";

interface Layout {
  /** Image pixels to canvas pixels. */
  scale: number;
  offsetX: number;
  offsetY: number;
}

export function Overlay({
  src,
  ir,
  traversal = [],
  selected = null,
  onSelect,
  flagLowConfidence = false,
}: OverlayProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [layout, setLayout] = useState<Layout | null>(null);
  /**
   * A `src` that was given and could not be loaded.
   *
   * Added in 16.2.10, and it is a hole this component had from the start: with only an `onload`
   * handler, an image that fails leaves `image` at `null` forever and the canvas is **blank with no
   * explanation** - indistinguishable from a reading that found nothing. Two ways in: 16.2.10 can
   * now pass `examples/<file>.png`, which is a real request that a real lack of network can refuse;
   * and a held object URL can be revoked under a screen that is still mounted.
   */
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!src) {
      setImage(null);
      setFailed(false);
      return;
    }
    const element = new Image();
    let live = true;
    setFailed(false);
    element.onload = () => live && setImage(element);
    element.onerror = () => live && setFailed(true);
    element.src = src;
    return () => {
      live = false;
    };
  }, [src]);

  const draw = useCallback(() => {
    const surface = canvas.current;
    const box = wrap.current;
    if (!surface || !box || !image) return;

    const rect = box.getBoundingClientRect();
    if (!rect.width) return;
    // Capped at 2. A phone at dpr 3 would otherwise allocate a canvas nine times the CSS area for
    // a gain nobody can see on a 1 px stroke.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const scale = rect.width / image.width;
    const height = image.height * scale;

    surface.width = Math.round(rect.width * dpr);
    surface.height = Math.round(height * dpr);
    surface.style.height = `${height}px`;

    const context = surface.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, rect.width, height);
    context.drawImage(image, 0, 0, rect.width, height);

    // A wash over the photograph so gold and emerald read against pencil on white paper. Without it
    // a pale stroke on a bright page is invisible, which defeats the point of the screen.
    context.fillStyle = "rgba(8, 8, 10, 0.34)";
    context.fillRect(0, 0, rect.width, height);

    setLayout({ scale, offsetX: 0, offsetY: 0 });
    if (!ir) return;

    const order = new Map(traversal.map((id, index) => [id, index + 1]));
    const at = (x: number, y: number) => ({ x: x * scale, y: y * scale });

    // Edges first, so a box sits on top of the arrow that reaches it rather than under it.
    for (const edge of ir.edges ?? []) {
      const points = edge.polyline ?? [];
      if (points.length < 2) continue;
      const low = flagLowConfidence && (edge.confidence ?? 1) < LOW_CONFIDENCE;
      context.beginPath();
      points.forEach(([x, y], i) => {
        const p = at(x, y);
        if (i === 0) context.moveTo(p.x, p.y);
        else context.lineTo(p.x, p.y);
      });
      context.strokeStyle = low ? "#fb923c" : "#e8c36a";
      context.lineWidth = low ? 2.5 : 2;
      if (low) context.setLineDash([6, 4]);
      context.stroke();
      context.setLineDash([]);

      if (edge.directed !== false) {
        const [ax, ay] = points[points.length - 2];
        const [bx, by] = points[points.length - 1];
        arrowhead(context, at(ax, ay), at(bx, by), low ? "#fb923c" : "#e8c36a");
      }
    }

    for (const node of ir.nodes ?? []) {
      const bbox = node.bbox as number[] | undefined;
      if (!bbox || bbox.length !== 4) continue;
      const [x, y, w, h] = bbox.map((v) => v * scale);
      const low =
        flagLowConfidence && node.corrected !== true && (node.confidence ?? 1) < LOW_CONFIDENCE;
      const isSelected = node.id === selected;

      context.lineWidth = isSelected ? 3 : 2;
      context.strokeStyle = isSelected ? "#ffffff" : low ? "#fb923c" : "#34d399";
      context.fillStyle = isSelected ? "rgba(255,255,255,0.14)" : "rgba(52, 211, 153, 0.10)";
      if (low && !isSelected) {
        context.fillStyle = "rgba(251, 146, 60, 0.12)";
        context.setLineDash([6, 4]);
      }
      roundRect(context, x, y, w, h, 4);
      context.fill();
      context.stroke();
      context.setLineDash([]);

      const step = order.get(node.id);
      if (step) badge(context, x, y, String(step));
      if (node.text) label(context, node, x, y, w, h, isSelected, low);
    }
  }, [image, ir, traversal, selected, flagLowConfidence]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    // Redraw on resize, because the canvas is sized from its container and a rotation changes it.
    const observer = new ResizeObserver(() => draw());
    if (wrap.current) observer.observe(wrap.current);
    return () => observer.disconnect();
  }, [draw]);

  /**
   * Hit test, smallest box first.
   *
   * A diagram nests: a label inside a box inside a swimlane. Taking the first hit in document order
   * selects the biggest thing under the finger, which is never what was aimed at.
   */
  const tap = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onSelect || !layout || !ir) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const x = (event.clientX - rect.left) / layout.scale;
      const y = (event.clientY - rect.top) / layout.scale;
      let hit: Node | null = null;
      let area = Infinity;
      for (const node of ir.nodes ?? []) {
        const bbox = node.bbox as number[] | undefined;
        if (!bbox || bbox.length !== 4) continue;
        const [bx, by, bw, bh] = bbox;
        if (x < bx || y < by || x > bx + bw || y > by + bh) continue;
        if (bw * bh < area) {
          area = bw * bh;
          hit = node;
        }
      }
      onSelect(hit ? hit.id : null);
    },
    [ir, layout, onSelect],
  );

  if (failed) {
    return (
      <div className="overlay-absent">
        <p>
          <strong>The picture could not be loaded.</strong>
        </p>
        <p className="muted">
          The reading itself is here — the graph and the code views show the same answer without it.
          With no network a bundled example&apos;s image has to come out of the cache, and this one
          did not.
        </p>
      </div>
    );
  }

  if (!src) {
    return (
      <div className="overlay-absent">
        <p>
          <strong>The photograph is not held here.</strong>
        </p>
        <p className="muted">
          The server keeps what it read — the shapes, the arrows, the code — and not the picture. On
          a reload or a shared link there is nothing to draw on, so the graph view shows the same
          reading without it.
        </p>
      </div>
    );
  }

  return (
    <div className="overlay" ref={wrap}>
      <canvas ref={canvas} onClick={tap} aria-label="The page, with what was detected drawn on it" />
    </div>
  );
}

function roundRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const radius = Math.min(r, w / 2, h / 2);
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + w, y, x + w, y + h, radius);
  context.arcTo(x + w, y + h, x, y + h, radius);
  context.arcTo(x, y + h, x, y, radius);
  context.arcTo(x, y, x + w, y, radius);
  context.closePath();
}

function arrowhead(
  context: CanvasRenderingContext2D,
  from: { x: number; y: number },
  to: { x: number; y: number },
  colour: string,
) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const size = 9;
  context.beginPath();
  context.moveTo(to.x, to.y);
  context.lineTo(to.x - size * Math.cos(angle - Math.PI / 7), to.y - size * Math.sin(angle - Math.PI / 7));
  context.lineTo(to.x - size * Math.cos(angle + Math.PI / 7), to.y - size * Math.sin(angle + Math.PI / 7));
  context.closePath();
  context.fillStyle = colour;
  context.fill();
}

/** 10.1's reading order, as a numbered disc on the node's corner. */
function badge(context: CanvasRenderingContext2D, x: number, y: number, text: string) {
  context.beginPath();
  context.arc(x, y, 9, 0, Math.PI * 2);
  context.fillStyle = "#e8c36a";
  context.fill();
  context.fillStyle = "#100d04";
  context.font = "600 11px ui-sans-serif, system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, x, y + 0.5);
}

/**
 * The recognised text, under its box.
 *
 * Under and not inside: inside covers the handwriting, and the whole question a person is answering
 * on this screen is whether the machine read the handwriting correctly. Covering the evidence with
 * the claim makes the screen useless.
 */
function label(
  context: CanvasRenderingContext2D,
  node: Node,
  x: number,
  y: number,
  w: number,
  h: number,
  isSelected: boolean,
  low: boolean,
) {
  const text = String(node.text ?? "");
  if (!text) return;
  context.font = "600 12px ui-sans-serif, system-ui, sans-serif";
  const width = Math.min(context.measureText(text).width + 12, Math.max(w, 90));
  const bx = x + (w - width) / 2;
  const by = y + h + 4;

  context.fillStyle = isSelected ? "#ffffff" : low ? "#fb923c" : "rgba(8,8,10,0.86)";
  roundRect(context, bx, by, width, 18, 9);
  context.fill();

  context.fillStyle = isSelected || low ? "#100d04" : "#f4f4f7";
  context.textAlign = "center";
  context.textBaseline = "middle";
  // Clipped to its own pill, so a long label cannot run across the page.
  context.save();
  roundRect(context, bx, by, width, 18, 9);
  context.clip();
  context.fillText(text, bx + width / 2, by + 9.5);
  context.restore();
}
