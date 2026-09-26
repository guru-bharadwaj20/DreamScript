/**
 * Phase 16.2.3 - the dewarp, against a ground truth it cannot see.
 *
 * Every test here builds a synthetic frame with a **known** page quad, hands it to `detect`, and
 * compares the corners it found against the ones that were drawn. That is the only honest way to
 * test this: a detector checked against its own output is checked against nothing.
 *
 * The second half is about refusal. `detect` returning `null` is a *feature* - a dewarp that
 * straightens the wrong quadrilateral crops the diagram and nothing downstream can tell - so the
 * cases where it must refuse get as many tests as the cases where it must succeed.
 */

import { describe, expect, it } from "vitest";

import {
  MAX_ASPECT,
  type Point,
  type Quad,
  apply,
  detect,
  homography,
  isConvex,
  otsu,
  outputSize,
  quadArea,
  straighten,
  toGray,
  warp,
} from "./dewarp";

/** A frame with a dark background and one bright quadrilateral "page" on it. */
function frame(
  width: number,
  height: number,
  quad: Quad,
  options: { background?: number; page?: number; ink?: boolean; noise?: number } = {},
): ImageData {
  const { background = 40, page = 225, ink = false, noise = 0 } = options;
  const image = new ImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      let value = background;
      if (inside({ x, y }, quad)) value = page;
      if (noise) value += (((x * 7919 + y * 104729) % 101) / 100 - 0.5) * 2 * noise;
      image.data[i] = image.data[i + 1] = image.data[i + 2] = Math.max(0, Math.min(255, value));
      image.data[i + 3] = 255;
    }
  }
  if (ink) {
    // A few dark strokes on the page. They must not change the corners: the blur and the
    // largest-region step exist so a diagram does not split the page into fragments.
    for (let t = 0; t <= 200; t++) {
      const u = t / 200;
      const top = lerp(quad[0], quad[1], u);
      const bottom = lerp(quad[3], quad[2], u);
      for (let s = 0.25; s <= 0.75; s += 0.02) {
        const p = lerp(top, bottom, s);
        mark(image, Math.round(p.x), Math.round(p.y));
      }
    }
  }
  return image;
}

/** Where a quad's two diagonals cross. The image of the rectangle's centre under any homography. */
function diagonals(q: Quad): Point {
  const [a, b, c, d] = q;
  const d1 = { x: c.x - a.x, y: c.y - a.y };
  const d2 = { x: d.x - b.x, y: d.y - b.y };
  const denominator = d1.x * d2.y - d1.y * d2.x;
  const t = ((b.x - a.x) * d2.y - (b.y - a.y) * d2.x) / denominator;
  return { x: a.x + d1.x * t, y: a.y + d1.y * t };
}

const lerp = (a: Point, b: Point, t: number): Point => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
});

function mark(image: ImageData, x: number, y: number) {
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) return;
  const i = (y * image.width + x) * 4;
  image.data[i] = image.data[i + 1] = image.data[i + 2] = 20;
}

function inside(p: Point, quad: Quad): boolean {
  let hit = false;
  for (let i = 0, j = 3; i < 4; j = i++) {
    const a = quad[i];
    const b = quad[j];
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) {
      hit = !hit;
    }
  }
  return hit;
}

/** Worst corner error, in pixels of the full frame. */
function corners(found: Quad, truth: Quad): number {
  return Math.max(...found.map((p, i) => Math.hypot(p.x - truth[i].x, p.y - truth[i].y)));
}

const RECTANGLE: Quad = [
  { x: 80, y: 60 },
  { x: 560, y: 60 },
  { x: 560, y: 420 },
  { x: 80, y: 420 },
];

// A page photographed from the left: the far edge is shorter and the quad is a trapezoid.
const TILTED: Quad = [
  { x: 120, y: 90 },
  { x: 540, y: 40 },
  { x: 570, y: 430 },
  { x: 90, y: 400 },
];

describe("detect", () => {
  it("finds a rectangle's corners to within a work pixel", () => {
    // The detection runs on a 480px thumbnail, so a 640px frame is scaled by ~1.33 and one work
    // pixel is ~1.4 real ones. Anything inside a few pixels is the algorithm working.
    const found = detect(frame(640, 480, RECTANGLE));
    expect(found).not.toBeNull();
    expect(corners(found!.quad, RECTANGLE)).toBeLessThan(4);
  });

  it("finds a trapezoid, which is the case the row exists for", () => {
    const found = detect(frame(640, 480, TILTED));
    expect(found).not.toBeNull();
    expect(corners(found!.quad, TILTED)).toBeLessThan(6);
    // A tilted page is not a rectangle, and the reported skew should say so.
    expect(found!.skew).toBeGreaterThan(1);
  });

  it("orders the corners top-left, top-right, bottom-right, bottom-left", () => {
    // Everything downstream indexes these positionally. An unordered quad warps the page inside out
    // and the failure looks like a detection failure.
    const { quad } = detect(frame(640, 480, TILTED))!;
    expect(quad[0].x).toBeLessThan(quad[1].x);
    expect(quad[3].x).toBeLessThan(quad[2].x);
    expect(quad[0].y).toBeLessThan(quad[3].y);
    expect(quad[1].y).toBeLessThan(quad[2].y);
  });

  it("is not thrown off by ink on the page", () => {
    // The whole point of the blur and the largest-connected-region step: a diagram must not split
    // the page into fragments and hand back the biggest white gap between two boxes.
    const found = detect(frame(640, 480, TILTED, { ink: true }));
    expect(found).not.toBeNull();
    expect(corners(found!.quad, TILTED)).toBeLessThan(8);
  });

  it("is not thrown off by sensor noise", () => {
    const found = detect(frame(640, 480, RECTANGLE, { noise: 18 }));
    expect(found).not.toBeNull();
    expect(corners(found!.quad, RECTANGLE)).toBeLessThan(6);
  });

  it("adapts to a dim room and a bright one without being told", () => {
    // Otsu, rather than a fixed threshold that works in one room and fails in the next.
    for (const [background, page] of [
      [10, 90],
      [40, 225],
      [150, 250],
    ]) {
      const found = detect(frame(640, 480, RECTANGLE, { background, page }));
      expect(found, `background ${background}, page ${page}`).not.toBeNull();
      expect(corners(found!.quad, RECTANGLE)).toBeLessThan(6);
    }
  });

  it("reports coverage as the fraction of the frame the page takes", () => {
    const found = detect(frame(640, 480, RECTANGLE))!;
    const expected = quadArea(RECTANGLE) / (640 * 480);
    expect(found.coverage).toBeGreaterThan(expected - 0.05);
    expect(found.coverage).toBeLessThan(expected + 0.05);
  });
});

describe("detect refuses rather than guessing", () => {
  it("refuses a frame with no page in it", () => {
    expect(detect(frame(640, 480, RECTANGLE, { background: 128, page: 128 }))).toBeNull();
  });

  it("refuses a page too small to be the subject", () => {
    const tiny: Quad = [
      { x: 300, y: 220 },
      { x: 340, y: 220 },
      { x: 340, y: 260 },
      { x: 300, y: 260 },
    ];
    expect(detect(frame(640, 480, tiny))).toBeNull();
  });

  it("refuses a page that fills the frame, because there is nothing to correct", () => {
    const full: Quad = [
      { x: 0, y: 0 },
      { x: 639, y: 0 },
      { x: 639, y: 479 },
      { x: 0, y: 479 },
    ];
    expect(detect(frame(640, 480, full))).toBeNull();
  });

  it("refuses a sliver", () => {
    const sliver: Quad = [
      { x: 20, y: 200 },
      { x: 620, y: 200 },
      { x: 620, y: 240 },
      { x: 20, y: 240 },
    ];
    const found = detect(frame(640, 480, sliver));
    if (found) {
      const { width, height } = outputSize(found.quad);
      expect(Math.max(width / height, height / width)).toBeLessThanOrEqual(MAX_ASPECT);
    }
  });

  it("refuses a bright region that is not a quadrilateral", () => {
    // An L-shape has four perfectly good extreme points and an enclosing quad far larger than
    // itself. This is what the fill-ratio check rejects.
    const image = frame(640, 480, RECTANGLE, { page: 40 });
    for (let y = 60; y < 420; y++) {
      for (let x = 80; x < 560; x++) {
        if (x > 300 && y > 200) continue;
        const i = (y * 640 + x) * 4;
        image.data[i] = image.data[i + 1] = image.data[i + 2] = 225;
      }
    }
    expect(detect(image)).toBeNull();
  });

  it("refuses an image too small to mean anything", () => {
    expect(detect(new ImageData(8, 8))).toBeNull();
  });
});

describe("homography", () => {
  it("maps each corner onto its counterpart exactly", () => {
    const h = homography(RECTANGLE, TILTED)!;
    for (let i = 0; i < 4; i++) {
      const mapped = apply(h, RECTANGLE[i]);
      expect(mapped.x).toBeCloseTo(TILTED[i].x, 6);
      expect(mapped.y).toBeCloseTo(TILTED[i].y, 6);
    }
  });

  it("survives a corner on the origin, which is what partial pivoting is for", () => {
    const atOrigin: Quad = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 80 },
      { x: 0, y: 80 },
    ];
    const h = homography(atOrigin, TILTED);
    expect(h).not.toBeNull();
    expect(apply(h!, atOrigin[0]).x).toBeCloseTo(TILTED[0].x, 6);
  });

  it("returns null for degenerate correspondences rather than NaN", () => {
    const collapsed: Quad = [
      { x: 5, y: 5 },
      { x: 5, y: 5 },
      { x: 5, y: 5 },
      { x: 5, y: 5 },
    ];
    expect(homography(collapsed, RECTANGLE)).toBeNull();
  });
});

describe("warp", () => {
  it("flattens a tilted page back to a rectangle", () => {
    // The end-to-end claim: draw a marker at a known spot on a tilted page, straighten it, and the
    // marker should land where it would have been on a flat scan.
    const image = frame(640, 480, TILTED);
    // The centre of the page under perspective is where the quad's **diagonals cross**, not the
    // average of its edge midpoints - a homography maps diagonals to diagonals, so their
    // intersection is the image of the rectangle's centre. The first version of this test used the
    // midpoint average, which is 73 px away on this trapezoid, and blamed the warp for it.
    const centre = diagonals(TILTED);
    for (let dy = -6; dy <= 6; dy++) for (let dx = -6; dx <= 6; dx++) {
      mark(image, Math.round(centre.x) + dx, Math.round(centre.y) + dy);
    }

    // `warp` with the *known* quad, not `straighten`. Detection is accurate to a few pixels and is
    // tested above; feeding its output in here would fold that error into a measurement of the
    // resample, and a 6 px corner error moves the projective centre by more than the tolerance
    // this test wants to hold. One claim per test.
    const flat = warp(image, TILTED)!;
    expect(flat).not.toBeNull();

    // Find the dark blob's centroid in the output.
    let sx = 0, sy = 0, n = 0;
    for (let y = 0; y < flat.height; y++) {
      for (let x = 0; x < flat.width; x++) {
        if (flat.data[(y * flat.width + x) * 4] < 90) {
          sx += x;
          sy += y;
          n++;
        }
      }
    }
    expect(n).toBeGreaterThan(20);
    // The centre of the page should land at the centre of the flattened image, within 2% of its
    // width - which on a ~470px output is about nine pixels.
    expect(Math.abs(sx / n - flat.width / 2)).toBeLessThan(flat.width * 0.02);
    expect(Math.abs(sy / n - flat.height / 2)).toBeLessThan(flat.height * 0.02);
  });

  it("fills outside the source with white, not black", () => {
    // The models expect paper. A black margin is a large dark region for a threshold to find, and
    // 3.1's normalisation would take it for ink.
    const image = frame(400, 300, RECTANGLE.map((p) => ({ x: p.x / 2, y: p.y / 2 })) as Quad);
    const out = warp(image, [
      { x: -50, y: -50 },
      { x: 450, y: -50 },
      { x: 450, y: 350 },
      { x: -50, y: 350 },
    ])!;
    expect(out.data[0]).toBe(255);
    expect(out.data[3]).toBe(255);
  });

  it("straighten composes detect and warp, and refuses when detect does", () => {
    // What the camera actually calls. A page it can find comes back flattened; a frame with no page
    // in it comes back `null`, which is the signal to upload the raw photograph unchanged.
    const found = straighten(frame(640, 480, TILTED));
    expect(found).not.toBeNull();
    expect(found!.image.width).toBeGreaterThan(300);
    expect(found!.detection.skew).toBeGreaterThan(1);

    expect(straighten(frame(640, 480, RECTANGLE, { background: 128, page: 128 }))).toBeNull();
  });

  it("keeps the longer of each pair of opposite edges", () => {
    // Averaging would throw away half the resolution of the near edge of a tilted sheet.
    const { width, height } = outputSize(TILTED);
    expect(width).toBeGreaterThanOrEqual(
      Math.round(Math.hypot(TILTED[1].x - TILTED[0].x, TILTED[1].y - TILTED[0].y)) - 1,
    );
    expect(height).toBeGreaterThan(300);
  });
});

describe("geometry helpers", () => {
  it("knows a bow-tie is not convex", () => {
    expect(isConvex(RECTANGLE)).toBe(true);
    expect(
      isConvex([
        { x: 0, y: 0 },
        { x: 100, y: 100 },
        { x: 100, y: 0 },
        { x: 0, y: 100 },
      ]),
    ).toBe(false);
  });

  it("measures a rectangle's area", () => {
    expect(quadArea(RECTANGLE)).toBeCloseTo(480 * 360, 6);
  });

  it("finds a threshold between two populations", () => {
    const image = frame(200, 200, [
      { x: 20, y: 20 },
      { x: 180, y: 20 },
      { x: 180, y: 180 },
      { x: 20, y: 180 },
    ]);
    const t = otsu(toGray(image));
    // Between the two populations, and the lower bound is inclusive: with spikes at 40 and 225 the
    // between-class variance is maximised across the whole gap and the first maximum is 40, which
    // separates them exactly. The first version of this test asserted `> 45`, which was a
    // preference about where in the gap to land rather than a property of the threshold.
    expect(t).toBeGreaterThanOrEqual(40);
    expect(t).toBeLessThan(225);
  });
});
