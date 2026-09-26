/**
 * Phase 16.2.3 - finding the page and flattening it, in about three hundred lines and no dependency.
 *
 * Every model in this repository is trained on **flat scans**. A phone held over paper adds
 * perspective, and perspective is not a nuisance the detector can shrug off: a rectangle becomes a
 * trapezoid, so a box's aspect ratio changes with where on the page it sits, and 4.1.5 already
 * established that global aspect is the strongest single feature the classifier has. Straightening
 * before upload is the row's word - **mandatory, not optional**.
 *
 * ## Why not OpenCV.js
 *
 * The plan says "OpenCV contour + perspective transform", and OpenCV.js is **8 to 10 MB of wasm**.
 * 16.3.1 makes this app installable with an offline shell, and an offline shell whose first useful
 * screen is a ten-megabyte download is not one. The subset actually needed here is small - a
 * threshold, one connected component, four corners, one homography, one bilinear resample - and it
 * is written out below at about 4 kB.
 *
 * **What is given up is real and is stated**: `findContours` with `approxPolyDP` is more robust than
 * corner extremes on a cluttered background, and this will lose a page lying on a patterned desk
 * where OpenCV would find it. The mitigation is not cleverness, it is **refusing**: when the
 * detection does not look like a page, `detect` returns `null` and the raw frame is uploaded
 * unchanged. A dewarp that quietly straightens the wrong quadrilateral is far worse than none,
 * because it crops the diagram and nothing downstream can tell.
 *
 * ## The pipeline
 *
 *   downscale    to 480 on the long edge. Detail is noise for this; the corners are found on a
 *                thumbnail and scaled back, which is also what makes it fast enough to run per
 *                preview frame
 *   luma         Rec. 601. Paper and ink differ in luminance, not in hue
 *   blur         3x3 box, twice - a cheap approximation of a Gaussian, enough to stop the threshold
 *                chasing sensor noise and pen strokes
 *   Otsu         the threshold that maximises between-class variance. No magic number, and it
 *                adapts to a dim room and a bright one without being told which it is
 *   component    the largest 4-connected region of the *bright* class, by union-find. The page
 *   corners      extremes of x+y, x-y and their negatives - the classic four-corner trick for a
 *                roughly rectangular blob, and exact for any true rectangle under perspective
 *   sanity       coverage, convexity, corner angles, aspect, and **edge support** - see below.
 *                Any failure is a `null`
 *
 * ## The gate that was wrong, and the one that replaced it
 *
 * The first version rejected a blob that did not *fill* the quad its own corners enclose, reasoning
 * that an L-shape encloses far more than it occupies. Measured on the benchmark set, that gate
 * rejected **three of the five real pages**: ink is dark, so it is not in the bright region, so a
 * page with more drawing on it scored lower. `wireframe` is 12.3% ink and scored **0.518** against
 * a 0.85 bar; `flowchart` and `er_diagram` scored 0.85 and 0.82. The metric was measuring how
 * little had been drawn on the page.
 *
 * What replaced it is `edgeSupport`, which asks the question directly: walk each edge of the quad
 * and require the blob to be present just inside and absent just outside. A real page's four edges
 * are genuine boundaries and score **1.00** on every page in the benchmark set, ink or no ink. An
 * L-shape's diagonal cuts through its own interior, where "outside" is blob too, and fails.
 *   homography   an 8x8 solve mapping the output rectangle onto the found quad
 *   resample     inverse-mapped bilinear, so the output has no holes
 */

export interface Point {
  x: number;
  y: number;
}

/** Four corners, always ordered top-left, top-right, bottom-right, bottom-left. */
export type Quad = [Point, Point, Point, Point];

export interface Detection {
  quad: Quad;
  /** 0..1. How much of the frame the page covers - shown in the viewfinder, not used as a gate. */
  coverage: number;
  /** How far from a rectangle, 0 is perfect. The worst corner's deviation from 90 degrees, in degrees. */
  skew: number;
}

/** Long edge of the thumbnail the detection runs on. */
export const WORK_EDGE = 480;

/** A page smaller than this fraction of the frame is probably not the page. */
export const MIN_COVERAGE = 0.12;

/**
 * Above this the quad is the whole frame, which means no background is visible - so either the page
 * fills the shot (nothing to correct) or the threshold found the frame itself. Either way, refuse:
 * warping an image onto its own corners is a no-op that costs a resample.
 */
export const MAX_COVERAGE = 0.985;

/** A corner further than this from a right angle is not a page seen at a plausible angle. */
export const MAX_CORNER_DEVIATION = 40;

/** Beyond this the quad is a sliver, not a sheet. */
export const MAX_ASPECT = 6;

/**
 * Fraction of each edge that must be a real boundary of the blob.
 *
 * 0.6 rather than something near 1: a genuine page edge loses samples to blur at the corners, to a
 * finger holding the sheet down, and to a shadow that merges the paper into the desk for a
 * centimetre. A false edge cutting through the blob's own interior scores far below this.
 */
export const MIN_EDGE_SUPPORT = 0.6;

// == the greyscale thumbnail ==================================================================

interface Gray {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  /** What the thumbnail was divided by, so corners scale back to the full frame. */
  scale: number;
}

/** Downscale by box-averaging and convert to luma in one pass. */
export function toGray(image: ImageData, longEdge = WORK_EDGE): Gray {
  const factor = Math.max(1, Math.max(image.width, image.height) / longEdge);
  const width = Math.max(1, Math.round(image.width / factor));
  const height = Math.max(1, Math.round(image.height / factor));
  const out = new Uint8ClampedArray(width * height);
  const step = Math.max(1, Math.floor(factor));

  for (let y = 0; y < height; y++) {
    const sy = Math.min(image.height - 1, Math.floor(y * factor));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(image.width - 1, Math.floor(x * factor));
      let sum = 0;
      let count = 0;
      // Box-average the source block rather than point-sampling it. Point sampling aliases a
      // pen stroke into or out of existence depending on where the grid lands, and the threshold
      // downstream is sensitive to exactly that.
      for (let dy = 0; dy < step; dy++) {
        const yy = Math.min(image.height - 1, sy + dy);
        for (let dx = 0; dx < step; dx++) {
          const xx = Math.min(image.width - 1, sx + dx);
          const i = (yy * image.width + xx) * 4;
          // Rec. 601 luma. Paper and ink differ in luminance, not in hue.
          sum += 0.299 * image.data[i] + 0.587 * image.data[i + 1] + 0.114 * image.data[i + 2];
          count++;
        }
      }
      out[y * width + x] = sum / count;
    }
  }
  return { data: out, width, height, scale: factor };
}

/** 3x3 box blur, twice. Cheap, separable, and close enough to a Gaussian for a threshold. */
export function blur(gray: Gray): Gray {
  let data = gray.data;
  for (let pass = 0; pass < 2; pass++) {
    data = boxPass(data, gray.width, gray.height);
  }
  return { ...gray, data };
}

function boxPass(src: Uint8ClampedArray, width: number, height: number): Uint8ClampedArray {
  const mid = new Uint8ClampedArray(src.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = src[y * width + Math.max(0, x - 1)];
      const b = src[y * width + x];
      const c = src[y * width + Math.min(width - 1, x + 1)];
      mid[y * width + x] = (a + b + c) / 3;
    }
  }
  const out = new Uint8ClampedArray(src.length);
  for (let y = 0; y < height; y++) {
    const up = Math.max(0, y - 1) * width;
    const down = Math.min(height - 1, y + 1) * width;
    for (let x = 0; x < width; x++) {
      out[y * width + x] = (mid[up + x] + mid[y * width + x] + mid[down + x]) / 3;
    }
  }
  return out;
}

/**
 * Otsu's threshold: the grey level that maximises between-class variance.
 *
 * No magic number. A fixed threshold works in one room and fails in the next; this one adapts to a
 * dim desk lamp and a bright window without being told which it is looking at.
 */
export function otsu(gray: Gray): number {
  const histogram = new Float64Array(256);
  for (const value of gray.data) histogram[value]++;
  const total = gray.data.length;

  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * histogram[i];

  let sumBackground = 0;
  let weightBackground = 0;
  let best = 0;
  let threshold = 127;
  for (let t = 0; t < 256; t++) {
    weightBackground += histogram[t];
    if (weightBackground === 0) continue;
    const weightForeground = total - weightBackground;
    if (weightForeground === 0) break;
    sumBackground += t * histogram[t];
    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sum - sumBackground) / weightForeground;
    const between = weightBackground * weightForeground * (meanBackground - meanForeground) ** 2;
    if (between > best) {
      best = between;
      threshold = t;
    }
  }
  return threshold;
}

// == the page ==================================================================================

/**
 * The largest 4-connected region brighter than `threshold`, as a label mask.
 *
 * Union-find over a single raster pass, which is one allocation and no recursion - a flood fill
 * recurses to the depth of the region and blows the stack on a full-frame page.
 */
function largestBrightRegion(
  gray: Gray,
  threshold: number,
): { labels: Int32Array; label: number; size: number } {
  const { data, width, height } = gray;
  const labels = new Int32Array(width * height).fill(-1);
  const parent: number[] = [];

  const find = (a: number): number => {
    let root = a;
    while (parent[root] !== root) root = parent[root];
    // Path compression, because a long page produces a long chain and this runs per preview frame.
    let walk = a;
    while (parent[walk] !== root) {
      const next = parent[walk];
      parent[walk] = root;
      walk = next;
    }
    return root;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (data[i] <= threshold) continue;
      const up = y > 0 && data[i - width] > threshold ? labels[i - width] : -1;
      const left = x > 0 && data[i - 1] > threshold ? labels[i - 1] : -1;
      if (up < 0 && left < 0) {
        const label = parent.length;
        parent.push(label);
        labels[i] = label;
      } else if (up >= 0 && left >= 0) {
        labels[i] = Math.min(up, left);
        union(up, left);
      } else {
        labels[i] = up >= 0 ? up : left;
      }
    }
  }

  const sizes = new Map<number, number>();
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] < 0) continue;
    const root = find(labels[i]);
    labels[i] = root;
    sizes.set(root, (sizes.get(root) ?? 0) + 1);
  }

  let label = -1;
  let size = 0;
  for (const [candidate, count] of sizes) {
    if (count > size) {
      size = count;
      label = candidate;
    }
  }
  return { labels, label, size };
}

/**
 * Fill the holes in a component: any non-component pixel that cannot reach the image border.
 *
 * Ink is dark, so a drawing punches holes in the bright page. Usually that costs nothing - the
 * corners are on the outside. But ink *near an edge* pulls the extreme point inward, and the
 * benchmark caught it: `wireframe.png` is 12.3% ink with strokes close to the border, and its
 * corners came back **41 to 55 px out** while the other four pages were at 2 px.
 *
 * Filling first makes the mask the page rather than the page-minus-the-drawing, which is what the
 * corners are supposed to be found on. One border flood fill over the complement, iterative rather
 * than recursive because a full-frame page is a deep region.
 */
function fillHoles(labels: Int32Array, label: number, width: number, height: number): Uint8Array {
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < labels.length; i++) mask[i] = labels[i] === label ? 1 : 0;

  // Everything outside that touches the border.
  const outside = new Uint8Array(width * height);
  const stack: number[] = [];
  const push = (i: number) => {
    if (mask[i] || outside[i]) return;
    outside[i] = 1;
    stack.push(i);
  };
  for (let x = 0; x < width; x++) {
    push(x);
    push((height - 1) * width + x);
  }
  for (let y = 0; y < height; y++) {
    push(y * width);
    push(y * width + width - 1);
  }
  while (stack.length) {
    const i = stack.pop() as number;
    const x = i % width;
    const y = (i / width) | 0;
    if (x > 0) push(i - 1);
    if (x < width - 1) push(i + 1);
    if (y > 0) push(i - width);
    if (y < height - 1) push(i + width);
  }

  // Anything neither component nor reachable from the border is enclosed by the component.
  for (let i = 0; i < mask.length; i++) if (!mask[i] && !outside[i]) mask[i] = 1;
  return mask;
}

/**
 * Four corners from a blob, by extremes of the rotated axes.
 *
 * `x + y` is smallest at the top-left and largest at the bottom-right; `x - y` is smallest at the
 * bottom-left and largest at the top-right. For any true rectangle - including one seen under
 * perspective, since perspective preserves straight lines - these four extremes *are* the corners.
 * For a blob that is not a quadrilateral they are its extreme points, which is why the sanity checks
 * below exist.
 */
function cornersOf(mask: Uint8Array, width: number, height: number): Quad | null {
  let tl = -Infinity,
    br = -Infinity,
    tr = -Infinity,
    bl = -Infinity;
  let pTL: Point | null = null,
    pBR: Point | null = null,
    pTR: Point | null = null,
    pBL: Point | null = null;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue;
      const plus = x + y;
      const minus = x - y;
      if (-plus > tl) {
        tl = -plus;
        pTL = { x, y };
      }
      if (plus > br) {
        br = plus;
        pBR = { x, y };
      }
      if (minus > tr) {
        tr = minus;
        pTR = { x, y };
      }
      if (-minus > bl) {
        bl = -minus;
        pBL = { x, y };
      }
    }
  }
  if (!pTL || !pTR || !pBR || !pBL) return null;
  return [pTL, pTR, pBR, pBL];
}

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

/** The interior angle at `b`, in degrees. */
function angleAt(a: Point, b: Point, c: Point): number {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);
  if (mag === 0) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

/** Twice the signed area. Positive for a counter-clockwise turn in screen coordinates. */
function cross(a: Point, b: Point, c: Point): number {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

export function isConvex(quad: Quad): boolean {
  let positive = false;
  let negative = false;
  for (let i = 0; i < 4; i++) {
    const turn = cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]);
    if (turn > 0) positive = true;
    if (turn < 0) negative = true;
  }
  // A convex polygon turns the same way at every vertex. Both signs means it crosses itself, which
  // is what a bow-tie "quad" from a bad blob looks like.
  return !(positive && negative);
}

export function quadArea(quad: Quad): number {
  let area = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    area += a.x * b.y - b.x * a.y;
  }
  return Math.abs(area) / 2;
}

/**
 * How much of each edge of `quad` is a real boundary of the blob.
 *
 * Walks every edge and, at each sample, looks a few pixels along the inward and outward normals.
 * A boundary has blob on one side and not the other. The score is the **worst** edge's fraction,
 * not the mean: three good edges and one cutting through the interior is exactly the failure this
 * exists to catch, and a mean would let it through at 0.75.
 */
function edgeSupport(mask: Uint8Array, quad: Quad, width: number, height: number): number {
  const at = (x: number, y: number): boolean => {
    const ix = Math.round(x);
    const iy = Math.round(y);
    if (ix < 0 || iy < 0 || ix >= width || iy >= height) return false;
    return mask[iy * width + ix] === 1;
  };
  const centre = {
    x: (quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4,
    y: (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4,
  };

  let worst = 1;
  for (let e = 0; e < 4; e++) {
    const a = quad[e];
    const b = quad[(e + 1) % 4];
    const length = Math.hypot(b.x - a.x, b.y - a.y);
    const samples = Math.max(8, Math.min(64, Math.round(length / 3)));
    // The inward normal, chosen by which side the centre is on rather than by winding order - the
    // corner ordering is fixed but a caller could hand this any quad.
    let nx = -(b.y - a.y) / (length || 1);
    let ny = (b.x - a.x) / (length || 1);
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    if ((centre.x - mid.x) * nx + (centre.y - mid.y) * ny < 0) {
      nx = -nx;
      ny = -ny;
    }

    let good = 0;
    let counted = 0;
    // The ends are skipped: within a few pixels of a corner both normals leave the blob, and every
    // real page would be penalised for having corners.
    for (let s = 1; s < samples - 1; s++) {
      const u = s / (samples - 1);
      const px = a.x + (b.x - a.x) * u;
      const py = a.y + (b.y - a.y) * u;
      const reach = 3;
      const inside = at(px + nx * reach, py + ny * reach);
      const outside = at(px - nx * reach, py - ny * reach);
      counted++;
      if (inside && !outside) good++;
    }
    worst = Math.min(worst, counted ? good / counted : 0);
  }
  return worst;
}

/**
 * Find the page in a frame, or refuse.
 *
 * Returns `null` far more readily than it returns a wrong quad. See the module docstring: a dewarp
 * that straightens the wrong quadrilateral crops the diagram, and nothing downstream can tell.
 */
export function detect(image: ImageData): Detection | null {
  if (image.width < 32 || image.height < 32) return null;
  const gray = blur(toGray(image));
  const threshold = otsu(gray);
  const { labels, label, size } = largestBrightRegion(gray, threshold);
  if (label < 0) return null;

  if (size < 16) return null;

  // Holes filled before the corners are taken. See `fillHoles`: ink near an edge otherwise pulls
  // an extreme point inward, which is how `wireframe` came back 41 px out while every other page
  // was at 2 px.
  const mask = fillHoles(labels, label, gray.width, gray.height);
  const small = cornersOf(mask, gray.width, gray.height);
  if (!small) return null;
  if (!isConvex(small)) return null;

  // Coverage is the **quad's** share of the frame, not the bright pixel count's.
  //
  // This was `size / frame` and it was wrong for a reason worth keeping: ink is dark, so it is not
  // in the bright region, so a page with more drawing on it counted as a smaller page. Measured on
  // the benchmark set: `wireframe` is 12.3% ink and its bright-pixel coverage came out at 0.079 at
  // severe tilt - under the 0.12 floor - while the page plainly occupied a fifth of the frame. The
  // question "how much of the frame is the page" is geometric, so it is answered geometrically.
  const frame = gray.width * gray.height;
  const enclosed = quadArea(small);
  const coverage = enclosed / frame;
  if (coverage < MIN_COVERAGE || coverage > MAX_COVERAGE) return null;

  let worst = 0;
  for (let i = 0; i < 4; i++) {
    const deviation = Math.abs(90 - angleAt(small[(i + 3) % 4], small[i], small[(i + 1) % 4]));
    worst = Math.max(worst, deviation);
  }
  if (worst > MAX_CORNER_DEVIATION) return null;

  // The check that actually rejects a page-shaped thing that is not a page.
  //
  // Neither the fill ratio nor the corner angles catch an L. Worked through on the test case: an
  // L-shaped bright region's four extreme points enclose a quad it fills at **0.963**, and whose
  // worst corner is 24.7 degrees off square - both comfortably inside their gates. The quad is
  // plausible; it is simply not bounded by the blob.
  //
  // So the edges are checked directly: walk each one and require the blob to be present just
  // inside and absent just outside. A true page's four edges are real boundaries and score near 1;
  // the L's diagonal cuts through its own interior, where "outside" is blob too, and it fails.
  if (edgeSupport(mask, small, gray.width, gray.height) < MIN_EDGE_SUPPORT) return null;

  const top = distance(small[0], small[1]);
  const bottom = distance(small[3], small[2]);
  const left = distance(small[0], small[3]);
  const right = distance(small[1], small[2]);
  const width = Math.max(top, bottom);
  const height = Math.max(left, right);
  if (width < 24 || height < 24) return null;
  const aspect = Math.max(width / height, height / width);
  if (aspect > MAX_ASPECT) return null;

  const scale = gray.scale;
  const quad = small.map((p) => ({
    // +0.5 recentres the thumbnail pixel before scaling; without it every corner is biased half a
    // work-pixel towards the origin, which at a scale of 4 is two pixels of the real frame.
    x: Math.min(image.width - 1, (p.x + 0.5) * scale),
    y: Math.min(image.height - 1, (p.y + 0.5) * scale),
  })) as Quad;

  return { quad, coverage, skew: worst };
}

// == the warp ==================================================================================

/**
 * The 3x3 homography taking the four `from` points to the four `to` points.
 *
 * Eight unknowns (h33 is fixed at 1), two equations per correspondence, solved by Gaussian
 * elimination with partial pivoting. Partial pivoting is not decoration: without it a quad with a
 * corner at x=0 divides by zero on the first column.
 */
export function homography(from: Quad, to: Quad): number[] | null {
  const a: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = from[i];
    const { x: u, y: v } = to[i];
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }

  for (let col = 0; col < 8; col++) {
    let pivot = col;
    for (let row = col + 1; row < 8; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-10) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    [b[col], b[pivot]] = [b[pivot], b[col]];

    for (let row = 0; row < 8; row++) {
      if (row === col) continue;
      const factor = a[row][col] / a[col][col];
      if (factor === 0) continue;
      for (let k = col; k < 8; k++) a[row][k] -= factor * a[col][k];
      b[row] -= factor * b[col];
    }
  }

  const h = new Array(9);
  for (let i = 0; i < 8; i++) h[i] = b[i] / a[i][i];
  h[8] = 1;
  return h;
}

export function apply(h: number[], point: Point): Point {
  const w = h[6] * point.x + h[7] * point.y + h[8];
  if (w === 0) return { x: 0, y: 0 };
  return {
    x: (h[0] * point.x + h[1] * point.y + h[2]) / w,
    y: (h[3] * point.x + h[4] * point.y + h[5]) / w,
  };
}

/**
 * The output size for a quad: the longer of each pair of opposite edges.
 *
 * Taking the longer rather than the average keeps the far edge of a tilted page at full resolution
 * instead of throwing away the half of the sheet that was closest to the lens.
 */
export function outputSize(quad: Quad): { width: number; height: number } {
  const width = Math.max(distance(quad[0], quad[1]), distance(quad[3], quad[2]));
  const height = Math.max(distance(quad[0], quad[3]), distance(quad[1], quad[2]));
  return { width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(height)) };
}

/**
 * Flatten `quad` out of `image` into a new `ImageData`.
 *
 * **Inverse mapped**: for each output pixel, find where it came from and sample. Forward mapping -
 * for each input pixel, find where it lands - leaves holes wherever the transform expands, and a
 * perspective correction expands the far half of the page by construction.
 *
 * Bilinear rather than nearest. The subject is pen strokes one or two pixels wide and nearest
 * neighbour turns a smooth line into a staircase, which is precisely the signal 9.1's detector and
 * 9.3's recogniser are reading.
 */
export function warp(image: ImageData, quad: Quad, size?: { width: number; height: number }): ImageData | null {
  const { width, height } = size ?? outputSize(quad);
  const destination: Quad = [
    { x: 0, y: 0 },
    { x: width - 1, y: 0 },
    { x: width - 1, y: height - 1 },
    { x: 0, y: height - 1 },
  ];
  // Solved in the inverse direction directly - destination to source - rather than solving forwards
  // and inverting a 3x3 by hand.
  const h = homography(destination, quad);
  if (!h) return null;

  const out = new ImageData(width, height);
  const src = image.data;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = apply(h, { x, y });
      const i = (y * width + x) * 4;
      if (p.x < 0 || p.y < 0 || p.x > image.width - 1 || p.y > image.height - 1) {
        // Outside the source. White rather than transparent black: the models expect paper, and a
        // black margin is a huge dark region for a threshold to find.
        out.data[i] = out.data[i + 1] = out.data[i + 2] = 255;
        out.data[i + 3] = 255;
        continue;
      }
      const x0 = Math.floor(p.x);
      const y0 = Math.floor(p.y);
      const x1 = Math.min(image.width - 1, x0 + 1);
      const y1 = Math.min(image.height - 1, y0 + 1);
      const fx = p.x - x0;
      const fy = p.y - y0;
      for (let c = 0; c < 3; c++) {
        const tl = src[(y0 * image.width + x0) * 4 + c];
        const tr = src[(y0 * image.width + x1) * 4 + c];
        const bl = src[(y1 * image.width + x0) * 4 + c];
        const br = src[(y1 * image.width + x1) * 4 + c];
        out.data[i + c] =
          tl * (1 - fx) * (1 - fy) + tr * fx * (1 - fy) + bl * (1 - fx) * fy + br * fx * fy;
      }
      out.data[i + 3] = 255;
    }
  }
  return out;
}

/** Detect and flatten in one call, or return `null` if there is no page worth flattening. */
export function straighten(image: ImageData): { image: ImageData; detection: Detection } | null {
  const detection = detect(image);
  if (!detection) return null;
  const flattened = warp(image, detection.quad);
  if (!flattened) return null;
  return { image: flattened, detection };
}
