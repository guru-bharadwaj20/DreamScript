/**
 * Phase 16.2.3 - build the measurement set, using the shipped dewarp.
 *
 * The row's Definition of Done is "dewarped crop measured against the raw photo on the same pages",
 * and the honest obstacle is that **this repository has no skewed photographs**. The corpora are
 * flat scans and the self-collected chaos corpus is not present in this checkout. So the skew is
 * synthesised, which is stated rather than glossed: each flat fixture is warped by a known
 * homography that simulates a phone held at an angle over the page, and that warped image is what
 * the pipeline is asked to read.
 *
 * Three images per page come out of this:
 *
 *   flat        the original. The baseline every other answer is compared against
 *   photo       flat, warped by a known homography, on a desk-coloured ground. What a phone sees
 *   dewarped    `photo` put through **the exact code the app ships** - `src/lib/dewarp.ts`, running
 *               in a real browser, reached through the built module rather than reimplemented here
 *
 * That last point is the reason this is a browser script rather than a Python one. A Python
 * reimplementation of the same algorithm would measure the reimplementation.
 *
 *     node scripts/dewarp_bench.mjs <out-dir> [dev-server-url]
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

/**
 * puppeteer is resolved at run time and is **not** a dependency of `app/frontend`.
 *
 * It is a ~200 MB install with a bundled Chromium, and `npm ci` in CI's `client` job would pay for
 * it on every run to build a bundle that does not need it. So it is brought in by whoever runs this
 * benchmark, and `PUPPETEER_PATH` points at it when it is not resolvable from here:
 *
 *     PUPPETEER_PATH=/path/to/node_modules/puppeteer/lib/esm/puppeteer/puppeteer.js  *       node scripts/dewarp_bench.mjs runs/dewarp
 */
const puppeteer = (await import(process.env.PUPPETEER_PATH || "puppeteer")).default;

const OUT = process.argv[2] ?? "runs/dewarp";
const BASE = process.argv[3] ?? "http://localhost:5173/";
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");

const PAGES = [
  "flowchart.png",
  "state_machine.png",
  "er_diagram.png",
  "wireframe.png",
  "circuit.png",
];

/**
 * Four tilts, from mild to severe, as fractional corner offsets.
 *
 * Expressed as a fraction of the page rather than in pixels so the same tilt means the same thing
 * on a 300px fixture and a 3000px photograph. `mild` is a page photographed slightly off-centre;
 * `severe` is someone leaning over a table.
 */
const TILTS = {
  mild: [
    [0.04, 0.02],
    [-0.03, 0.03],
    [-0.02, -0.02],
    [0.03, -0.03],
  ],
  moderate: [
    [0.1, 0.06],
    [-0.08, 0.09],
    [-0.06, -0.05],
    [0.09, -0.08],
  ],
  severe: [
    [0.18, 0.12],
    [-0.15, 0.17],
    [-0.12, -0.1],
    [0.16, -0.14],
  ],
};

mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
await page.goto(BASE, { waitUntil: "networkidle0", timeout: 30000 });

const results = [];

for (const name of PAGES) {
  const bytes = await readFile(path.join(ROOT, "tests", "fixtures", name));
  const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;

  for (const [tilt, offsets] of Object.entries(TILTS)) {
    const outcome = await page.evaluate(
      async (source, corners, margin) => {
        // The shipped module, imported from the dev server - not a copy of it.
        const { detect, warp, straighten } = await import("/src/lib/dewarp.ts");

        const image = new Image();
        image.src = source;
        await image.decode();

        // Lay the page on a desk-coloured ground with a margin, then push its corners outward by
        // the tilt. The margin is what makes a page *findable*: a photograph with no visible
        // background has no outline to detect, which the detector correctly refuses.
        const pad = Math.round(Math.max(image.width, image.height) * margin);
        const W = image.width + pad * 2;
        const H = image.height + pad * 2;

        const flatCanvas = document.createElement("canvas");
        flatCanvas.width = W;
        flatCanvas.height = H;
        const fc = flatCanvas.getContext("2d");
        // A mid-grey desk. Not black: a black ground makes the page trivially findable and would
        // flatter the detector.
        fc.fillStyle = "#6b6a66";
        fc.fillRect(0, 0, W, H);
        fc.drawImage(image, pad, pad);
        const flat = fc.getImageData(0, 0, W, H);

        // Where the page's four corners end up after the tilt.
        const base = [
          { x: pad, y: pad },
          { x: pad + image.width, y: pad },
          { x: pad + image.width, y: pad + image.height },
          { x: pad, y: pad + image.height },
        ];
        const tilted = base.map((p, i) => ({
          x: p.x + corners[i][0] * image.width,
          y: p.y + corners[i][1] * image.height,
        }));

        // `warp` maps the *destination rectangle* onto a quad in the source. To push the page out
        // to `tilted`, warp with the inverse quad - the region of a hypothetical larger frame that
        // maps back onto the flat page - so this is done by warping the flat image through the
        // homography that takes `tilted` to `base`.
        //
        // Concretely: build an output the size of the frame whose pixel (x, y) samples the flat
        // image at H(x, y), where H takes the tilted quad back to the page rectangle.
        const photoCanvas = document.createElement("canvas");
        photoCanvas.width = W;
        photoCanvas.height = H;
        const pc = photoCanvas.getContext("2d");
        pc.fillStyle = "#6b6a66";
        pc.fillRect(0, 0, W, H);
        const photo = pc.getImageData(0, 0, W, H);

        const { homography, apply } = await import("/src/lib/dewarp.ts");
        const h = homography(tilted, base);
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const s = apply(h, { x, y });
            const i = (y * W + x) * 4;
            if (s.x < pad || s.y < pad || s.x >= pad + image.width || s.y >= pad + image.height) {
              continue; // stays desk
            }
            const sx = Math.min(W - 1, Math.max(0, Math.round(s.x)));
            const sy = Math.min(H - 1, Math.max(0, Math.round(s.y)));
            const j = (sy * W + sx) * 4;
            photo.data[i] = flat.data[j];
            photo.data[i + 1] = flat.data[j + 1];
            photo.data[i + 2] = flat.data[j + 2];
            photo.data[i + 3] = 255;
          }
        }
        pc.putImageData(photo, 0, 0);

        // Now the shipped detection, on the photograph.
        const found = detect(photo);
        let dewarpedUrl = null;
        let corner_error = null;
        if (found) {
          const flattened = warp(photo, found.quad);
          if (flattened) {
            const dc = document.createElement("canvas");
            dc.width = flattened.width;
            dc.height = flattened.height;
            dc.getContext("2d").putImageData(flattened, 0, 0);
            dewarpedUrl = dc.toDataURL("image/png");
          }
          // Against the corners the tilt actually used - a ground truth the detector cannot see.
          corner_error = Math.max(
            ...found.quad.map((p, i) => Math.hypot(p.x - tilted[i].x, p.y - tilted[i].y)),
          );
        }

        return {
          width: W,
          height: H,
          flat: flatCanvas.toDataURL("image/png"),
          photo: photoCanvas.toDataURL("image/png"),
          dewarped: dewarpedUrl,
          detected: !!found,
          corner_error,
          coverage: found?.coverage ?? null,
          skew: found?.skew ?? null,
          truth_quad: tilted,
          found_quad: found?.quad ?? null,
        };
      },
      dataUrl,
      offsets,
      0.18,
    );

    const stem = `${name.replace(/\.png$/, "")}-${tilt}`;
    const save = (suffix, url) => {
      if (!url) return null;
      const file = path.join(OUT, `${stem}-${suffix}.png`);
      writeFileSync(file, Buffer.from(url.split(",")[1], "base64"));
      return path.relative(ROOT, file).replace(/\\/g, "/");
    };

    results.push({
      page: name,
      tilt,
      width: outcome.width,
      height: outcome.height,
      detected: outcome.detected,
      corner_error: outcome.corner_error,
      coverage: outcome.coverage,
      skew: outcome.skew,
      flat: save("flat", outcome.flat),
      photo: save("photo", outcome.photo),
      dewarped: save("dewarped", outcome.dewarped),
    });
    console.error(
      `  ${stem}: ${outcome.detected ? `found, corner error ${outcome.corner_error.toFixed(1)} px` : "NOT FOUND"}`,
    );
  }
}

await browser.close();
writeFileSync(path.join(OUT, "index.json"), JSON.stringify(results, null, 2));
console.log(JSON.stringify({ count: results.length, out: OUT }, null, 2));
