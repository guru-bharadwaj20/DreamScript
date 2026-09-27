"""Phase 16.3.1 — the home-screen icons, generated from the mark the favicon already draws.

An installed web app needs raster icons and a browser will not accept the inline SVG favicon
``index.html`` carries. Rather than draw a second, unrelated mark in an editor, this renders **the
same path** at the sizes the manifest and iOS ask for, so the tab, the install prompt and the home
screen show one thing. A brand that differs between the tab and the home screen is one a person
cannot recognise in either place.

Four files, and the differences between them are not arbitrary:

``icon-192.png`` / ``icon-512.png`` (``purpose: any``)
    A rounded square with transparent corners. Chrome and desktop shells draw these as-is, so the
    corner radius has to be in the image.

``icon-maskable-512.png`` (``purpose: maskable``)
    Full bleed, and the mark scaled into the inner **80%** circle. Android applies its own mask -
    circle, squircle, teardrop, whatever the launcher uses - and a maskable icon with the mark at
    the same size as the ``any`` icon comes out with its edges sliced off. The two purposes need two
    files; one file declared as both is the most common way this goes wrong.

``apple-touch-icon.png`` (180x180)
    Full bleed and **square**, because iOS applies its own squircle. Pre-rounding it leaves a black
    square visible in the corners of the rounded shape iOS then draws.

The gold is a vertical gradient rather than a flat fill (``#f0d488`` to ``#c9a227``, the two ends of
the palette's gold). At home-screen size a flat ``#e8c36a`` reads as yellow; a gradient through the
stroke reads as metal, which is what the palette is for.

Regenerate with:  .venv/Scripts/python.exe scripts/make_app_icons.py
Deterministic - regenerating produces byte-identical files.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "frontend" / "public" / "icons"

#: `--ground` in the dark theme, which is the ground the mark was designed on.
GROUND = (8, 8, 10, 255)
#: The two ends of the palette's gold: `--gold` lightened, and `--gold-deep`.
GOLD_TOP = (240, 212, 136)
GOLD_BOTTOM = (201, 162, 39)

#: Supersampling. The mark is one stroked curve and Pillow has no antialiased stroke, so it is drawn
#: large and reduced - which is also why the radius and the stroke width are floats below.
SS = 8

#: The favicon's viewBox. Every coordinate here is in those units so the two cannot drift.
VIEW = 32.0


def _gold(size: int) -> Image.Image:
    """A vertical gradient the full height of the icon, to be used through a mask."""
    gradient = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / max(size - 1, 1)
        gradient.putpixel(
            (0, y),
            tuple(round(a + (b - a) * t) for a, b in zip(GOLD_TOP, GOLD_BOTTOM, strict=False)),  # type: ignore[arg-type]
        )
    return gradient.resize((size, size), Image.Resampling.NEAREST)


def _mark_mask(size: int, scale: float) -> Image.Image:
    """The favicon's `D`, as an alpha mask at `size` px, occupying `scale` of the frame.

    The path is `M9 9h8a7 7 0 0 1 0 14H9z` with `stroke-width: 2.6` and a round join: a left-hand
    bracket closed by a semicircle, which at r=7 over a 14-unit span is exact rather than
    approximate.

    **Drawn as a filled silhouette minus its own inner silhouette, not as strokes.** The first
    version stroked three lines and an arc, and the result had a notch where the bowl met the bar -
    `ImageDraw.arc` measures its width *inward from the bounding box*, so an arc whose box has
    radius 7 has its centreline at 7 - width/2 and sits a stroke-width low against a line whose
    centreline is at 7. It rendered as a mug with a handle. Two filled shapes subtracted have no
    seam to get wrong.
    """
    canvas = Image.new("L", (size, size), 0)

    unit = size * scale / VIEW
    offset = (size - VIEW * unit) / 2

    def at(x: float, y: float) -> tuple[float, float]:
        return (offset + x * unit, offset + y * unit)

    half = 2.6 * unit / 2

    def silhouette(grow: float) -> Image.Image:
        """The `D` as a solid shape, its edge `grow` px outside (or inside) the path."""
        layer = Image.new("L", (size, size), 0)
        pen = ImageDraw.Draw(layer)
        left_x, top_y = at(9, 9)
        stem_x, bottom_y = at(17, 16)[0], at(9, 23)[1]
        # The straight part. Rounded on the **left** only, which is where the path actually turns;
        # rounding the right-hand corners put a nick in the join with the bowl, because the pie's
        # edge is tangent there and a radius pulls the rectangle away from it.
        pen.rounded_rectangle(
            [left_x - grow, top_y - grow, stem_x, bottom_y + grow],
            radius=max(0.0, grow),
            fill=255,
            corners=(True, False, False, True),
        )
        cx, cy = at(17, 16)
        r = 7 * unit + grow
        # A pie rather than an arc: the bowl is part of the solid, and the counter is cut out by
        # subtracting the inner silhouette below.
        pen.pieslice([cx - r, cy - r, cx + r, cy + r], start=-90, end=90, fill=255)
        return layer

    outer = silhouette(half)
    inner = silhouette(-half)
    canvas.paste(outer, (0, 0))
    canvas.paste(0, (0, 0), inner)
    return canvas


def _ground_mask(size: int, radius_fraction: float | None) -> Image.Image:
    """The ground's alpha: a rounded square, or a full square when `radius_fraction` is None."""
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    if radius_fraction is None:
        draw.rectangle([0, 0, size - 1, size - 1], fill=255)
    else:
        draw.rounded_rectangle(
            [0, 0, size - 1, size - 1], radius=round(size * radius_fraction), fill=255
        )
    return canvas


def icon(size: int, *, scale: float, radius_fraction: float | None, hairline: bool) -> Image.Image:
    big = size * SS
    ground_alpha = _ground_mask(big, radius_fraction)

    out = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ground = Image.new("RGBA", (big, big), GROUND)
    out.paste(ground, (0, 0), ground_alpha)

    if hairline:
        # A gold hairline just inside the edge. At home-screen size a flat black square floats on a
        # dark wallpaper with no edge at all; this gives it one without becoming a border.
        edge = Image.new("L", (big, big), 0)
        pen = ImageDraw.Draw(edge)
        inset = round(big * 0.012)
        pen.rounded_rectangle(
            [inset, inset, big - 1 - inset, big - 1 - inset],
            radius=round(big * (radius_fraction or 0.22)) - inset,
            outline=255,
            width=max(1, round(big * 0.006)),
        )
        tint = Image.new("RGBA", (big, big), (232, 195, 106, 70))
        out.paste(tint, (0, 0), edge)

    mark = _mark_mask(big, scale)
    gradient = _gold(big).convert("RGBA")
    out.paste(gradient, (0, 0), mark)

    return out.resize((size, size), Image.Resampling.LANCZOS)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int]] = []

    for size in (192, 512):
        # 0.22 of the side is the radius Android and Chrome's own shells use for a non-maskable
        # icon; iOS never sees these.
        image = icon(size, scale=0.78, radius_fraction=0.22, hairline=True)
        path = OUT / f"icon-{size}.png"
        image.save(path, "PNG", optimize=True)
        written.append((path.name, path.stat().st_size))

    # Maskable: full bleed, and the mark inside the inner 80% circle. 0.56 of the frame keeps the
    # whole `D` - which is 26 of 32 viewBox units at its widest - inside a circle of diameter 0.8.
    maskable = icon(512, scale=0.56, radius_fraction=None, hairline=False)
    path = OUT / "icon-maskable-512.png"
    maskable.save(path, "PNG", optimize=True)
    written.append((path.name, path.stat().st_size))

    # iOS masks this itself, so it is square and un-rounded.
    apple = icon(180, scale=0.74, radius_fraction=None, hairline=False)
    path = OUT / "apple-touch-icon.png"
    apple.save(path, "PNG", optimize=True)
    written.append((path.name, path.stat().st_size))

    for name, size in written:
        print(f"  {name:26} {size / 1024:6.1f} kB")
    print(f"wrote {len(written)} icons to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a script
    raise SystemExit(main())
