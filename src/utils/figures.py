"""Phase 15 - saving a figure so that rebuilding it is not a diff.

    from src.utils.figures import save
    save(fig, path)            # instead of fig.savefig(path, dpi=150)

`reports/figures/` is tracked, and a rebuild of an unchanged figure produced different bytes:
`p3_rectification.png` moved 1,046,758 -> 1,026,337 in one batch for no content change. Every
figure rebuild was therefore a megabyte-scale binary diff, which makes `git log --stat` unusable
on any commit that regenerates a report and makes "did this figure change?" unanswerable.

Two causes, both fixed here:

**Metadata.** matplotlib writes a `Software` tag carrying its own version, and for PNG a
creation time. Both change without the picture changing. `metadata={"Software": None}` removes
the tag; matplotlib's PNG writer emits no date when `Software` is suppressed.

**Compression.** The default PNG compression level comes from `rcParams["savefig.pil_kwargs"]`
and Pillow's default, which has changed between Pillow releases - the 20 KB move above is that
shape of difference. Pinning `compress_level` makes the bytes a function of the picture.

`dpi` stays the caller's, because it is a property of the figure rather than of the encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: matplotlib writes its own version into every file it saves. Suppressed: a figure that
#: differs because matplotlib was upgraded is a diff that says nothing about the figure.
METADATA: dict[str, Any] = {"Software": None}

#: Pinned rather than left to Pillow's default, which has moved between releases. 6 is Pillow's
#: long-standing default; the point is that it is stated, not that it is 6.
PIL_KWARGS: dict[str, Any] = {"compress_level": 6}

DEFAULT_DPI = 150


def save(fig, path: str | Path, *, dpi: int = DEFAULT_DPI, **kwargs: Any) -> Path:
    """`fig.savefig` with the two sources of byte noise removed. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**METADATA, **kwargs.pop("metadata", {})}
    if path.suffix.lower() == ".png":
        kwargs.setdefault("pil_kwargs", PIL_KWARGS)
    fig.savefig(path, dpi=dpi, metadata=metadata, **kwargs)
    return path
