"""Phase 3.2 - geometric primitives extracted from a cleaned stroke mask.

Nine steps, each a module, each producing one kind of evidence about what was drawn:

    components   where the separate blobs of ink are, and their basic statistics
    contours     the outlines of those blobs, with the nesting between them
    polygons     those outlines simplified to corners
    segments     straight runs, wherever they occur
    curves       how sharply the skeleton bends, and where the corners are
    arrowheads   the convergent V at the end of a stroke
    text_regions the parts of the page that are writing rather than drawing
    layers       the page split into a shape layer and a text layer
    cache        all of the above, serialized per image

Nothing here classifies anything. The primitives are the *features* Phase 4 turns into vectors
and Phase 5 learns from; keeping the measuring separate from the deciding is what lets a bad
classifier be replaced without touching the measurements it was wrong about.
"""

from __future__ import annotations

__all__: list[str] = []
