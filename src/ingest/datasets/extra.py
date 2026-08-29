"""Phase 1.2 support — the two datasets acquired to fill the state-machine and circuit slots.

Neither was in the original plan. They were found while assembling the chaos corpus, because
nothing in the five planned sources covers state machines or circuits at all.

- **FA database** (Bresler, CTU Prague): 300 finite automata by 25 writers, InkML strokes with
  full symbol/relation annotation. Notably, this is the *sibling* of the FC-A/FC-B flowchart
  database whose download is dead - the FA archive link still works.
- **CGHD** (DFKI): 4,157 photographed hand-drawn circuits across 33 drafters, VOC annotations
  and segmentation maps. Its capture protocol - 4 photos per drawing at varying angle, lighting
  and blur - is the closest thing in public data to this project's own adverse-capture plan.
"""

from __future__ import annotations

from src.ingest.registry import Availability, Dataset, Redistribution, register

FA = register(
    Dataset(
        slug="fa_bresler",
        name="FA database (finite automata, Bresler)",
        url="https://cmp.felk.cvut.cz/~breslmar/finite_automata/",
        license="free to download for research (per the authors' page)",
        availability=Availability.OPEN,
        redistribution=Redistribution.UNKNOWN,
        diagram_types=("state_machine",),
        phases=("1.2.3", "7.3"),
        data_card="docs/data_cards/fa_bresler.md",
        notes=(
            "300 diagrams, 25 writers, 12 patterns each; InkML with stroke position, time and "
            "pressure, plus symbol/relation annotation and arrow connection points. Captured "
            "on a Lenovo X61 tablet, so `stylus` is documented rather than estimated. The "
            "sibling FC flowchart database at the same site is dead; this one is not."
        ),
    )
)

CGHD = register(
    Dataset(
        slug="cghd",
        name="CGHD (Circuit Graph Hand-Drawn, DFKI)",
        url="https://zenodo.org/records/17469897",
        license="CC-BY (see repository LICENSE)",
        availability=Availability.OPEN,
        redistribution=Redistribution.ATTRIBUTION,
        diagram_types=("circuit",),
        phases=("1.2.5", "9.1"),
        data_card="docs/data_cards/cghd.md",
        notes=(
            "4.9 GB zip, 4,157 images over 33 drafter folders, PASCAL VOC boxes for 45 symbol "
            "classes plus segmentation maps and some LTspice netlists. Capture protocol varies "
            "angle, lighting and blur across 4 photos per drawing, with the first shot clean - "
            "a natural adverse-condition spread. A 198-image subset is extracted for the chaos "
            "corpus; the full zip stays archived for Phase 9.1 detector training."
        ),
    )
)
