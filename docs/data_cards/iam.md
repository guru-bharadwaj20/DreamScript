# Data Card — IAM Handwriting (line-level)

Verified by `python -m src.ingest.datasets.iam` on 2026-08-29.

## Identity

| Field | Value |
| :--- | :--- |
| Name | IAM Handwriting Database, line-segmented |
| Copy used | `Teklia/IAM-line` (HuggingFace) |
| Original source | https://fki.tic.heia-fr.ch/databases/iam-handwriting-database |
| Paper / citation | Marti & Bunke — *The IAM-database: an English sentence database for offline handwriting recognition* (IJDAR 2002) |
| Retrieved on | 2026-08-29 |
| Local path | `data/raw/iam_line/` |
| DVC-tracked | yes |

## License and permissions

| Field | Value |
| :--- | :--- |
| License | **MIT** (the Teklia mirror); the original IAM is research-use only and requires registration |
| Redistribution allowed | yes, with attribution (mirror) |
| Commercial use allowed | mirror says yes; **the original IAM terms say research only** |
| Attribution required | yes — cite Marti & Bunke |
| Restrictions carried into DreamScript | this project is academic, so the stricter original terms are satisfied either way. Raw files stay local; only the trained OCR model and its metrics are published |

**Why a mirror:** the canonical IAM download sits behind a registration form and manual terms
acceptance, which cannot be scripted. The mirror is recorded as the copy actually used, with
the original cited as the true provenance — not silently swapped.

## Contents (measured)

| Field | Value |
| :--- | :--- |
| Number of items | **10,373 text lines** |
| Splits | 6,482 train / 976 validation / 2,915 test |
| Modality | grayscale images of handwritten English text lines |
| Columns | `image`, `text` |
| Total size on disk | 265,569,932 bytes (~253 MB) |
| Character set observed | **78 distinct characters** |
| Annotation types | ground-truth transcription per line |

Sample transcripts:

```
"It was a splendid interpretation of the"
"sympathetic C O . Paul Daneman gave another"
"part . The rest of the cast were well chosen ,"
```

The plan estimated "13K handwritten text samples"; the line-level split of IAM actually
contains 10,373 lines. The difference is a units question — IAM has ~13k lines counted
differently across releases, and ~115k words — not a shortfall. **10,373 lines is the number
this project has.**

## Provenance

- **Who created it, and how?** 657 writers copied passages from the Lancaster-Oslo/Bergen
  corpus onto forms, which were scanned at 300 dpi and segmented into lines.
- **Quality control:** transcriptions are verified against the source text.

## Fit for DreamScript

| Field | Value |
| :--- | :--- |
| Diagram types covered | none — this is a text-only source |
| Phases that consume it | 1.1.4 acquisition, **9.3 OCR pretraining** |
| Maps to the IR how? | it does not; it trains the recognizer that fills the IR's `text` fields |
| Conversion script | not needed; consumed directly as (image, transcript) pairs |

**How it is used:** the CRNN+CTC recognizer is pretrained here, then fine-tuned on text crops
from actual diagrams (Phase 9.3.2). IAM supplies the volume of handwriting variety that the
260-diagram chaos corpus never could.

## Bias and representativeness — the important caveat

**IAM is prose, and diagram labels are not prose.** This mismatch is the main risk in using it:

| IAM | Diagram labels |
| :--- | :--- |
| Full English sentences, 40+ characters per line | Short fragments: `yes`, `no`, `age > 18`, `user_id` |
| Natural word frequencies and letter transitions | Identifiers, operators, `snake_case`, digits, symbols |
| Consistent baseline, ruled writing | Text squeezed inside a box, rotated along an edge, cramped |
| Cursive and connected script | Often printed capitals |

A language-model-flavoured decoder trained only on IAM will confidently "correct"
`user_id` into an English word. That is exactly why Phase 9.3.4 adds lexicon-biased decoding
over a diagram vocabulary, and why IAM is used for **pretraining only**, never for reporting
OCR accuracy on diagrams.

- **Writers:** 657 — excellent style variety, all Latin script.
- **Language:** English only.
- **Capture conditions:** clean 300 dpi scans; no shadows, no perspective, no whiteboard glare.
- **What this dataset will make the model bad at:** short symbolic labels, non-horizontal
  text, and anything not in an English lexicon.

## Known problems

- Prose/label domain gap (above) — the dominant limitation.
- Test split is large (2,915) relative to train (6,482); the split is kept as published so
  results stay comparable with the literature.
- The mirror's relationship to the original release version is not documented, so the exact
  IAM release is recorded as "unknown".

## Ethical and privacy notes

Handwriting from consenting contributors to an academic corpus; content is copied literary
prose, not personal writing. No personal data.

## Changelog

| Date | Change |
| :--- | :--- |
| 2026-08-29 | initial card; 10,373 lines / 78-character set verified locally via the Teklia mirror |
