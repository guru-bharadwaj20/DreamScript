# Scribe Consent and Registry

Phase 1.2.6. Handwriting identifies people. Before anyone draws for this corpus they are told
what happens to the images, and that record lives in the registry — not in someone's memory.

## What each contributor is told

> These sketches are for a student research project that converts hand-drawn diagrams into
> code. Your drawings will be photographed and stored **on one machine only**. They will not
> be uploaded, published, or included in any repository, report or demo. What may be
> published is what a model learned from them: accuracy numbers and model weights, never your
> images.
>
> Please do not draw anything personal — no names, addresses, passwords, or real data. Use the
> example scenarios you are given.
>
> You can ask for your drawings to be deleted at any time, without giving a reason. Say the
> word and everything under your scribe id is removed and the models are retrained without it.

## Why handwriting is treated as personal data here

Handwriting is biometric-adjacent: a person is often identifiable from it, and unlike a
password it cannot be changed. That is why:

- The registry stores an **opaque id** (`scribe03`), never a name or contact detail.
- `data/raw/` is excluded from git (`.gitignore`) and tracked by DVC to a **local** remote.
- The published artifacts are metrics and weights. Raw photographs never leave the machine.
- Withdrawal is a supported operation, not a favour — see below.

## The registry

`data/raw/chaos/scribes.csv`, created from
[`scribes_template.csv`](scribes_template.csv):

| Column | Meaning |
| :--- | :--- |
| `scribe_id` | opaque id, `scribeNN`; the join key for splits and evaluation |
| `style` | self-declared: `neat`, `average`, or `messy` |
| `handedness` | `left` or `right` — affects slant, which Phase 8.6 clusters on |
| `media_used` | semicolon-separated subset of pencil, ballpoint, marker, whiteboard, stylus |
| `consent` | must be `yes`; a row without it is not used |
| `notes` | anything relevant, no personal data |

The registry file itself stays out of git (it lives under `data/raw/`); only the template is
committed.

## The style mix is deliberate

The template ships eight rows: **2 neat, 4 average, 2 messy**, with two left-handed
contributors. This is not decoration. Risk D3 in [`../risks.md`](../risks.md) is that a corpus
drawn mostly by one person — or by eight equally tidy people — produces excellent scores that
collapse on a real user. The grouped cross-validation in Phase 5.2.2 splits on `scribe_id`
precisely so that "neat drafter vs. chaotic scribbler" is measured rather than assumed, and it
can only measure a range that was actually collected.

## Checks

```
python -m src.ingest.scribes            # show the registry and its state
python -m src.ingest.scribes --validate # exit non-zero if anything is wrong
```

| Check | Requirement |
| :--- | :--- |
| `min_scribes_registered` | ≥ 8 consented contributors (Phase 1.2.6) |
| `all_consented` | every registered row has `consent=yes` |
| `style_range_covered` | at least one neat, one average and one messy |
| `no_unregistered_drawings` | no image on disk belongs to an unknown scribe |
| `no_validation_problems` | ids, styles, handedness and media all valid |

## Withdrawal procedure

1. Delete `data/raw/chaos/*/<scribe_id>/`.
2. Remove the row from `scribes.csv`.
3. Rebuild the manifest (`python -m src.ingest --config configs/ingest.yaml`) — the splits
   change, because they are keyed on scribe.
4. Retrain anything that consumed that data, and note the change in the affected model cards.

Step 4 is the reason consent is captured before drawing rather than after: withdrawal is
cheap early and expensive late.

## Status

The registry template is committed. **No drawings have been collected yet** — the corpus is
empty, and `python -m src.ingest.collection --progress` reports 0 of 260. Every check above
reads PENDING until real people draw real diagrams.
