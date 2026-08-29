# Data Versioning (DVC)

Phase 1.3.5. The corpus is ~5 GB of images that must never enter git, yet every experiment
has to be reproducible against an exact version of it. DVC resolves that: git carries small
`.dvc` pointer files containing content hashes, while the bytes live in a cache and a remote.

## What is tracked

| Path | Contents | Size |
| :--- | :--- | ---: |
| `data/raw/hdbpmn.dvc` | 704 BPMN photos + annotations + word transcripts | 1.1 GB |
| `data/raw/didi.dvc` | 22,287 ink diagrams + 6,555 dot prompts | 1.4 GB |
| `data/raw/iam_line.dvc` | 10,373 handwriting lines | 254 MB |
| `data/raw/flowchartseg.dvc` | 1,319 flowcharts with node masks | 105 MB |
| `data/raw/sketch2code.dvc` | 731 sketches + 484 HTML pages | 127 MB |
| `data/raw/fa_bresler.dvc` | 300 finite automata, 25 writers | 35 MB |
| `data/raw/chaos.dvc` | the curated chaos corpus | 84 MB |

Each pointer file records an md5 of the directory, its size and its file count, so a changed
byte anywhere produces a different hash and a visibly dirty `dvc status`.

## Setup on this machine

```
dvc init
dvc config cache.type "hardlink,symlink,copy"
dvc remote add -d local ~/dreamscript-dvc-store
```

`cache.type` matters at this scale. DVC's default is to *copy* every tracked file into
`.dvc/cache`, which would double 5 GB of images for no benefit. Hardlinks make the cache and
the working copy the same bytes on disk. The fallback chain exists because hardlinks fail
across volumes and symlinks need developer mode on Windows.

**The remote is local**, a directory outside the repository. That is a deliberate consequence
of the licence audit: `flowchartseg` has no declared licence and the chaos corpus contains
identifiable handwriting, so pushing either to a public cloud remote would breach the
"unknown means no" rule in `reports/license_audit.md`. A private remote can be added later
per dataset; nothing in the workflow changes if it is.

## Daily use

```
dvc status              # is the working copy in sync with the pointers?
dvc add data/raw/<new>  # start tracking a new dataset
dvc push                # copy cached data to the remote
dvc pull                # fetch it on another machine
dvc checkout            # restore the working copy to match the current .dvc files
```

The `.gitignore` excludes `data/raw/*` but explicitly re-includes `*.dvc`:

```
data/raw/*
!data/raw/*.dvc
```

Without that second line DVC refuses to work at all — it will not create a pointer file that
git is configured to ignore, which is a good error and worth keeping in mind if the ignore
rules are ever rewritten.

## Reproducing the corpus elsewhere

Two routes, and both are supported on purpose:

1. **From the remote:** `git clone` then `dvc pull`. Exact bytes, instantly.
2. **From the sources:** run the downloaders in `src/ingest/datasets/`. Slower, and it depends
   on those hosts still being up — two of the five have already died once (see
   `docs/data_cards/flowchart_fc.md` and `sketch2code.md`), which is precisely why route 1
   exists.

## What is deliberately *not* tracked

- `data/processed/manifest.parquet` and `data/interim/phashes.parquet` are **derived**. They
  are rebuilt by `python -m src.ingest.manifest` and `python -m src.ingest.dedup` in seconds
  from tracked inputs, so versioning them would store the same information twice and invite
  the two copies to disagree.
- `experiments/` — each run is immutable and self-describing (Phase 0.2.4); the run directory
  is the record.
- Model weights — Phase 15.3 puts those in a model registry, where staging and promotion are
  first-class rather than a file hash.
