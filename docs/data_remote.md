# The DVC remote

**There is no shared remote. `dvc pull` works on one machine and nowhere else.**

That sentence is the whole of this document's content, and it exists because four other places
in this repository imply the opposite.

## What is actually configured

`.dvc/config` defines four remotes and all four are directories on a single Windows machine:

```
[core]
    remote = local
['remote "local"']      url = C:/Users/Temp/dreamscript-dvc-store
['remote "raw"']        url = C:/Users/Temp/dreamscript-dvc-store/raw
['remote "interim"']    url = C:/Users/Temp/dreamscript-dvc-store/interim
['remote "processed"']  url = C:/Users/Temp/dreamscript-dvc-store/processed
```

3.4 GB, on that filesystem, reachable from no other host. A fresh clone - CI above all - gets
the `.dvc` pointers in git and cannot fetch a single byte behind them.

## Why that is not an accident, and is still a problem

A local-directory remote is a legitimate DVC setup: it gives content-addressed storage, `dvc
status -c`, and `dvc checkout` after a branch switch, without a cloud account. For a
single-machine research project those are most of the benefits.

What it does not give is the thing "remote" implies to a reader, and the repository has been
reading it that way:

| Where | What it says | What happens |
| :--- | :--- | :--- |
| `tests/conftest.py` | *"needs the DVC payload; run `dvc pull`"* | `dvc pull` fails on any other machine |
| `.github/workflows/ci.yml` | *"`data/` is `.dvc` pointers in git and content in a remote"* | there is no remote a runner can reach |
| `dvc.yaml` | twelve stages with declared outputs | four of them are frozen and their content is on one disk |
| `README` / onboarding | clone, install, pull | the third step cannot succeed |

The four sources whose content this has already cost are listed in [data_losses.md](data_losses.md), with what each one was used for.

The practical consequence is measured elsewhere in this audit: `dvc status -c` reports four cache
objects as *"missing from remote and local"*, and `data/raw/didi` is one of them - which is why
`configs/llm.yaml` no longer lists didi as a training source and why
`tests/test_assemble_serialise.py` names it in `ABSENT_SOURCES`.

## The cache is not the store

Even on the one machine, the two are different places and only one of them is durable:

| | |
| :--- | :--- |
| `.dvc/cache` | a working directory. `dvc gc` prunes it, and it is inside the clone. |
| the store | the copy that survives deleting the clone. |

`dvc status -c` prints `new:` for an object that is in the first and not the second, which reads
like "recently added" rather than "the only copy is the one you can delete by accident". It
printed that for **1,507 objects**: the whole Phase 12.1 target corpus (1,478 files), the arrow
pose checkpoint, the TrOCR fine-tune and every feature table. All of them were one `dvc gc` away
from the state the four sources above are in permanently.

They are pushed now, which is what took the store from 1.2 GB to the 3.4 GB quoted above. What
notices if it happens again:

```bash
make store     # python -m src.ingest store --check - non-zero if anything is cache-only
make push      # dvc push, then the same survey to say whether it finished
```

`python -m src.ingest store` reads the pointers, `dvc.lock` and the content-addressed layout
directly, so it answers in a second, while a `dvc repro` holds the DVC lock, and on a machine
with no DVC installed - where it says so and exits 0 rather than reporting the whole corpus
missing.

## What to do about it

Three options, in order of cost:

1. **Leave it and say so.** What this document does. The skip messages now name this file instead
   of telling a reader to run a command that cannot work.
2. **Add a real remote** (`dvc remote add -d s3 …`, or any of the twenty backends DVC supports)
   and `dvc push`. 3.4 GB. This is the only option that makes a fresh clone reproducible, and it
   is the one the plan assumes.
3. **Commit the small artefacts and stop tracking the rest.** `data/processed/ir/` is 2,796 JSON
   files and a few tens of MB; the checkpoints are not.

Until 2 happens, the honest description of this project's data is: **the code and the pointers are
portable, the payload is not.**
