# What is gone

4 of the 8 raw sources are tracked by a `.dvc` pointer whose content exists **nowhere** - not in the working tree, not in `.dvc/cache`, not in the store. They are not "not pulled yet". They are unrecoverable from this repository.

| Source | files | `.dvc` md5 | worktree | local cache | store |
| :--- | ---: | :--- | :---: | :---: | :---: |
| `fa_bresler` | 607 | `cc79fb580ba5...` | ✓ | ✓ | ✓ |
| `flowchartseg` | 10 | `9a46baf77132...` | ✓ | ✓ | ✓ |
| `hdbpmn` | 2119 | `4dae1ba8d1f2...` | ✓ | ✓ | ✓ |
| `sketch2code` | 3406 | `86354fb67cab...` | ✓ | ✓ | ✓ |
| `cghd_extracted` | 198 | `b05a95d20a6d...` | ✗ | ✗ | ✗ |
| `chaos` | 262 | `88fbd6e7d078...` | ✗ | ✗ | ✗ |
| `didi` | 19668 | `ec368b7884c2...` | ✗ | ✗ | ✗ |
| `iam_line` | 12 | `3ccfd84dcd30...` | ✗ | ✗ | ✗ |

Regenerate with `python -m src.ingest losses`.

## Why this needed its own page

`dvc status -c` reports these as "missing from remote and local", which reads exactly like the line it prints for a source you simply have not pulled. The distinction is the whole point:

* a **not pulled** source is one `dvc pull` away;
* a **lost** source needs re-acquiring from its origin, and for three of these four that means an external download that may or may not still resolve.

`reports/license_audit.md` already said "4 of 9 registered sources are present locally", which is true and does not say which of the two states the other four are in.

## What each one cost

| Source | Where it was used | What its absence means |
| :--- | :--- | :--- |
| `didi` | `configs/llm.yaml` training sources, `src/ir/convert/didi.py`, `pairs.REAL_SOURCES` | ~3,000 IR documents the corpus does not have. Removed from the training sources in this batch; the converter is kept because the payload is re-acquirable from the published DIDI release. |
| `iam_line` | `src/ingest/manifest.from_iam` | handwriting lines for the Phase 9.3 recogniser, which was trained on label crops instead. |
| `cghd_extracted` | `routing.SOURCE_TYPE` maps `cghd -> circuit` | why the router can never learn `circuit`; named in `routing.UNCOVERED_SOURCES`. |
| `chaos` | `src/ingest/chaos_builder.py`, `reports/chaos_corpus.md` | the adverse-condition corpus. Its report survives; its images do not. |

## What to do

The `.dvc` pointers stay. They are the record of what the corpus was built from, and deleting them would make the manifest, the reports and `contributing.md` refer to sources that leave no trace. Re-acquiring any of them is a matter of running its downloader in `src/ingest/datasets/` and `dvc add`-ing the result; the md5 above is what a successful re-acquisition should reproduce.

See also [data_remote.md](data_remote.md), which is why none of this is recoverable from a second machine either.
