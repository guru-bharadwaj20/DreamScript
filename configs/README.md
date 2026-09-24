# `configs/` — what is live, and what is a declaration

Thirteen files. `base.yaml` is inherited by all of them; the other twelve are one per pipeline
package, named after it, as `docs/conventions.md` section 5 requires.

**Not all of them steer code, and this file says which do.** The honest position matters here
because the failure mode is silent: a value that nothing reads can be tuned, committed and
reported, and the number it was supposed to change never moves.

## Read by code

| File | Read by | What it decides |
| :--- | :--- | :--- |
| `base.yaml` | `src.utils.config.load_config`, every config's `defaults:` | paths, seed, `deterministic`, logging (`run_name`, `level`, `to_file`, `track`) |
| `llm.yaml` | `src.llm.run`, `src.llm.train` | the whole QLoRA run: LoRA rank and alpha, learning rate, token budget, eval cadence |

## Checked against code

| File | Checked by | Against |
| :--- | :--- | :--- |
| `preprocess.yaml` | `tests/test_config_matches_code.py` | `binarize` (defaults + `PHOTO`), `denoise`, `illumination`, `deskew` and `rules` |

These are settings the code owns — they are the values the reported numbers were measured at, so
the code is the source of truth and the config is the published record of it. The test fails if
either side moves without the other, which is what makes the record worth reading.

## Declarations

The other ten — `classify.yaml`, `detect.yaml`, `eval.yaml`, `features.yaml`, `ingest.yaml`,
`ocr.yaml`, `parse.yaml`, `rl.yaml`, `serve.yaml`, `synth.yaml` — carry `logging.run_name`, which
**is** read, by `start_run`, and becomes both the run directory name and the MLflow experiment.
Everything else in them records the intended shape of a stage that is driven by its module's own
argparse rather than by a config tree.

Reaching them:

```
python -m src.detect --print-config            # resolve and print, overrides applied
python -m src.detect --print-config model=x    # dotted overrides work here too
python -m src.detect                           # what the package can actually run
```

## The rule

A value in this directory is either read by code, checked against code, or a declaration —
and `tests/test_config.py` asserts every file here appears in exactly one of the three sections
above. Adding a config without saying which it is fails.
