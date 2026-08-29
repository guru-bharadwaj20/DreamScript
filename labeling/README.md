# Labelling

Phase 2.2.1. Label Studio is the annotation tool; this directory holds its configuration and
the task files generated from the corpus.

## Why Label Studio and not CVAT

CVAT is the stronger tool for boxes and masks, and if this project only needed to localise
shapes it would be the right choice. It is not what this project needs. The hardest label here
is a **relation** - which arrow leaves which shape - and Label Studio's `<Relations>` control
makes that something the annotator draws directly, with a type on it. Per-region `<Choices>`
and `<TextArea>` then cover the semantic role and the transcription in the same pass, so one
image is labelled once instead of three times in three tools.

The cost is that Label Studio is weaker at pixel masks. Nothing in the plan needs them: the
IR stores boxes and polylines.

## Install

Label Studio pins its own versions of Django, pydantic and others, so it gets its own
environment rather than being allowed near the project's:

```
uv venv .venv-labelstudio --python 3.11
VIRTUAL_ENV=$PWD/.venv-labelstudio uv pip install label-studio
```

## Files

| Path | What it is |
| :--- | :--- |
| `label_studio/config.xml` | The labelling interface. Its shape and role lists are generated from `src.ir.vocab`; `python -m src.ir.labelstudio --check` fails if they drift |
| `tasks/*.json` | Generated task files, one per source, each task carrying a pre-annotation from the converters |

## Generate tasks

```
python -m src.ir.convert.hdbpmn                                  # IR first
python -m src.ir.labelstudio --tasks data/processed/ir/hdbpmn    # then tasks
```

Every task arrives **pre-annotated** with the converter's output. The annotator corrects rather
than starts blank, which is the difference between labelling 260 diagrams and labelling them
twice. `docs/annotation_guide.md` is explicit that a pre-annotation is not truth - the hdBPMN
shapes in particular are a BPMN drawing convention, not an observation of the photo.

## Run it

```
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$PWD \
.venv-labelstudio/Scripts/label-studio start
```

Then, in the UI: create a project, paste `label_studio/config.xml` into Labelling Interface,
add a **Local Files** source storage pointing at the image directory, sync it, and import the
task JSON.

That storage step is not optional. `/data/local-files/` serves nothing on the strength of the
document-root variable alone - a file is only served if a `LocalFilesImportStorage` covers it.
Without it every task renders as a broken image with no error message anywhere.

## Verify

```
.venv-labelstudio/Scripts/python scripts/labelstudio_verify.py
```

This starts a throwaway server, creates the project from the config, registers and syncs the
storage, imports the pre-annotated tasks, and fetches an image over HTTP. Eleven checks, all of
which must pass; the temporary instance is deleted afterwards. Recorded result:

```
PASS  server_started                    PASS  pre_annotated_tasks_imported
PASS  legacy_tokens_enabled             PASS  predictions_survived_the_import
PASS  project_created_with_our_config   PASS  tasks_listed
PASS  local_files_storage_registered    PASS  tasks_reference_local_images
PASS  storage_synced                    PASS  image_served_over_http
                                        PASS  image_bytes_are_an_image
```

Two things to expect if you run it yourself: the first boot takes several minutes while Label
Studio migrates a cold database, and legacy API tokens are disabled by default in 1.23 - the
script turns them on through the Django ORM, because there is no API for it that does not
already need a token.

## Export

Export in **JSON** format (not JSON-MIN: the relations are lost) and convert with
`src.ir.labelstudio.to_ir`, which puts annotator flags where they belong - `crossed-out` into
`crossed_out[]`, `text-uncertain` into `low_conf_text[]`, and an `uncertain` relation into
`unresolved_edges[]`.
