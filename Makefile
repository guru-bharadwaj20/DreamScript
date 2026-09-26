# DreamScript task runner (Phase 0.2.5).
#
#   make help          list every target
#   make verify        prove the environment is intact
#   make data          build the corpus manifest
#   make train-clf     train the Phase 5 classifiers
#   make eval          run the evaluation suite
#   make app           serve the web demo
#
# Windows note: this repo is developed on Windows with Git Bash, where `make` may be absent.
# Every target is a one-line command you can also run directly, and `tasks.ps1` mirrors the
# whole file for PowerShell users (`./tasks.ps1 verify`).

SHELL := /bin/bash
PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

CONFIG ?= configs/base.yaml
OVERRIDES ?=

.DEFAULT_GOAL := help
# Every target, in one place and in sorted order. This list and the rules below had drifted
# apart in both directions: `verify-classical` was declared with no rule behind it (so
# `make verify-classical` was 'No rule to make target'), and `dummy-run` had a rule and was
# not declared. tests/test_task_targets.py compares the two sets now.
.PHONY: app backend clean clean-experiments dag data detect determinism dummy-run env eval features \
        finetune format help hooks lint ocr parse preprocess profile push repro rl serve stages store test test-fast typecheck \
        train-clf train-ens train-nn verify verify-classical verify-cv verify-genai verify-gpu

## -- environment -------------------------------------------------------------

help:  ## list available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

env:  ## create the venv and install every requirements layer
	# Every layer, not five of six. This installed base, torch, cv, classical and genai and
	# skipped serve.txt and dev.txt, so a clean `make env` produced an environment where
	# `make serve` failed on uvicorn, `make lint` failed on ruff, and `make repro`/`make dag`
	# failed on dvc - while CI installed a different set again. The requirements test asserts
	# this target, env.yml, the Dockerfile and the CI job agree.
	uv venv --python 3.11 .venv
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -r requirements/base.txt
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -r requirements/torch.txt \
		--index-url https://download.pytorch.org/whl/cu124
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -r requirements/classical.txt \
		-r requirements/cv.txt -r requirements/serve.txt \
		-r requirements/app.txt -r requirements/genai.txt \
		-r requirements/test.txt
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -r requirements/dev.txt
	# The hooks are configured in .pre-commit-config.yaml and had never been installed in
	# this clone - which is how a 4.96 MB log got past check-added-large-files. Installing
	# is per clone and there is no way to do it from the config, so it belongs here.
	$(PY) -m pre_commit install

verify:  ## Phase 0.1 — check every installed stack
	$(PY) scripts/verify_env.py

verify-gpu:  ## Phase 0.1.2 — CUDA, bf16 and VRAM
	$(PY) scripts/check_cuda.py

verify-cv:  ## Phase 0.1.3 — CV stack binarization smoke test
	$(PY) scripts/smoke_cv.py

verify-classical:  ## Phase 0.1.4 - sklearn, xgboost, lightgbm, hmmlearn, imblearn
	$(PY) scripts/verify_env.py --only classical

verify-genai:  ## Phase 0.1.5 — load the 7B base model in 4-bit
	$(PY) scripts/check_4bit_load.py

determinism:  ## Phase 0.1.6 — two runs must produce identical metrics
	$(PY) scripts/determinism_check.py --compare

dummy-run:  ## Phase 0.2.4 — write a complete experiment run directory
	$(PY) scripts/dummy_run.py

## -- quality -----------------------------------------------------------------

test:  ## run the full pytest suite
	$(PY) -m pytest

test-fast:  ## skip anything marked slow
	$(PY) -m pytest -m "not slow"

lint:  ## ruff + black --check + isort --check + mypy against its baseline
	# `app` joined this list in 16.1. It is a fourth tree of Python and every gate that named
	# three would have let it through unformatted and unchecked - which is how a tree ends up
	# with its own house style.
	$(PY) -m ruff check src tests scripts app
	$(PY) -m black --check src tests scripts app
	$(PY) -m isort --check-only src tests scripts app
	$(PY) scripts/typecheck.py

hooks:  ## install the pre-commit hooks into .git/hooks (run once per clone)
	$(PY) -m pre_commit install

typecheck:  ## mypy, held to reports/mypy_baseline.json - the count may fall, not rise
	$(PY) scripts/typecheck.py

format:  ## apply black + isort + ruff --fix
	$(PY) -m isort src tests scripts app
	$(PY) -m black src tests scripts app
	$(PY) -m ruff check --fix src tests scripts app

## -- pipeline ----------------------------------------------------------------
#
# Every target below ran `python -m src.<package> --config ...`, and every one of those raised
# `StageNotImplemented` and exited 2. Fourteen targets that did nothing, while the work they name
# sat in the packages under names nobody could guess. They name it now, and
# `python -m src.<package>` with no argument still lists what else is in there.
#
# `make serve` was the worst of them: it pointed at the stub while the Dockerfile and the CI
# image job both ran `uvicorn src.serve.api:app`, which is the actual server.

data:  ## Phase 1 - build the corpus manifest and splits
	$(PY) -m src.ingest manifest $(OVERRIDES)
	$(PY) -m src.ingest splits $(OVERRIDES)

preprocess:  ## Phase 3 - photos to clean strokes, text/shape layers and primitives
	$(PY) -m src.preprocess layers $(OVERRIDES)

features:  ## Phase 4 - build the handcrafted feature table
	$(PY) -m src.features build $(OVERRIDES)

train-clf:  ## Phase 5 - Decision Tree / KNN / Logistic Regression
	$(PY) -m src.classify s1 $(OVERRIDES)

train-nn:  ## Phase 6 - MLP and SVM on image embeddings
	$(PY) -m src.classify mlp $(OVERRIDES)
	$(PY) -m src.classify svm $(OVERRIDES)

train-ens:  ## Phase 7 - Random Forest / boosting ensembles
	$(PY) -m src.classify forest $(OVERRIDES)
	$(PY) -m src.classify boosting $(OVERRIDES)

detect:  ## Phase 9.1 - component detector
	$(PY) -m src.detect train $(OVERRIDES)

ocr:  ## Phase 9.3 - handwriting recognition
	$(PY) -m src.ocr s3 $(OVERRIDES)

parse:  ## Phases 7.3/10 - HMM roles and graph assembly into the IR
	$(PY) -m src.parse s4 $(OVERRIDES)

rl:  ## Phase 11 - traversal policy
	$(PY) -m src.rl dqn $(OVERRIDES)

finetune:  ## Phase 12.2 - one QLoRA training run from configs/llm.yaml
	$(PY) -m src.llm.run --config configs/llm.yaml $(OVERRIDES)

eval:  ## Phase 14 - the master table and every artefact it reads
	$(PY) -m src.eval stagewise
	$(PY) -m src.eval ablate
	$(PY) -m src.eval compute
	$(PY) -m src.eval humanbaseline
	$(PY) -m src.eval master

app: backend  ## alias for `make backend` — Phase 16's app, whose server side is the backend

backend:  ## Phase 16.1 - run the app backend (proxies the model server, holds no model)
	$(PY) -m uvicorn app.backend.main:app --host 127.0.0.1 --port 3000

profile:  ## Phase 16.1.5 - GPU and CPU latency for the served pipeline, into reports/
	$(PY) -m src.serve profile

serve:  ## Phase 15.11 - run the inference service (what the Dockerfile and CI run)
	$(PY) -m uvicorn src.serve.api:app --host 127.0.0.1 --port 8000

stages:  ## list every command each pipeline package can run
	@for pkg in ingest preprocess features classify detect ocr parse rl synth eval serve; do $(PY) -m src.$$pkg; echo; done

## -- housekeeping ------------------------------------------------------------

clean:  ## remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info

clean-experiments:  ## delete every run directory (irreversible)
	rm -rf experiments/*/

## -- pipeline DAG (Phase 15.4) ------------------------------------------------

repro:  ## Phase 15.4 — run the DVC DAG with the project venv on PATH
	PATH="$(CURDIR)/$(dir $(PY)):$$PATH" $(PY) -m dvc repro $(STAGES)

dag:  ## Phase 15.4 — print the pipeline graph
	$(PY) -m dvc dag

push:  ## put every cached DVC output in the store (the only copy that outlives .dvc/cache)
	$(PY) -m dvc push
	$(PY) -m src.ingest store --check

store:  ## Phase 1.1.9 — which tracked outputs are in the store and which are cache-only
	$(PY) -m src.ingest store --check
