# Windows: install make with `choco install make`, or run from Git Bash.
PYTHON ?= python

.PHONY: install lint format test test-integration run dev up up-all down logs ingest index-toy eval-baseline

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

# Needs OpenSearch + Qdrant running (make up)
test-integration:
	$(PYTHON) -m pytest -m integration

run:
	$(PYTHON) -m uvicorn healthcare_rag.main:app --host 0.0.0.0 --port 8010

# API on the host with auto-reload, against OpenSearch + Qdrant from `make up`
dev:
	$(PYTHON) -m uvicorn healthcare_rag.main:app --reload --host 0.0.0.0 --port 8010

# OpenSearch + Qdrant in Docker, for use with `make dev`
up:
	docker compose up -d opensearch qdrant

# Everything in Docker, including the API
up-all:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

# Download 50 CC BY/CC0 PMC OA articles per topic, chunk, embed via the gateway, index.
# Idempotent: re-running skips downloaded articles and chunks already in Qdrant.
ingest:
	$(PYTHON) -m healthcare_rag.ingest all --per-topic 50

# M1 baseline: BM25-only (OpenSearch) + one gateway call over the toy corpus.
# Needs `make up` and the gateway on :8000. HARDWARE is a free-text note for the report.
HARDWARE ?=
index-toy:
	$(PYTHON) -m healthcare_rag.evaluation index

eval-baseline: index-toy
	$(PYTHON) -m healthcare_rag.evaluation run --markdown docs/baseline_results.md --hardware "$(HARDWARE)"
