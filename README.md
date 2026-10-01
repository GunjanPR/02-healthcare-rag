# Healthcare Research RAG over PubMed Central

Grounded question answering and summarization over PubMed Central Open Access literature. It uses hybrid retrieval (OpenSearch BM25 + Qdrant dense), reranking, and citation-required answers. All model inference goes through the [LLM Gateway](../01-llm-gateway/).

> Part of the Live Projects portfolio. See [SHARED_STANDARDS.md](../SHARED_STANDARDS.md) for conventions and the milestone definition of done.

## Overview

Researchers need answers they can check against the literature. This system ingests the
**PubMed Central Open Access** subset with PMCID normalization and deduplication, then
chunks articles by section. Retrieval is hybrid: BM25 in OpenSearch plus dense vectors in
Qdrant, followed by reranking. A LangGraph workflow produces **citation-required
answers**, and it abstains when the evidence is weak.

The service never calls a model provider directly. All chat and embedding calls go to
the **LLM Gateway (project 01)** through its OpenAI-compatible API, so routing, caching,
rate limits, and per-query cost and latency come from the gateway.

## Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI, Uvicorn |
| Orchestration | LangGraph (retrieve → rerank → generate → verify / abstain) |
| Sparse retrieval | OpenSearch (BM25) |
| Dense retrieval | Qdrant + sentence-transformers / biomedical embeddings |
| LLM inference | 01-llm-gateway (OpenAI-compatible), backed by Ollama/vLLM |
| Data | PubMed Central Open Access subset (JSONL ingestion) |
| Observability | OpenTelemetry, gateway Prometheus metrics |

## Architecture

| Component | Responsibility |
|---|---|
| **Ingestion** | Download the PMC OA subset, normalize PMCIDs, dedupe, chunk by section, tag entities for metadata filters |
| **Indexers** | Write chunks to OpenSearch (BM25) and their embeddings to Qdrant |
| **Retriever** | Hybrid search (reciprocal rank fusion of BM25 and dense results) with metadata filters |
| **Reranker** | Cross-encoder or LLM reranking of the top-k chunks |
| **LangGraph agent** | Retrieve → rerank → generate with citations → verify grounding → answer or abstain |
| **API** | `/ask`, `/summarize`, and `/health` endpoints |
| **Gateway client** | OpenAI-compatible client pointed at `GATEWAY_BASE_URL` for all LLM and embedding calls |

```
  PMC OA bulk files ──► ingestion (normalize, dedupe, chunk)
                             │                 │
                             ▼                 ▼
                     OpenSearch (BM25)   Qdrant (dense vectors)
                             ▲                 ▲
                             └──── hybrid ─────┘
                                  retriever
                                     │
  client ──► FastAPI /ask ──► LangGraph: retrieve → rerank → generate → verify
                                                     │            │
                                                     ▼            ▼
                                       ┌───────────────────────────────┐
                                       │   01-llm-gateway  /v1/...     │
                                       │   (routing, cache, metrics)   │
                                       └──────────────┬────────────────┘
                                                      ▼
                                               Ollama / vLLM
```

## Setup

### Prerequisites
- Python 3.11
- **01-llm-gateway running**, on `http://localhost:8000` by default, with a client API key
- OpenSearch 2.x: `docker run -d -p 9200:9200 -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true opensearchproject/opensearch:2`
- Qdrant: `docker run -d -p 6333:6333 qdrant/qdrant`
- Optional: Docker Desktop (full stack via `docker compose up` from M6)
- Optional: GNU Make (see the Windows note below)

### Install
```bash
git clone <repo-url> 02-healthcare-rag && cd 02-healthcare-rag
py -3.11 -m venv .venv            # Windows; on macOS/Linux use: python3.11 -m venv .venv
source .venv/Scripts/activate     # Git Bash; PowerShell: .venv\Scripts\Activate.ps1; macOS/Linux: source .venv/bin/activate
make install                      # = pip install -e ".[dev]"
```

### Environment config
```bash
cp .env.example .env              # then fill in values; .env is git-ignored
```
Set `GATEWAY_BASE_URL` and `GATEWAY_API_KEY` to match the gateway. `GATEWAY_API_KEY`
must be one of the gateway's `GATEWAY_API_KEYS`. Set `OPENSEARCH_URL` and `QDRANT_URL`
to your local services. NCBI settings are optional but are good etiquette for bulk
downloads.

## How to Run

> **Windows note:** `make` isn't installed on Windows by default. Install it with `choco install make`, or run the
> targets from **Git Bash**. You can also run the commands in the `Makefile` yourself
> (for example, `python -m pytest` instead of `make test`).

### Run the service
```bash
# 1. Start the LLM Gateway (in ../01-llm-gateway): make run
# 2. Start OpenSearch + Qdrant (see Prerequisites)
# 3. Ingest a small PMC sample (M3)
make ingest                           # = python -m healthcare_rag.ingest --limit 1000
# 4. Start the API on http://localhost:8010
make run                              # = uvicorn healthcare_rag.main:app --reload --port 8010
```
> `make run` starts working at **M2**, once `src/healthcare_rag/main.py` is added.
> `make ingest` starts working at **M3**.

### Try it
```bash
curl -X POST http://localhost:8010/ask -H "Content-Type: application/json" \
  -d '{"question": "What variants in BRCA1 are associated with reduced DNA repair?"}'
```

### Lint and test
```bash
make lint     # ruff check + ruff format --check
make test     # pytest (unit tests only; integration tests are marked `integration`)
make format   # auto-fix lint and formatting
```

## Evaluation & Results

- **Baseline:** BM25-only retrieval (OpenSearch), with no reranking.
- **Retrieval metrics:** recall@k, NDCG@10, MRR on a 100–200-question hand-labeled QA set.
- **Answer metrics:** grounded-answer rate, citation accuracy, faithfulness (RAGAS-style),
  and abstention precision.
- **Cost / latency:** per-query cost and p95 latency, taken from gateway metrics.
- **Results:** *targets until measured*: NDCG@10 lift over BM25-only, and grounded-answer
  rate above 90%.

## Milestones

Milestones follow [SHARED_STANDARDS.md](../SHARED_STANDARDS.md). Each one has a definition of done there.

### M1: Scaffold & baseline
- [x] Repo scaffold: `pyproject.toml`, Makefile, `.gitignore`, `.env.example`
- [x] Smoke test passes (`make test`)
- [ ] Baseline: BM25-only retrieval + single gateway call on a toy corpus

### M2: Core functionality
- [ ] Gateway client (OpenAI-compatible, configured via `GATEWAY_BASE_URL`)
- [ ] LangGraph flow: retrieve → generate with citations → abstain on weak evidence
- [ ] FastAPI `/ask` and `/summarize` endpoints

### M3: Data / storage layer
- [ ] PMC OA ingestion: PMCID normalization, dedupe, section-aware chunking
- [ ] OpenSearch BM25 index and Qdrant dense index (idempotent re-ingest)
- [ ] Hybrid retrieval (RRF) + metadata filters + reranking

### M4: Evaluation & testing
- [ ] Hand-labeled QA set (100–200 questions)
- [ ] Retrieval eval: recall@k, NDCG@10, MRR vs BM25-only baseline
- [ ] Answer eval: groundedness, citation accuracy, faithfulness
- [ ] Unit + integration tests

### M5: Observability / monitoring
- [ ] OpenTelemetry traces across API → retriever → gateway
- [ ] Per-query cost and latency dashboard (using gateway metrics)

### M6: Containerization & CI/CD
- [ ] Dockerfile and `docker compose up` (API, OpenSearch, Qdrant, gateway)
- [ ] GitHub Actions: lint + test (+ eval regression gate)

### M7: Docs, benchmarks & demo
- [ ] README with measured results against the BM25-only baseline
- [ ] Evaluation report
- [ ] Demo (Hugging Face Spaces / Streamlit or recording)

## Project Layout
```
02-healthcare-rag/
├── src/healthcare_rag/__init__.py
├── tests/test_smoke.py
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```
