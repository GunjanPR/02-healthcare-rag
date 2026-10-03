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
| Dense retrieval | Qdrant + biomedical embeddings (via the gateway; optional local reranker with `pip install -e ".[rerank]"`) |
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
| **API** | `/health` (liveness) and `/ready` (OpenSearch + Qdrant reachable) today; `/ask` and `/summarize` from M2 |
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
- Docker Desktop, for OpenSearch 2.17 and Qdrant 1.12 (`docker compose`). OpenSearch runs
  single-node with the security plugin disabled (local dev only) and a 512 MB heap; the
  whole stack idles at about 1 GB of RAM.
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

### Run everything in Docker
```bash
docker compose up -d --build          # API :8010, OpenSearch :9200, Qdrant :6333 (= make up-all)
curl localhost:8010/health            # liveness
curl localhost:8010/ready             # 200 when OpenSearch and Qdrant are reachable, else 503
docker compose down                   # = make down (add -v to drop the indexes)
```
The app container reaches the gateway on the host through `host.docker.internal:8000`
(override with `GATEWAY_BASE_URL_DOCKER`). `/ready` reports the gateway's status but does
not require it.

### Run the API on the host
```bash
# 1. Start the LLM Gateway (in ../01-llm-gateway): make run
make up                               # 2. OpenSearch + Qdrant in Docker
make dev                              # 3. API on http://localhost:8010 with auto-reload
make ingest                           # 4. Ingest a small PMC sample (works from M3)
```

### Run the M1 baseline
```bash
# Needs `make up` and the gateway on :8000, with GATEWAY_API_KEY set in .env
make eval-baseline HARDWARE="<your machine and power state>"
# = index eval/toy/corpus.json into OpenSearch, answer eval/toy/questions.json,
#   write eval/results/baseline_bm25.json (git-ignored) and docs/baseline_results.md
```

### Try it
```bash
# Ask a question. Optional filters: sections, topics, pmcids, year_min, year_max
curl -X POST http://127.0.0.1:8010/ask -H "Content-Type: application/json" \
  -d '{"question": "Why do some patients on lisinopril get a dry cough?"}'

# Summarize one indexed article
curl -X POST http://127.0.0.1:8010/summarize -H "Content-Type: application/json" \
  -d '{"pmcid": "PMC1234567"}'
```
`/ask` returns the answer, numbered citations (PMCID, section, link and snippet), per-stage
timings and token usage and cost. If the evidence is weak, it abstains with a fixed message
and a `reason`:
- `no_hits`: nothing was retrieved.
- `low_score`: the best reranker score is below `ABSTAIN_THRESHOLD`.
- `model_insufficient`: the model replied that the passages don't contain the answer.
- `uncited`: the answer cited none of the passages it was given.

To try it on the toy corpus before ingesting PMC (M3):
```bash
make index-toy && OPENSEARCH_INDEX=toy_chunks make dev
```

### Lint and test
```bash
make lint     # ruff check + ruff format --check
make test     # pytest (unit tests only; no Docker or network needed)
make test-integration  # tests marked `integration` (needs `make up`)
make format   # auto-fix lint and formatting
```

## Evaluation & Results

- **Baseline:** BM25-only retrieval (OpenSearch), with no reranking.
- **Retrieval metrics:** recall@k, NDCG@10, MRR on a 100–200-question hand-labeled QA set.
- **Answer metrics:** grounded-answer rate, citation accuracy, faithfulness (RAGAS-style),
  and abstention precision.
- **Cost / latency:** per-query cost and p95 latency, taken from gateway metrics.
- **Method (full system):** *to be defined in M4* (QA set construction, labeling protocol, N).
- **Results (full system):** [pending]. No numbers are reported until they are measured.
- **Targets (not measured):** NDCG@10 lift over BM25-only, and grounded-answer rate above 90%.

### M1 baseline: BM25-only + one gateway call (measured 2026-10-03)

**What it is:** the control that later pipelines are compared against. The question is sent
as a BM25 query to OpenSearch (English analyzer, `title` + `text`). The top 3 passages go
into one chat call to `llama3.2` (3B) through the LLM Gateway, at temperature 0 with the
gateway cache bypassed. There is no reranking, no dense retrieval and no abstention logic.

**Data:** a self-written toy corpus of **24 short passages** on common clinical topics
(`eval/toy/corpus.json`) and **24 questions** (`eval/toy/questions.json`). Each question
has one relevant passage and a reference answer, given as keyword groups. An answer counts
as correct when every keyword group appears as a whole term in it.

**Setup:** Acer Nitro V 15 (i5-13420H, RTX 4050 Laptop GPU 6 GB) on AC power over USB-C,
so the GPU is held to a **20 W power limit** (60 W default), as in project 01. Ollama 0.35.0
runs natively on the GPU, the gateway on the host, and OpenSearch 2.17.1 in Docker Desktop
with a 512 MB heap. Requests are sequential, after one warm-up request. Latency is client-side
wall-clock time.

| Metric (N = 24) | Run 1 | Run 2 |
|---|---|---|
| Recall@1 / Recall@3 / Recall@5 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| MRR@10 | 1.000 | 1.000 |
| Answer accuracy (keyword match) | 1.000 (24/24) | 1.000 (24/24) |
| Retrieval latency p50 / p95 | 6 ms / 8 ms | 6 ms / 6 ms |
| Generation latency p50 / p95 | 902 ms / 1439 ms | 942 ms / 1455 ms |
| End-to-end latency p50 / p95 | 908 ms / 1446 ms | 948 ms / 1461 ms |
| Mean prompt / completion tokens | 368 / 27 | 368 / 27 |
| Cost (gateway `x-gateway-cost-usd`) | $0.00 | $0.00 |

Per-question answers and ranks for run 1 are in
[docs/baseline_results.md](docs/baseline_results.md).

**How to read these numbers:**
- **The quality numbers are at the ceiling.** Each toy passage covers one topic, and each
  question shares key terms with its passage, so BM25 ranks the right passage first every
  time. This set checks that the pipeline works end to end. It cannot show an improvement
  from reranking or hybrid retrieval. That comparison needs the M4 QA set over real PMC
  articles, where BM25-only is again the named baseline.
- **Latency is almost all generation:** retrieval is under 1% of the end-to-end time.
- **Cost is $0** because the model runs locally and the gateway prices the Ollama backend at
  zero. Token counts are recorded so cost can be computed for a priced backend.
- **Use `127.0.0.1` rather than `localhost` for OpenSearch on Windows.** An earlier run with
  `localhost` measured 48 ms retrieval p50, but OpenSearch itself took about 2.5 ms. The rest
  was Docker Desktop's IPv6 (`::1`) port forwarding. The defaults now use `127.0.0.1`, which
  brings retrieval to 6 ms.

## Milestones

Milestones follow [SHARED_STANDARDS.md](../SHARED_STANDARDS.md). Each one has a definition of done there.

### M1: Scaffold & baseline
- [x] Repo scaffold: `pyproject.toml`, Makefile, `.gitignore`, `.env.example`
- [x] Smoke test passes (`make test`)
- [x] FastAPI app with `/health` and `/ready`; Dockerfile + `docker compose up` (API, OpenSearch, Qdrant)
- [x] GitHub Actions: lint + test + compose smoke test
- [x] Baseline: BM25-only retrieval + single gateway call on a toy corpus (measured; see Evaluation & Results)

### M2: Core functionality
- [x] Gateway client (OpenAI-compatible, configured via `GATEWAY_BASE_URL`): chat + embeddings, token usage and cost from gateway headers
- [x] LangGraph flow: retrieve → rerank → generate with citations → verify, abstaining on weak evidence
- [x] FastAPI `/ask` and `/summarize` endpoints (verified live against the gateway on the toy corpus)

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
├── .github/workflows/ci.yml   # lint, test, compose smoke test
├── src/healthcare_rag/
│   ├── __init__.py
│   ├── baseline.py            # M1 baseline: BM25 top-k -> one gateway chat call
│   ├── bm25.py                # OpenSearch index (idempotent upsert) + BM25 search
│   ├── config.py              # pydantic-settings, reads .env
│   ├── evaluation.py          # recall@k, MRR, keyword accuracy, latency; CLI
│   ├── gateway.py             # OpenAI-compatible client for 01-llm-gateway
│   └── main.py                # FastAPI app: /health, /ready
├── eval/toy/                  # self-written toy corpus + questions (small, committed)
├── docs/baseline_results.md   # generated per-question baseline report
├── tests/
│   ├── integration/           # needs OpenSearch (make test-integration)
│   ├── test_api.py            # TestClient, backends mocked
│   ├── test_baseline.py       # metrics, pipeline and gateway client with fakes
│   └── test_smoke.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```
