# Demo transcript

Recorded 2026-10-03 05:08 UTC against the full system (hybrid retrieval, MedCPT rerank, llama3.2 via the LLM gateway) by `scripts/demo.py`. Responses are real and only trimmed (long snippets shortened).

The API uses the gateway's response cache (`GATEWAY_BYPASS_CACHE=false`), so a question that was asked before (the paraphrased one was part of the dashboard run) is answered from cache in a few hundred milliseconds. Measured latencies without the cache are in [evaluation_report.md](evaluation_report.md). The answers are model output and can contain errors; check the cited passages.

## Liveness

`GET /health` -> **200** in 0.00 s

Response:
```json
{
  "status": "ok",
  "service": "healthcare-rag",
  "version": "0.1.0"
}
```

## Readiness: OpenSearch and Qdrant reachable, gateway reported

`GET /ready` -> **200** in 0.63 s

Response:
```json
{
  "status": "ready",
  "checks": {
    "opensearch": "ok",
    "qdrant": "ok"
  },
  "gateway": "ok"
}
```

## Answerable question: cited answer with PMC links

`POST /ask` -> **200** in 3.16 s

Request:
```json
{
  "question": "Which gyrA mutation was most common among ofloxacin-resistant multidrug-resistant TB strains in Morocco?"
}
```

Response:
```json
{
  "answer": "The substitution of GAC > GCC at codon 94 (Asp91Ala) was the most recorded mutation in the QRDR region of gyrA gene, accounting for 47.6% (10/21) among the 20 phenotypic OFX-resistant strains [1, 5].",
  "abstained": false,
  "reason": "answered",
  "citations": [
    {
      "n": 1,
      "chunk_id": "PMC5830342:12",
      "pmcid": "PMC5830342",
      "title": "Molecular characterization of mutations associated with resistance to second-line tuberculosis drug among multidrug-resistant tuberculosis patients from high prevalence tuberculosis city in Morocco",
      "section": "discussion",
      "year": 2018,
      "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5830342/",
      "snippet": "FQs are the mainstay of treatment for patients with MDR and XDR TB since their inclusion in therapeutic regimens improves treatment outcome. However, resistance..."
    },
    {
      "n": 5,
      "chunk_id": "PMC5830342:0",
      "pmcid": "PMC5830342",
      "title": "Molecular characterization of mutations associated with resistance to second-line tuberculosis drug among multidrug-resistant tuberculosis patients from high prevalence tuberculosis city in Morocco",
      "section": "abstract",
      "year": 2018,
      "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5830342/",
      "snippet": "The emergence of extensively drug-resistant tuberculosis (XDR-TB) has raised public health concern for global TB control. Although multi drug-resistant tubercul..."
    }
  ],
  "timings_ms": {
    "retrieve": 66.5,
    "rerank": 444.7,
    "generate": 2638.6,
    "total": 3155.2
  },
  "usage": {
    "prompt_tokens": 1911,
    "completion_tokens": 64,
    "cost_usd": 0.0,
    "llm_calls": 1
  }
}
```

## Paraphrased question that BM25 ranks poorly (relevant passage not in its top 10)

`POST /ask` -> **200** in 0.29 s

Request:
```json
{
  "question": "In children aged 12 and older, is the Asthma Control Test more useful for ruling out or for confirming exercise-induced bronchoconstriction?"
}
```

Response:
```json
{
  "answer": "According to the discussion section [4], ACT scores are more effective as a means of excluding (ruling out) EIB in asthmatic patients aged 12 years and older, rather than confirming it.",
  "abstained": false,
  "reason": "answered",
  "citations": [
    {
      "n": 4,
      "chunk_id": "PMC4782564:13",
      "pmcid": "PMC4782564",
      "title": "Asthma Control Test and Bronchial Challenge with Exercise in Pediatric Asthma",
      "section": "discussion",
      "year": 2016,
      "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4782564/",
      "snippet": "The relationship between ACT scores and EIB has rarely been assessed in asthmatic children, particularly in those aged 12 years and above. No studies have compa..."
    }
  ],
  "timings_ms": {
    "retrieve": 62.7,
    "rerank": 213.0,
    "generate": 6.6,
    "total": 285.0
  },
  "usage": {
    "prompt_tokens": 1622,
    "completion_tokens": 43,
    "cost_usd": 0.0,
    "llm_calls": 1
  }
}
```

## Out-of-scope question: declined by the rerank-score gate, no LLM call

`POST /ask` -> **200** in 0.32 s

Request:
```json
{
  "question": "Which bacterium causes Lyme disease?"
}
```

Response:
```json
{
  "answer": "I can't answer this from the indexed literature with enough confidence.",
  "abstained": true,
  "reason": "low_score",
  "citations": [],
  "timings_ms": {
    "retrieve": 49.4,
    "rerank": 267.5,
    "total": 318.7
  },
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost_usd": 0.0,
    "llm_calls": 0
  }
}
```

## Metadata filters: only results sections of osteoporosis articles from 2020 on

`POST /ask` -> **200** in 3.25 s

Request:
```json
{
  "question": "What share of patients started osteoporosis treatment after a fracture liaison service was introduced?",
  "filters": {
    "topics": [
      "osteoporosis"
    ],
    "sections": [
      "results"
    ],
    "year_min": 2020
  }
}
```

Response:
```json
{
  "answer": "The osteoporosis treatment initiation rate in fiscal year 2022 was 97.8% [1, 2].",
  "abstained": false,
  "reason": "answered",
  "citations": [
    {
      "n": 1,
      "chunk_id": "PMC10497696:9",
      "pmcid": "PMC10497696",
      "title": "Admission screening form and osteoporosis educational appointment: a novel fracture liaison service system for identifying osteoporosis patients and facilitating medication initiation",
      "section": "results",
      "year": 2023,
      "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10497696/",
      "snippet": "The results of this study are presented using fiscal years: April 2020 to March 2021, April 2021 to March 2022, and April 2022 to March 2023 are referred to as ..."
    },
    {
      "n": 2,
      "chunk_id": "PMC10497696:10",
      "pmcid": "PMC10497696",
      "title": "Admission screening form and osteoporosis educational appointment: a novel fracture liaison service system for identifying osteoporosis patients and facilitating medication initiation",
      "section": "results",
      "year": 2023,
      "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10497696/",
      "snippet": "There were 504 intervention orders, with 131 (26.0%) for orthopedic and 373 (74.0%) for non-orthopedic patients. DEXA examinations were performed on 346 (68.6%)..."
    }
  ],
  "timings_ms": {
    "retrieve": 76.8,
    "rerank": 296.5,
    "generate": 2870.8,
    "total": 3247.4
  },
  "usage": {
    "prompt_tokens": 1889,
    "completion_tokens": 26,
    "cost_usd": 0.0,
    "llm_calls": 1
  }
}
```

## Summarize one article

`POST /summarize` -> **200** in 11.23 s

Request:
```json
{
  "pmcid": "PMC10497696"
}
```

Response:
```json
{
  "answer": "Here is a summary of the article in 4-6 sentences:\n\nThe authors developed a novel fracture liaison service (FLS) system to identify osteoporosis patients and facilitate medication initiation, which aims to prevent both primary and secondary fractures. The system uses an admission screening form and osteoporosis educational appointments conducted by rehabilitation staff to identify patients at high risk of fracture and initiate treatment. The FLS team analyzed data from April 2020 to March 2023 and found that the average monthly number of dual-energy X-ray absorptiometry (DEXA) scans increased significantly, with a 100% increase in DEXA rate among proximal femoral fracture patients. The rate of treatment initiation also increased remarkably, from 21.7% in 2020 to 97.8% in 2022. The FLS system was effective in identifying osteoporosis patients and initiating medication, with 251 patients diagnosed with osteoporosis and 134 successfully starting medication. The study demonstrates the impact of the FLS team's interventions on fracture prevention and highlights the importance of secondary fracture prevention in osteoporosis management.",
  "abstained": false,
  "reason": "answered",
  "citations": [],
  "timings_ms": {
    "generate": 11198.4,
    "total": 11198.5
  },
  "usage": {
    "prompt_tokens": 2587,
    "completion_tokens": 228,
    "cost_usd": 0.0,
    "llm_calls": 1
  }
}
```

## Input validation

`POST /ask` -> **422** in 0.00 s

Request:
```json
{
  "question": ""
}
```

Response:
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": [
        "body",
        "question"
      ],
      "msg": "String should have at least 3 characters",
      "input": "",
      "ctx": {
        "min_length": 3
      }
    }
  ]
}
```

## Metrics after the demo

`GET /metrics` (excerpt):
```
rag_requests_total{endpoint="ask",outcome="ok"} 5.0
rag_requests_total{endpoint="summarize",outcome="ok"} 1.0
rag_answers_total{reason="low_score"} 2.0
rag_answers_total{reason="answered"} 3.0
```
