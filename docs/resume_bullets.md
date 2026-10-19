# Proposed resume bullets: Healthcare RAG

Every number below is measured and comes from [evaluation_report.md](evaluation_report.md)
or the README (test split, 2026-10-03, RTX 4050 laptop GPU held to 20 W). Keep the
qualifiers if you shorten a bullet. Do not round the confidence-interval caveat away.

## Recommended bullets (pick 2–3)

- Built a citation-grounded medical RAG service over 500 PubMed Central articles
  (FastAPI, LangGraph, OpenSearch, Qdrant). Hybrid BM25 + dense retrieval with MedCPT
  reranking improved NDCG@10 by 24% (0.592 → 0.734) and Recall@5 from 0.717 to 0.891
  against a BM25-only baseline across 92 held-out questions.
- Raised answer accuracy from 77.2% to 87.0% against the BM25-only baseline on 92
  held-out questions. Every answer carries passage-level citations, and the system
  declined 14/14 out-of-scope questions with a dev-tuned reranker-score gate that skips
  the LLM call.
- Built a reproducible evaluation harness (`make eval`) for a 159-question dev/test set.
  It reports recall@k, MRR, NDCG@10, citation accuracy and abstention, with paired
  bootstrap confidence intervals. An error analysis on dev added a citation-repair step
  that lifted test accuracy from 81.5% to 87.0%.
- Routed all inference through a self-built LLM gateway and instrumented the service with
  Prometheus/Grafana and OpenTelemetry. A single Jaeger trace spans the RAG API and the
  gateway, and alerts cover latency, abstention and citation failures. p95 latency is
  7.6 s on a 20 W laptop GPU at $0 per query.
- Measured answer faithfulness with a second-pass LLM judge from another model family
  (Qwen2.5 7B grading Llama 3.2 3B answers) after validating it against hand labels:
  95% agreement, Cohen's kappa 0.83, against 0.30 for the generator judging itself. 86.4%
  of answers were supported by their cited passages (95% CI 77.7-92.0%, n = 88).
  *(docs/benchmarks/2026-10-19-faithfulness)*
- Audited the keyword-graded accuracy metric by hand-checking all 27 answers where it and
  the LLM judge disagreed, which showed keywords overstate accuracy by about 4 points
  (87.0% → 82.6%) and that the judge is too strict to replace them (right in 6 of 27).

## Shorter variants

- Medical RAG over PubMed Central: +24% NDCG@10 and +9.8 pts answer accuracy vs BM25-only
  (92 held-out questions); declined 14/14 out-of-scope questions.
- Hybrid retrieval + MedCPT reranking: Recall@5 0.717 → 0.891 vs BM25 on 92 held-out
  biomedical questions.

## Caveats to keep in mind (be ready to discuss)

- The answer-accuracy gain has a 95% CI of +0.0 to +19.6 points (N = 92). Say "improved",
  not "significantly improved". The NDCG@10 gain is clearly significant (CI +0.08 to
  +0.21).
- Questions and reference keywords were written by an AI from sampled passages, not by
  clinicians. Accuracy is keyword-graded.
- Faithfulness (86.4%) is measured by an LLM judge validated on only 20 hand labels, 3 of
  them unfaithful. It is below the project's own >90% target; say so if asked.
- Hand-corrected accuracy is 82.6% (full) vs 75.0% (baseline). The 87.0% headline is
  keyword-graded; quote the corrected figure if pressed.
- PubMedQA: 61.8% for both the full system and the baseline (no gain; retrieval was not the
  bottleneck there). Don't cite it as an improvement.
- Latency was measured on one consumer laptop GPU at a 20 W cap with requests run one at
  a time. It is not a throughput benchmark.
