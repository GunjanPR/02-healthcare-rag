# Proposed resume bullets: Healthcare RAG

Every number below is measured and comes from [evaluation_report.md](evaluation_report.md),
the README or `docs/benchmarks/` (test split; 2026-10-03 for retrieval, 2026-10-19 for the
qwen2.5:7b generator and hand labels; RTX 4050 laptop GPU held to 20 W). Keep the
qualifiers if you shorten a bullet. Do not round the confidence-interval caveat away.

## Recommended bullets (pick 2–3)

- Built a citation-grounded medical RAG service over 500 PubMed Central articles
  (FastAPI, LangGraph, OpenSearch, Qdrant). Hybrid BM25 + dense retrieval with MedCPT
  reranking improved NDCG@10 by 24% (0.592 → 0.734) and Recall@5 from 0.717 to 0.891
  against a BM25-only baseline across 92 held-out questions.
- Raised answer accuracy from 77.2% to 88.0% against the BM25-only baseline on 92
  held-out questions (84.8% after hand-checking the keyword grader). Every answer carries
  passage-level citations, and the system declined 14/14 out-of-scope questions with a
  dev-tuned reranker-score gate that skips the LLM call.
- Built a reproducible evaluation harness (`make eval`) for a 159-question dev/test set.
  It reports recall@k, MRR, NDCG@10, citation accuracy and abstention, with paired
  bootstrap confidence intervals. An error analysis on dev added a citation-repair step
  that lifted test accuracy from 81.5% to 87.0%.
- Routed all inference through a self-built LLM gateway and instrumented the service with
  Prometheus/Grafana and OpenTelemetry. A single Jaeger trace spans the RAG API and the
  gateway, and alerts cover latency, abstention and citation failures. p95 latency is
  8.5 s on a 20 W laptop GPU at $0 per query (7B generator).
- Raised answer faithfulness from 88.6% to 95.3% (blind hand labels on every answered test
  question, n = 85-88), past the project's 90% target, by choosing a 7B generator on the dev
  split after prompt, citation-relabeling and 3B-model variants failed there; cost was +1.5 s
  p50 latency on a 20 W GPU. *(docs/benchmarks/2026-10-19-generator)*
- Showed that LLM-as-judge faithfulness scores could not be trusted on their own: against 88
  blind hand labels the best judge reached kappa 0.58 (a 20-item audit had suggested 0.83),
  a claim-by-claim prompt did worse (0.31), and a judge from the generator's own model
  family caught 0 of 4 unfaithful answers. *(docs/benchmarks/2026-10-19-faithfulness)*
- Audited the keyword-graded accuracy metric by hand-checking every answer where it and an
  LLM judge disagreed (38 across both generators): keywords overstate the full system's
  accuracy by 3-4 points, and the judge is too strict to replace them.

## Shorter variants

- Medical RAG over PubMed Central: +24% NDCG@10 and +10.9 pts answer accuracy vs BM25-only
  (92 held-out questions); 95.3% of answers faithful by hand labels; declined 14/14
  out-of-scope questions.
- Hybrid retrieval + MedCPT reranking: Recall@5 0.717 → 0.891 vs BM25 on 92 held-out
  biomedical questions.

## Caveats to keep in mind (be ready to discuss)

- The answer-accuracy gain has a 95% CI of +0.0 to +20.7 points (N = 92). Say "improved",
  not "significantly improved". The NDCG@10 gain is clearly significant (CI +0.08 to
  +0.21).
- Questions and reference keywords were written by an AI from sampled passages, not by
  clinicians. Accuracy is keyword-graded.
- Faithfulness 95.3% is from hand labels by Claude (an AI), not clinicians. Its 95% CI is
  88.5-98.2, so the 90% target is met on the point estimate only, and the gain over the 3B
  generator is not significant on the same 85 questions (McNemar p = 0.39). Two of the 3B
  model's unfaithful answers became abstentions with the 7B model.
- Hand-corrected accuracy is 84.8% (full) vs 77.2% (baseline). The 88.0% headline is
  keyword-graded; quote the corrected figure if pressed.
- p95 latency is 8.5 s with the 7B generator (7.6 s with the 3B one).
- PubMedQA: 61.8% for both the full system and the baseline (no gain; retrieval was not the
  bottleneck there). Don't cite it as an improvement.
- Latency was measured on one consumer laptop GPU at a 20 W cap with requests run one at
  a time. It is not a throughput benchmark.
