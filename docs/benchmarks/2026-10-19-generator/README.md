# Generator: llama3.2 (3B) vs qwen2.5:7b-instruct (2026-10-19)

The full system's answers were 88.6% faithful by hand (78/88, see
[../2026-10-19-faithfulness](../2026-10-19-faithfulness/README.md) section 4), short of the
>90% target. The ten unfaithful answers misread a number or group (4), added claims the
passages do not make (4), denied information the passages had (1) or reversed a finding (1).
Those are reading errors of a 3B model, so this round changed the pipeline on dev first and
ran test once with the chosen configuration.

**Setup:** Acer Nitro V 15 on AC power (`BatteryStatus` 2), RTX 4050 Laptop 6 GB held to 20 W,
Ollama serving through the LLM gateway (project 01), MedCPT reranker on the same GPU. qwen2.5:7b
(Q4_K_M, 4.7 GB) does not fit next to the other models, so Ollama keeps 18% of it on the CPU;
a 3,072-token context did not change that.

## 1. Dev split: what was tried (47 answerable + 6 out-of-scope questions)

Faithfulness here is the one-word qwen2.5:7b judge (`dev_*__faithfulness_judge.json`),
except the last column, which is hand-labeled (`labels_dev.json`).

| Variant | Accuracy (keywords) | Answered | Judge: faithful | Hand: faithful |
|---|---|---|---|---|
| llama3.2, prompt v1 (shipped) | 93.6% | 97.9% | 89.1% | 44/46 |
| + sentence-level citation relabel | 93.6% | 97.9% | 87.0% | - |
| + drop sentences without lexical support | 74.5% | 97.9% | 80.4% | - |
| prompt v2 + relabel | 91.5% | 100% | 87.2% | - |
| terse prompt ("answer only what is asked, 1-2 sentences") | 95.7% | 97.9% | 82.6% | - |
| qwen2.5:3b-instruct | 51.1% | 57.4% | 88.9% | - |
| **qwen2.5:7b-instruct, prompt v1** | **93.6%** | 95.7% | **97.8%** | **45/45** |
| qwen2.5:7b-instruct, terse prompt | 89.4% | 95.7% | 93.3% | - |

- Relabeling or dropping sentences with the lexical support scorer does not help: the
  scorer is too crude to tell a wrong claim from a paraphrase, and dropping costs 19 points of
  accuracy.
- The terse prompt made llama3.2 answer with fragments such as "82.9% [2]", which the judge
  cannot check, and cost qwen 4.2 points of accuracy. It was not kept.
- qwen2.5:3b declined or failed to cite on 43% of answerable questions.
- qwen2.5:7b with the unchanged prompt was the only change that helped, and the hand labels
  confirmed it (45/45 vs 44/46). It became the candidate for one test run.

## 2. Test split, run once (92 answerable + 14 out-of-scope questions)

All three systems re-run with qwen2.5:7b (`answers_qwen7b.json`); the llama3.2 numbers are the
2026-10-03 run. Retrieval, reranking, the abstention threshold and the prompt are unchanged.

| Full system | llama3.2 (3B) | qwen2.5:7b | Difference (95% CI or test) |
|---|---|---|---|
| **Faithful, hand-labeled** | 88.6% (78/88) | **95.3% (81/85)**, CI 88.5-98.2 | same 85 questions: 77 vs 81, McNemar p = 0.39 |
| Answer accuracy, keywords | 87.0% | 88.0% | +1.1 pts (-4.3 to +6.5) |
| Answer accuracy, adjudicated | 82.6% | 84.8% | +2.2 pts |
| Accuracy when answered | 90.9% | 95.3% | |
| Answered rate | 95.7% | 92.4% | qwen declined q045, q072, q101 |
| Cites the source chunk | 63.6% | 75.3% | same 85 questions: 56 vs 64, McNemar p = 0.12 |
| Cites the source article | 98.9% | 100% | |
| Out-of-scope declined | 14/14 | 14/14 | |
| Latency p50 / p95 | 4.1 s / 7.6 s | 5.5 s / 8.5 s | |
| Mean prompt / completion tokens | 1,689 / 50 | 1,604 / 34 | |
| Cost per query (gateway) | $0 | $0 | |

Against the baseline with the same generator (M1, BM25 top 3, one call): keyword accuracy
77.2% → 88.0%, +10.9 points (paired bootstrap 95% CI +0.0 to +20.7); adjudicated 77.2% →
84.8%.

What this does and does not show:

- **The >90% faithfulness target is met on the point estimate (95.3%), not with confidence:**
  the 95% interval reaches down to 88.5%, and the paired difference against llama3.2 on the
  same questions is not significant (p = 0.39). Two of llama's ten unfaithful answers became
  abstentions with qwen (q045, q072), so part of the gain is declining, not answering better.
- The four unfaithful qwen answers: whole-cohort subtypes presented as the men's subtypes
  (q008), an overall claim the passages make only for patients 65+ (q018), the CISS
  definition presented as TOAST's (q104), and swapped percentages (q131). Notes are in
  `eval/qa/faithfulness_labels_test_qwen7b.json`.
- **A judge cannot grade its own family.** The qwen2.5:7b judge also gives qwen's answers
  95.3%, but it caught 0 of the 4 unfaithful ones (kappa -0.05 against the hand labels; it
  agreed only by cancelling errors). That is why the faithfulness figure above comes from hand
  labels, and why the judge is reported only as a cross-check.
- Accuracy and citation gains point the same way but are within noise at N = 92.
- The cost is about 1.5 s more at p50 on this GPU, because 18% of the 7B model runs on the CPU.

## 3. Correctness adjudication (`answers_qwen7b__correctness_test.json`)

The keyword grader and the qwen2.5:7b correctness judge disagree on 11 of 276 answers (27 with
llama3.2). Read by Claude, with the same rules as before (partly right counts as wrong):

| Who was right | Answers |
|---|---|
| Keywords (judge too strict) | m1: q027, q072, q093; bm25: q008, q093; full: q027, q096 |
| Judge, keyword false negative | bm25: q122 ("Thirty-seven ... four") |
| Judge, keyword false positive | full: q008 (whole-cohort classes given as the men's) |
| Partly right, counted wrong | full: q060 (SIRT1 only, nothing on Klotho), q131 (right direction, swapped numbers) |

Adjudicated accuracy: M1 baseline 77.2% (71/92), BM25 RAG 83.7% (77/92), full system 84.8%
(78/92).

## Reproduce

```
# gateway (01) must serve the model: OLLAMA_MODELS=...,qwen2.5:7b-instruct
python -m healthcare_rag.experiments answers qwen7b --bench pmc --split all \
    --systems m1_baseline,bm25_rag,full --set CHAT_MODEL=qwen2.5:7b-instruct      # 42 min
python -m healthcare_rag.judge faithfulness eval/results/experiments/qwen7b__pmc__all.json \
    --split test --scorer judge                                                    # 12 min
python -m healthcare_rag.judge correctness eval/results/experiments/qwen7b__pmc__all.json \
    --split test                                                                   # 4 min
python -m healthcare_rag.judge labels <faithfulness run> \
    --labels eval/qa/faithfulness_labels_test_qwen7b.json
```

The terse prompt (`PROMPT_VERSION=v3` in `dev_v3.json` and `dev_qwen_v3.json`) was removed
from the code after this test. Its rules were: answer only what the question asks, in one or
two sentences, with no background, other findings, limitations or advice; state only what a
passage states, copying numbers, units, groups and outcomes exactly; end every sentence with
the supporting passage number; reply `INSUFFICIENT_EVIDENCE` if the passages lack the answer.
