"""M4 evaluation on the PMC corpus: retrieval, abstention and answers vs the BM25 baseline.

Usage (needs OpenSearch + Qdrant + the gateway; the reranker needs the [rerank] extra):
    python -m healthcare_rag.rag_eval retrieval   # rankings for every retrieval system
    python -m healthcare_rag.rag_eval tune        # pick ABSTAIN_THRESHOLD on the dev split
    python -m healthcare_rag.rag_eval answers     # generate + grade answers per system
    python -m healthcare_rag.rag_eval report      # docs/evaluation_report.md
    python -m healthcare_rag.rag_eval all

Question set: eval/qa/questions.json. Each answerable question was written from one
sampled source passage, which is its relevant chunk; its article is the relevant document.
Only that one chunk counts as relevant, so other chunks that also contain the answer
count as misses for every system alike (recall is a lower bound). Out-of-scope questions
have no relevant chunk and should be declined.
"""

import argparse
import json
import math
import platform
import random
import re
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from healthcare_rag.baseline import BM25Baseline
from healthcare_rag.config import Settings, get_settings
from healthcare_rag.evaluation import keywords_match, percentile
from healthcare_rag.factory import build_bm25, build_retriever, build_scorer, rag_config
from healthcare_rag.gateway import GatewayClient
from healthcare_rag.rag import RAGPipeline, number_passages, parse_citations
from healthcare_rag.types import Hit

QUESTIONS = Path("eval/qa/questions.json")
RESULTS = Path("eval/results")
REPORT = Path("docs/evaluation_report.md")
KS = (1, 5, 10, 20)
DEPTH = 20  # ranked list length kept per system

REFUSAL = re.compile(
    r"insufficient_evidence|don't know|do not know|not (?:contain|provide|mention|include)"
    r"|no (?:information|mention)|cannot (?:answer|determine)|can't answer|not enough information",
    re.IGNORECASE,
)

JUDGE_PROMPT = (
    "You check whether an answer is supported by source passages. Reply with exactly one "
    "word: YES if every factual claim in the answer is stated in or directly follows from "
    "the passages, otherwise NO."
)


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    answerable: bool
    topic: str
    question: str
    relevant_chunks: list[str]
    relevant_docs: list[str]
    answer_keywords: list[list[str]]


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    return [Question(**q) for q in json.loads(path.read_text(encoding="utf-8"))]


# ---- metrics -------------------------------------------------------------------------------


def first_rank(ranked: list[str], relevant: list[str]) -> int | None:
    return next((i for i, cid in enumerate(ranked, start=1) if cid in relevant), None)


def ndcg_at_k(ranked: list[str], relevant: list[str], k: int = 10) -> float:
    """Binary-relevance NDCG@k."""
    dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(ranked[:k], start=1) if c in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(ranked: list[str], q: Question) -> dict[str, float]:
    rank = first_rank(ranked, q.relevant_chunks)
    docs = [cid.split(":")[0] for cid in ranked]
    return {
        **{f"recall@{k}": float(rank is not None and rank <= k) for k in KS},
        "mrr@10": 1 / rank if rank and rank <= 10 else 0.0,
        "ndcg@10": ndcg_at_k(ranked, q.relevant_chunks, 10),
        "doc_hit@5": float(any(d in q.relevant_docs for d in docs[:5])),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 7) -> tuple:
    """95% CI of mean(b - a) by resampling questions with replacement."""
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    boots = sorted(mean([rng.choice(diffs) for _ in diffs]) for _ in range(n))
    return mean(diffs), boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def choose_threshold(answerable: list[float], unanswerable: list[float]) -> dict[str, float]:
    """Threshold on the top reranker score that maximizes balanced accuracy of
    "answer" (answerable) vs "abstain" (unanswerable). Ties go to the lowest threshold."""
    candidates = sorted(set(answerable + unanswerable))
    best: dict[str, float] = {}
    for i, t in enumerate(candidates):
        # threshold just above the previous value: scores >= t are answered
        thr = t if i == 0 else (candidates[i - 1] + t) / 2
        keep = mean([float(s >= thr) for s in answerable])
        refuse = mean([float(s < thr) for s in unanswerable])
        bal = (keep + refuse) / 2
        if not best or bal > best["balanced_accuracy"]:
            best = {
                "threshold": thr,
                "balanced_accuracy": bal,
                "answerable_kept": keep,
                "unanswerable_refused": refuse,
            }
    return best


# ---- retrieval -----------------------------------------------------------------------------


def _timed(fn: Callable[..., list[Hit]], *args: Any, **kwargs: Any) -> tuple[list[Hit], float]:
    start = time.perf_counter()
    hits = fn(*args, **kwargs)
    return hits, time.perf_counter() - start


def run_retrieval(settings: Settings, questions: list[Question]) -> dict[str, Any]:
    from healthcare_rag.factory import build_reranker

    retrievers = {m: build_retriever(settings, m) for m in ("bm25", "dense", "hybrid")}
    reranker = build_reranker(settings)
    if reranker is None:
        raise SystemExit("set RERANKER_MODEL to evaluate reranking")
    depths = (20, 50)
    for r in retrievers.values():  # warm-up: model loads, connections, CUDA kernels
        r.search("warm-up query about asthma", k=5)
    reranker.rerank("warm-up", retrievers["bm25"].search("asthma control", k=8))

    rows = []
    for q in questions:
        row: dict[str, Any] = {"id": q.id, "systems": {}}
        for name, r in retrievers.items():
            hits, secs = _timed(r.search, q.question, k=max(depths))
            row["systems"][name] = {
                "ranked": [h.id for h in hits[:DEPTH]],
                "latency_s": secs,
            }
            if name in ("bm25", "hybrid"):
                for depth in depths:
                    cands = hits[:depth]
                    reranked, rsecs = _timed(reranker.rerank, q.question, cands)
                    row["systems"][f"{name}+rerank@{depth}"] = {
                        "ranked": [h.id for h in reranked[:DEPTH]],
                        "latency_s": secs + rsecs,
                        "top_score": reranked[0].score if reranked else float("-inf"),
                    }
        rows.append(row)
        print(f"  {q.id} done", flush=True)
    return {"meta": _meta(settings), "rows": rows}


def summarize_retrieval(data: dict, questions: list[Question], split: str) -> dict[str, dict]:
    in_split = {q.id for q in questions if split == "all" or q.split == split}
    qs = {q.id: q for q in questions if q.answerable and q.id in in_split}
    out: dict[str, dict] = {}
    systems = list(data["rows"][0]["systems"])
    for name in systems:
        per_q = [
            retrieval_metrics(r["systems"][name]["ranked"], qs[r["id"]])
            for r in data["rows"]
            if r["id"] in qs
        ]
        lat = [r["systems"][name]["latency_s"] for r in data["rows"] if r["id"] in in_split]
        out[name] = {k: mean([m[k] for m in per_q]) for k in per_q[0]}
        out[name] |= {"n": len(per_q), "p50_s": percentile(lat, 50), "p95_s": percentile(lat, 95)}
    return out


# ---- answers -------------------------------------------------------------------------------


def _answer_row(q: Question, result: Any, system: str) -> dict[str, Any]:
    return {
        "id": q.id,
        "system": system,
        "answer": result["answer"],
        "abstained": result["abstained"],
        "reason": result["reason"],
        "passages": result["passages"],
        "cited": result["cited"],
        "top_score": result.get("top_score"),
        "latency_s": result["latency_s"],
        "generate_s": result.get("generate_s", 0.0),
        "timings": result.get("timings", {}),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "cost_usd": result["cost_usd"],
    }


def build_answer_systems(settings: Settings, threshold: float) -> dict[str, Callable]:
    from healthcare_rag.factory import build_reranker

    chat = GatewayClient.from_settings(settings, bypass_cache=True)
    bm25 = build_bm25(settings)
    hybrid = build_retriever(settings, "hybrid")
    reranker = build_reranker(settings)
    scorer = build_scorer(settings)
    base_cfg = rag_config(settings)
    bm25_cfg = replace(base_cfg, retrieval_top_k=base_cfg.rerank_top_k)
    full_cfg = replace(base_cfg, abstain_threshold=threshold)
    m1 = BM25Baseline(bm25, chat, k=3)
    bm25_rag = RAGPipeline(bm25, chat, None, bm25_cfg, scorer)
    full = RAGPipeline(hybrid, chat, reranker, full_cfg, scorer)

    def run_m1(question: str, instruction: str = "") -> dict[str, Any]:
        start = time.perf_counter()
        r = m1.answer(question, instruction)
        n = len(r.hits)
        return {
            "answer": r.answer,
            "abstained": bool(REFUSAL.search(r.answer)),
            "reason": "refusal_text" if REFUSAL.search(r.answer) else "answered",
            "passages": [h.id for h in r.hits],
            "passage_text": number_passages(r.hits),
            "cited": [r.hits[i - 1].id for i in parse_citations(r.answer, n)],
            "latency_s": time.perf_counter() - start,
            "generate_s": r.chat.latency_s,
            "prompt_tokens": r.chat.prompt_tokens,
            "completion_tokens": r.chat.completion_tokens,
            "cost_usd": r.chat.cost_usd,
        }

    def runner(pipe: RAGPipeline) -> Callable[[str], dict[str, Any]]:
        def run(question: str, instruction: str = "") -> dict[str, Any]:
            r = pipe.ask(question, instruction=instruction)
            return {
                "answer": r.raw_answer if r.abstained and r.raw_answer else r.answer,
                "abstained": r.abstained,
                "reason": r.reason,
                "passages": [h.id for h in r.passages],
                "passage_text": number_passages(r.passages),
                "cited": [h.id for h in r.cited_passages],
                "top_score": r.passages[0].score if r.passages and pipe.reranker else None,
                "latency_s": r.timings["total"],
                "generate_s": r.timings.get("generate", 0.0),
                "timings": r.timings,
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
                "cost_usd": r.usage.cost_usd,
                "llm_calls": r.usage.llm_calls,
                "attributed": r.attributed,
                "relabeled": r.relabeled,
                "support_rate": r.support_rate,
            }

        return run

    return {"m1_baseline": run_m1, "bm25_rag": runner(bm25_rag), "full": runner(full)}


def judge(chat: GatewayClient, passages: str, answer: str) -> bool:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Passages:\n{passages}\n\nAnswer:\n{answer}"},
    ]
    reply = chat.chat(messages, max_tokens=3).text.strip().upper()
    return reply.startswith("YES")


def run_answers(settings: Settings, questions: list[Question], threshold: float) -> dict:
    systems = build_answer_systems(settings, threshold)
    judge_chat = GatewayClient.from_settings(settings, bypass_cache=True)
    for run in systems.values():  # warm-up
        run("warm-up: what is asthma?")
    rows = []
    for name, run in systems.items():
        for q in questions:
            r = run(q.question)
            row = _answer_row(q, r, name)
            if not r["abstained"]:
                row["faithful"] = judge(judge_chat, r["passage_text"], r["answer"])
            rows.append(row)
            print(f"  {name} {q.id} {'ABSTAIN' if r['abstained'] else 'answer'}", flush=True)
    return {"meta": _meta(settings) | {"abstain_threshold": threshold}, "rows": rows}


def grade_answers(data: dict, questions: list[Question], split: str) -> dict[str, dict]:
    qs = {q.id: q for q in questions if split == "all" or q.split == split}
    out: dict[str, dict] = {}
    for name in dict.fromkeys(r["system"] for r in data["rows"]):
        rows = [r for r in data["rows"] if r["system"] == name and r["id"] in qs]
        ans = [r for r in rows if qs[r["id"]].answerable]
        una = [r for r in rows if not qs[r["id"]].answerable]
        answered = [r for r in ans if not r["abstained"]]
        correct = [
            not r["abstained"] and keywords_match(r["answer"], qs[r["id"]].answer_keywords)
            for r in ans
        ]
        abstentions = [r for r in rows if r["abstained"]]
        all_answered = [r for r in rows if not r["abstained"]]
        out[name] = {
            "n_answerable": len(ans),
            "n_unanswerable": len(una),
            "answer_accuracy": mean([float(c) for c in correct]),
            "answered_rate": len(answered) / len(ans),
            "accuracy_when_answered": mean(
                [float(keywords_match(r["answer"], qs[r["id"]].answer_keywords)) for r in answered]
            ),
            "citation_accuracy_chunk": mean(
                [float(any(c in qs[r["id"]].relevant_chunks for c in r["cited"])) for r in answered]
            ),
            "citation_accuracy_doc": mean(
                [
                    float(any(c.split(":")[0] in qs[r["id"]].relevant_docs for c in r["cited"]))
                    for r in answered
                ]
            ),
            "cited_rate": mean([float(bool(r["cited"])) for r in all_answered]),
            "faithful_rate": mean([float(r["faithful"]) for r in all_answered if "faithful" in r]),
            "unanswerable_abstained": mean([float(r["abstained"]) for r in una]),
            "abstention_precision": (
                mean([float(not qs[r["id"]].answerable) for r in abstentions])
                if abstentions
                else float("nan")
            ),
            "p50_s": percentile([r["latency_s"] for r in rows], 50),
            "p95_s": percentile([r["latency_s"] for r in rows], 95),
            "mean_prompt_tokens": mean([r["prompt_tokens"] for r in rows]),
            "mean_completion_tokens": mean([r["completion_tokens"] for r in rows]),
            "cost_usd_per_query": mean([r["cost_usd"] for r in rows]),
        }
    return out


# ---- report --------------------------------------------------------------------------------


def power_state() -> str:
    """ "ac", "battery" or "unknown". Latency numbers only count on AC power: on battery
    this laptop's GPU is throttled. Windows only (Win32_Battery.BatteryStatus 2 = on AC)."""
    if platform.system() != "Windows":
        return "unknown"
    cmd = "(Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not out:
        return "ac"  # no battery: a desktop
    return "ac" if out == "2" else "battery"


def _meta(settings: Settings) -> dict[str, Any]:
    return {
        "power": power_state(),
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "reranker_device": settings.reranker_device,
        "retrieval_top_k": settings.retrieval_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "host": f"{platform.system()} {platform.version()}, Python {platform.python_version()}",
    }


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def _judge_audit_text() -> str:
    path = Path("eval/qa/judge_audit.json")
    if not path.exists():
        return "Treat it as a rough signal."
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    yes = [i for i in items if i["judge"]]
    no = [i for i in items if not i["judge"]]
    return (
        f"A manual check of {len(items)} of its verdicts on the full system "
        "(`eval/qa/judge_audit.json`) confirmed "
        f"{sum(i['manual'] for i in yes)}/{len(yes)} YES verdicts but only "
        f"{sum(not i['manual'] for i in no)}/{len(no)} NO verdicts: it often rejects answers "
        "that repeat the passage nearly verbatim. Its faithfulness rate is therefore a lower "
        "bound, and no faithfulness percentage is claimed from it."
    )


def write_report(questions: list[Question], hardware: str) -> str:
    ret = json.loads((RESULTS / "retrieval.json").read_text(encoding="utf-8"))
    ans = json.loads((RESULTS / "answers.json").read_text(encoding="utf-8"))
    tune = json.loads((RESULTS / "tune.json").read_text(encoding="utf-8"))
    rt = summarize_retrieval(ret, questions, "test")
    at = grade_answers(ans, questions, "test")
    ad = grade_answers(ans, questions, "dev")
    test_ids = [q.id for q in questions if q.split == "test" and q.answerable]
    qmap = {q.id: q for q in questions}

    def per_q(system: str, metric: str) -> list[float]:
        by_id = {r["id"]: r["systems"][system]["ranked"] for r in ret["rows"]}
        return [retrieval_metrics(by_id[i], qmap[i])[metric] for i in test_ids]

    def per_q_correct(system: str) -> list[float]:
        rows = {r["id"]: r for r in ans["rows"] if r["system"] == system}
        return [
            float(
                not rows[i]["abstained"]
                and keywords_match(rows[i]["answer"], qmap[i].answer_keywords)
            )
            for i in test_ids
        ]

    best = "hybrid+rerank@20"
    d_ndcg = paired_bootstrap(per_q("bm25", "ndcg@10"), per_q(best, "ndcg@10"))
    d_acc = paired_bootstrap(per_q_correct("m1_baseline"), per_q_correct("full"))
    n_test = sum(1 for q in questions if q.split == "test")
    n_ans = len(test_ids)

    lines = [
        "# Evaluation report: Healthcare RAG vs the BM25-only baseline",
        "",
        f"Generated by `make eval` on {ans['meta']['date']}. Numbers below are measured; "
        "nothing is estimated.",
        "",
        "## Setup",
        "",
        f"- **Questions:** {len(questions)} in `eval/qa/questions.json`: "
        f"{sum(q.answerable for q in questions)} answerable (14 per topic, 13 for "
        "hypertension), each written from one passage sampled from the corpus, plus "
        f"{sum(not q.answerable for q in questions)} out-of-scope questions the corpus "
        "cannot answer.",
        f"- **Split:** dev = every third question ({sum(q.split == 'dev' for q in questions)}),"
        f" used only to choose the abstention threshold; **test = {n_test}** "
        f"({n_ans} answerable, {n_test - n_ans} out of scope). All tables below are on test.",
        "- **Corpus:** 500 PMC OA articles (CC BY/CC0), 8,369 chunks; see the README.",
        f"- **Models:** generation `{ans['meta']['chat_model']}` via the LLM gateway "
        "(temperature 0, cache bypassed); embeddings "
        f"`{ans['meta']['embedding_model']}` via the gateway; reranker "
        f"`{ans['meta']['reranker_model']}` on {ans['meta']['reranker_device']}.",
        f"- **Hardware:** {hardware}",
        f"- **Host:** {ans['meta']['host']}",
        "",
        "## Retrieval (test, answerable questions, N = " + str(n_ans) + ")",
        "",
        "Relevance = the single source chunk of each question.",
        "",
        "| System | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | NDCG@10 | "
        "Doc hit@5 | Latency p50 / p95 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in rt.items():
        lines.append(
            f"| {name} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['recall@20']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | "
            f"{m['doc_hit@5']:.3f} | {1000 * m['p50_s']:.0f} / {1000 * m['p95_s']:.0f} ms |"
        )
    lines += [
        "",
        f"NDCG@10, {best} minus bm25: {d_ndcg[0]:+.3f} (95% paired bootstrap CI "
        f"{d_ndcg[1]:+.3f} to {d_ndcg[2]:+.3f}).",
        "",
        "Rerank depth: on dev, reranking the top 20 hybrid candidates matched reranking 50 "
        "(same NDCG@10) at less than half the latency, so the full system uses 20.",
        "",
        "## Abstention threshold (dev)",
        "",
        "The full system declines to answer when the best MedCPT score of the reranked "
        "passages is below a threshold. It was chosen on the dev split to maximize "
        "balanced accuracy between keeping answerable questions and refusing out-of-scope "
        "ones:",
        "",
        f"- threshold = **{tune['threshold']:.2f}** (logit); dev balanced accuracy "
        f"{_pct(tune['balanced_accuracy'])}, answerable kept {_pct(tune['answerable_kept'])}, "
        f"out-of-scope refused {_pct(tune['unanswerable_refused'])}.",
        "",
        "## Answers (test)",
        "",
        "- `m1_baseline`: the M1 control. BM25 top 3 passages, one gateway call, no "
        "abstention logic (a reply that says the passages lack the answer is counted as an "
        "abstention).",
        "- `bm25_rag`: the M2 LangGraph flow with BM25 top 5 and no reranker. The answer "
        "must cite a passage or the system abstains.",
        "- `full`: hybrid retrieval (top 20) -> MedCPT rerank -> top 5 -> generate -> "
        "verify, with the abstention threshold from dev.",
        "",
        "| Metric | " + " | ".join(at) + " |",
        "|---|" + "---|" * len(at),
    ]
    rows = [
        ("Answer accuracy, answerable (keyword match; abstain = wrong)", "answer_accuracy", _pct),
        ("Answered rate, answerable", "answered_rate", _pct),
        ("Accuracy when answered", "accuracy_when_answered", _pct),
        ("Citation accuracy: cites the source chunk", "citation_accuracy_chunk", _pct),
        ("Citation accuracy: cites the source article", "citation_accuracy_doc", _pct),
        ("Answers with at least one valid citation", "cited_rate", _pct),
        ("Faithful (LLM judge: claims supported by passages)", "faithful_rate", _pct),
        ("Out-of-scope questions declined", "unanswerable_abstained", _pct),
        ("Abstention precision (declines that were out of scope)", "abstention_precision", _pct),
        ("End-to-end latency p50", "p50_s", lambda x: f"{1000 * x:.0f} ms"),
        ("End-to-end latency p95", "p95_s", lambda x: f"{1000 * x:.0f} ms"),
        ("Mean prompt tokens", "mean_prompt_tokens", lambda x: f"{x:.0f}"),
        ("Mean completion tokens", "mean_completion_tokens", lambda x: f"{x:.0f}"),
        ("Cost per query (gateway-reported)", "cost_usd_per_query", lambda x: f"${x:.5f}"),
    ]
    for label, key, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt(m[key]) for m in at.values()) + " |")
    v1_path = RESULTS / "answers_v1.json"
    if v1_path.exists():
        v1 = grade_answers(json.loads(v1_path.read_text(encoding="utf-8")), questions, "test")
        lines += [
            "",
            "**Iteration history.** The first measured run (v1) had no citation repair: an "
            "answer without citations was discarded. On dev, every such discarded answer "
            "from the full system was otherwise correct, so v2 (above) adds one follow-up "
            "turn asking for citations. v1 on test, for comparison: "
            + "; ".join(
                f"{name} accuracy {_pct(m['answer_accuracy'])}, answered "
                f"{_pct(m['answered_rate'])}, p50 {1000 * m['p50_s']:.0f} ms"
                for name, m in v1.items()
            )
            + ".",
        ]
    lines += [
        "",
        f"Answer accuracy, full minus m1_baseline: {100 * d_acc[0]:+.1f} points (95% paired "
        f"bootstrap CI {100 * d_acc[1]:+.1f} to {100 * d_acc[2]:+.1f}).",
        "",
        "Dev split, for reference (the threshold was tuned here): "
        + "; ".join(
            f"{name} accuracy {_pct(m['answer_accuracy'])}, out-of-scope declined "
            f"{_pct(m['unanswerable_abstained'])}"
            for name, m in ad.items()
        )
        + ".",
        "",
        "## How to read this",
        "",
        "- Keyword grading: an answer is correct when every keyword group of the reference "
        "appears as a whole term. It does not catch a right answer phrased with other "
        "words, and it gives no credit for a refusal on an answerable question.",
        "- The faithfulness judge is the same 3B model that writes the answers. "
        + _judge_audit_text(),
        "- Latency is client-side and sequential on one laptop GPU shared by generation, "
        "embeddings and the reranker. Cost is $0 because the gateway prices the local "
        "Ollama backend at zero; token counts are recorded to price other backends.",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    return text


# ---- CLI -----------------------------------------------------------------------------------


def _save(name: str, data: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(data, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=["retrieval", "tune", "answers", "report", "all"])
    parser.add_argument("--hardware", default="", help="free-text hardware/power note")
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    args = parser.parse_args(argv)
    settings = get_settings()
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    if args.cmd in ("retrieval", "all"):
        _save("retrieval.json", run_retrieval(settings, questions))
        for split in ("dev", "test"):
            print(
                split,
                json.dumps(
                    summarize_retrieval(
                        json.loads((RESULTS / "retrieval.json").read_text()), questions, split
                    ),
                    indent=1,
                ),
            )
    if args.cmd in ("tune", "all"):
        ret = json.loads((RESULTS / "retrieval.json").read_text(encoding="utf-8"))
        qmap = {q.id: q for q in questions}
        scores = {
            r["id"]: r["systems"]["hybrid+rerank@20"]["top_score"]
            for r in ret["rows"]
            if qmap[r["id"]].split == "dev"
        }
        best = choose_threshold(
            [s for i, s in scores.items() if qmap[i].answerable],
            [s for i, s in scores.items() if not qmap[i].answerable],
        )
        _save("tune.json", best)
        print("tuned on dev:", best)
    if args.cmd in ("answers", "all"):
        threshold = json.loads((RESULTS / "tune.json").read_text(encoding="utf-8"))["threshold"]
        data = run_answers(settings, questions, threshold)
        _save("answers.json", data)
        for split in ("dev", "test"):
            print(split, json.dumps(grade_answers(data, questions, split), indent=1))
    if args.cmd in ("report", "all"):
        print(write_report(questions, args.hardware))


if __name__ == "__main__":
    main()
