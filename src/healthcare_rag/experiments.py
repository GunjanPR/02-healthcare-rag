"""Experiment runner: each variant runs with settings overrides (the same names as the
environment variables) and is saved under eval/results/experiments/.

Usage (needs make up and the gateway; tune on dev, report test once per final config):
    # answers for one variant on the PMC questions and/or PubMedQA
    python -m healthcare_rag.experiments answers v2_k8 --bench pmc,pubmedqa --split dev \
        --set PROMPT_VERSION=v2 --set RERANK_TOP_K=8
    # reranker scores of the hybrid top 20 (cheap; no LLM), then NDCG@10 of fusion variants
    python -m healthcare_rag.experiments rerank-scores --split dev \
        --models ncbi/MedCPT-Cross-Encoder,BAAI/bge-reranker-v2-m3
    python -m healthcare_rag.experiments fusion --split dev
    # reranker latency at several batch sizes (AC power only)
    python -m healthcare_rag.experiments rerank-bench --batch-sizes 16,32
    # abstention threshold sweep over a run made with ABSTAIN_MODE=model
    python -m healthcare_rag.experiments sweep model_only --bench pmc --split dev
    python -m healthcare_rag.experiments list
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from healthcare_rag.config import Settings
from healthcare_rag.evaluation import keywords_match, percentile
from healthcare_rag.rag_eval import (
    RESULTS,
    _answer_row,
    _meta,
    grade_answers,
    load_questions,
    mean,
    ndcg_at_k,
)
from healthcare_rag.types import Hit

EXPERIMENTS = RESULTS / "experiments"
BENCHES = ("pmc", "pubmedqa")
DEPTH = 20


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        out[key.strip().lower()] = value.strip()
    unknown = set(out) - set(Settings.model_fields)
    if unknown:
        raise SystemExit(f"unknown settings {sorted(unknown)}")
    return out


def make_settings(overrides: dict[str, str]) -> Settings:
    """Settings from .env and the environment, with the overrides on top (validated)."""
    return Settings(**overrides)


def result_path(name: str, bench: str, split: str) -> Path:
    return EXPERIMENTS / f"{name}__{bench}__{split}.json"


def in_split(split: str, item_split: str) -> bool:
    return split == "all" or item_split == split


# ---- answers -------------------------------------------------------------------------------


def run_pmc(settings: Settings, split: str, systems: tuple[str, ...]) -> dict[str, Any]:
    from healthcare_rag.rag_eval import build_answer_systems

    questions = [q for q in load_questions() if in_split(split, q.split)]
    built = build_answer_systems(settings, settings.abstain_threshold)
    rows = []
    for name in systems:
        run = built[name]
        run("warm-up: what is asthma?")
        for q in questions:
            r = run(q.question)
            rows.append(
                _answer_row(q, r, name)
                | {
                    "llm_calls": r.get("llm_calls", 1),
                    "attributed": r.get("attributed", False),
                }
            )
            print(f"  {name} {q.id} {r['reason']}", flush=True)
    return {"rows": rows}


def summarize_pmc(data: dict, split: str) -> dict[str, dict]:
    questions = load_questions()
    out = grade_answers(data, questions, split)
    qs = {q.id for q in questions if in_split(split, q.split)}
    for name, m in out.items():
        rows = [r for r in data["rows"] if r["system"] == name and r["id"] in qs]
        m["mean_llm_calls"] = mean([r.get("llm_calls", 1) for r in rows])
        m["repair_rate"] = mean([float("repair" in r.get("timings", {})) for r in rows])
        m["attributed_rate"] = mean([float(r.get("attributed", False)) for r in rows])
    return out


def run_pubmedqa(
    settings: Settings, split: str, systems: tuple[str, ...], out: Path
) -> dict[str, Any]:
    from healthcare_rag import pubmedqa

    items = [i for i in pubmedqa.load_items() if in_split(split, i.split)]
    return pubmedqa.run(settings, items, systems, out)


def summarize_pubmedqa(data: dict, split: str) -> dict[str, dict]:
    from healthcare_rag import pubmedqa

    items = {i.pmid: i for i in pubmedqa.load_items()}
    rows = [r for r in data["rows"] if in_split(split, items[r["id"]].split)]
    return pubmedqa.summarize(rows, items)


def cmd_answers(args: argparse.Namespace) -> None:
    overrides = parse_overrides(args.set)
    settings = make_settings(overrides)
    systems = tuple(s for s in args.systems.split(",") if s)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    for bench in args.bench.split(","):
        path = result_path(args.name, bench, args.split)
        start = time.perf_counter()
        if bench == "pmc":
            data = run_pmc(settings, args.split, systems)
            summary = summarize_pmc(data, args.split)
        elif bench == "pubmedqa":
            data = run_pubmedqa(settings, args.split, systems, path)
            summary = summarize_pubmedqa(data, args.split)
        else:
            raise SystemExit(f"unknown bench {bench!r}; choose from {BENCHES}")
        data["meta"] = _meta(settings) | {
            "experiment": args.name,
            "bench": bench,
            "split": args.split,
            "overrides": overrides,
            "abstain_threshold": settings.abstain_threshold,
            "wall_s": round(time.perf_counter() - start, 1),
        }
        data["summary"] = summary
        path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        print(path, json.dumps(summary, indent=1, default=str))


# ---- abstention threshold sweep ------------------------------------------------------------


def sweep_thresholds(rows: list[dict], correct: dict[str, bool], answerable: dict[str, bool]):
    """Replay a model-only run (no threshold) at each candidate threshold: below it the
    gate would have declined before generation, so the outcome is known without a rerun."""
    scores = sorted({r["top_score"] for r in rows if r["top_score"] is not None})
    out = []
    for t in [float("-inf"), *scores]:
        kept = [r for r in rows if r["top_score"] is not None and r["top_score"] >= t]
        kept_ids = {r["id"] for r in kept}
        ans = [r for r in rows if answerable[r["id"]]]
        una = [r for r in rows if not answerable[r["id"]]]
        out.append(
            {
                "threshold": t,
                "accuracy": mean([float(r["id"] in kept_ids and correct[r["id"]]) for r in ans]),
                "answered_rate": mean(
                    [float(r["id"] in kept_ids and not r["abstained"]) for r in ans]
                ),
                "oos_declined": mean(
                    [float(r["id"] not in kept_ids or r["abstained"]) for r in una]
                )
                if una
                else float("nan"),
            }
        )
    return out


def cmd_sweep(args: argparse.Namespace) -> None:
    path = result_path(args.name, args.bench, args.split)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["system"] == "full"]
    if args.bench == "pmc":
        qs = {q.id: q for q in load_questions()}
        answerable = {r["id"]: qs[r["id"]].answerable for r in rows}
        correct = {
            r["id"]: not r["abstained"] and keywords_match(r["answer"], qs[r["id"]].answer_keywords)
            for r in rows
        }
    else:
        from healthcare_rag import pubmedqa

        items = {i.pmid: i for i in pubmedqa.load_items()}
        answerable = {r["id"]: True for r in rows}
        correct = {r["id"]: r["pred"] == items[r["id"]].label for r in rows}
    sweep = sweep_thresholds(rows, correct, answerable)
    for s in sweep:
        print(
            f"  t={s['threshold']:7.2f} acc={100 * s['accuracy']:.1f}% "
            f"answered={100 * s['answered_rate']:.1f}% oos_declined={100 * s['oos_declined']:.1f}%"
        )
    out = path.with_name(path.stem + "__sweep.json")
    out.write_text(json.dumps(sweep, indent=1), encoding="utf-8")


# ---- reranker scores, fusion and latency ---------------------------------------------------


def _queries(split: str) -> list[dict[str, Any]]:
    """(id, bench, question, relevant chunk ids) for answerable PMC questions and PubMedQA."""
    out = [
        {"id": q.id, "bench": "pmc", "question": q.question, "relevant": q.relevant_chunks}
        for q in load_questions()
        if q.answerable and in_split(split, q.split)
    ]
    from healthcare_rag import pubmedqa

    data_dir = pubmedqa.DATA_DIR
    if (data_dir / "ori_pqal.json").exists():
        out += [
            {"id": i.pmid, "bench": "pubmedqa", "question": i.question, "relevant": [i.chunk_id]}
            for i in pubmedqa.load_items(data_dir)
            if in_split(split, i.split)
        ]
    return out


def cmd_rerank_scores(args: argparse.Namespace) -> None:
    from healthcare_rag.factory import build_retriever
    from healthcare_rag.pubmedqa import pubmedqa_settings
    from healthcare_rag.rerank import CrossEncoderReranker

    base = make_settings({})
    retrievers = {
        "pmc": build_retriever(base, "hybrid"),
        "pubmedqa": build_retriever(pubmedqa_settings(base), "hybrid"),
    }
    queries = _queries(args.split)
    for q in queries:
        hits = retrievers[q["bench"]].search(q["question"], k=DEPTH)
        q["candidates"] = [h.id for h in hits]
        q["texts"] = [f"{h.title}. {h.text}" for h in hits]
    scores: dict[str, list[list[float]]] = {}
    for model in args.models.split(","):
        rr = CrossEncoderReranker(model, device=base.reranker_device)
        scores[model] = [rr.scores(q["question"], q["texts"]) for q in queries]
        del rr
        _free_gpu()
        print(f"  scored {len(queries)} queries with {model}", flush=True)
    for q in queries:
        del q["texts"]
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    out = EXPERIMENTS / f"rerank_scores__{args.split}.json"
    out.write_text(json.dumps({"queries": queries, "scores": scores}), encoding="utf-8")
    print(out)


def _free_gpu() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


class StaticReranker:
    """Replays precomputed scores, so fusion variants need no model."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        scored = [replace(h, score=self._scores[h.id]) for h in hits]
        return sorted(scored, key=lambda h: h.score, reverse=True)


def fusion_table(data: dict, variants: dict[str, dict]) -> dict[str, dict[str, float]]:
    from healthcare_rag.rerank import EnsembleReranker

    out: dict[str, dict[str, float]] = {}
    for name, v in variants.items():
        per_bench: dict[str, list[float]] = {}
        for qi, q in enumerate(data["queries"]):
            hits = [Hit(cid, "", "", 0.0) for cid in q["candidates"]]
            members = [
                StaticReranker(dict(zip(q["candidates"], data["scores"][m][qi], strict=True)))
                for m in v["models"]
            ]
            if not members:
                ranked = [h.id for h in hits]
            elif len(members) == 1 and not v.get("retrieval_weight"):
                ranked = [h.id for h in members[0].rerank(q["question"], hits)]
            else:
                ens = EnsembleReranker(
                    members, weights=v.get("weights"), retrieval_weight=v.get("retrieval_weight", 0)
                )
                ranked = [h.id for h in ens.rerank(q["question"], hits)]
            per_bench.setdefault(q["bench"], []).append(ndcg_at_k(ranked, q["relevant"], 10))
        out[name] = {f"ndcg@10_{b}": mean(v) for b, v in per_bench.items()}
        out[name] |= {f"n_{b}": len(v) for b, v in per_bench.items()}
    return out


def default_variants(models: list[str]) -> dict[str, dict]:
    variants: dict[str, dict] = {"hybrid (no rerank)": {"models": []}}
    for m in models:
        variants[m] = {"models": [m]}
        for w in (0.5, 1.0):
            variants[f"{m} + retrieval w={w}"] = {"models": [m], "retrieval_weight": w}
    if len(models) > 1:
        variants["rrf(all)"] = {"models": models}
        for w in (0.5, 1.0):
            variants[f"rrf(all) + retrieval w={w}"] = {"models": models, "retrieval_weight": w}
        variants["rrf(all), first model x2"] = {
            "models": models,
            "weights": [2.0] + [1.0] * (len(models) - 1),
        }
    return variants


def cmd_fusion(args: argparse.Namespace) -> None:
    path = EXPERIMENTS / f"rerank_scores__{args.split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    table = fusion_table(data, default_variants(list(data["scores"])))
    for name, m in table.items():
        print(f"  {name:55s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if "ndcg" in k))
    out = EXPERIMENTS / f"fusion__{args.split}.json"
    out.write_text(json.dumps(table, indent=1), encoding="utf-8")


def cmd_rerank_bench(args: argparse.Namespace) -> None:
    from healthcare_rag.factory import build_retriever
    from healthcare_rag.rag_eval import power_state
    from healthcare_rag.rerank import CrossEncoderReranker

    settings = make_settings({})
    retriever = build_retriever(settings, "hybrid")
    questions = [q.question for q in load_questions()]
    cands = [retriever.search(q, k=DEPTH) for q in questions]
    rr = CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)
    sizes = [int(b) for b in args.batch_sizes.split(",")]
    for b in sizes:  # warm-up per batch shape
        rr.batch_size = b
        for q, c in zip(questions[:5], cands[:5], strict=True):
            rr.rerank(q, c)
    times: dict[int, list[float]] = {b: [] for b in sizes}
    for _ in range(args.rounds):
        for q, c in zip(questions, cands, strict=True):
            for b in sizes:  # interleaved, so drift affects every size alike
                rr.batch_size = b
                start = time.perf_counter()
                rr.rerank(q, c)
                times[b].append(time.perf_counter() - start)
    result = {
        "power": power_state(),
        "model": settings.reranker_model,
        "device": settings.reranker_device,
        "candidates": DEPTH,
        "n": {b: len(t) for b, t in times.items()},
        "p50_ms": {b: 1000 * percentile(t, 50) for b, t in times.items()},
        "p95_ms": {b: 1000 * percentile(t, 95) for b, t in times.items()},
        "mean_ms": {b: 1000 * mean(t) for b, t in times.items()},
    }
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS / "rerank_bench.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


def cmd_list(args: argparse.Namespace) -> None:
    for path in sorted(EXPERIMENTS.glob("*__*__*.json")):
        if path.stem.endswith("__sweep"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        for system, m in data.get("summary", {}).items():
            acc = m.get("answer_accuracy", m.get("accuracy"))
            ans = m.get("answered_rate")
            print(
                f"{path.stem:45s} {system:13s} acc={100 * acc:.1f}% answered={100 * ans:.1f}% "
                f"p50={1000 * m['p50_s']:.0f}ms p95={1000 * m['p95_s']:.0f}ms "
                f"power={meta.get('power', '?')} {meta.get('overrides', {})}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("answers")
    p.add_argument("name")
    p.add_argument("--bench", default="pmc")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--systems", default="full")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(fn=cmd_answers)
    p = sub.add_parser("sweep")
    p.add_argument("name")
    p.add_argument("--bench", choices=BENCHES, default="pmc")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("rerank-scores")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--models", default="ncbi/MedCPT-Cross-Encoder,BAAI/bge-reranker-v2-m3")
    p.set_defaults(fn=cmd_rerank_scores)
    p = sub.add_parser("fusion")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.set_defaults(fn=cmd_fusion)
    p = sub.add_parser("rerank-bench")
    p.add_argument("--batch-sizes", default="16,32")
    p.add_argument("--rounds", type=int, default=3)
    p.set_defaults(fn=cmd_rerank_bench)
    p = sub.add_parser("list")
    p.set_defaults(fn=cmd_list)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
