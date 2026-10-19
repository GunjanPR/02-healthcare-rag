"""PubMedQA (PQA-L) benchmark: 1,000 expert-annotated yes/no/maybe research questions.

Usage (needs OpenSearch + Qdrant + the gateway; `full` needs the [rerank] extra):
    python -m healthcare_rag.pubmedqa download        # data/pubmedqa, sha256-checked
    python -m healthcare_rag.pubmedqa index           # OpenSearch index + Qdrant collection
    python -m healthcare_rag.pubmedqa run --split test
    python -m healthcare_rag.pubmedqa report          # docs/pubmedqa_results.md

PQA-L (Jin et al., 2019, MIT license) pairs each question with the abstract it was
written from. The abstract's CONTEXTS exclude its conclusion, which holds the answer, so
the label must be reasoned from the results. The official split puts the 500 ids in
test_ground_truth.json in test and the other 500 in dev.

Each abstract's joined CONTEXTS is indexed as one chunk in its own OpenSearch index and
Qdrant collection (`pubmedqa`), apart from the PMC corpus. Systems:

- oracle_context: the gold abstract only, one LLM call (the published
  "reasoning-required" setting).
- m1_baseline: BM25 top 3 over the 1,000 abstracts, one LLM call.
- full: hybrid retrieval -> MedCPT rerank -> abstention gate -> generate -> verify, the
  same pipeline and threshold as on PMC.

Every system appends ANSWER_INSTRUCTION to the question in the LLM prompt (retrieval
sees the plain question). The first yes/no/maybe in the reply is the prediction; an
abstention or a reply without a label counts as wrong.
"""

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from healthcare_rag.baseline import build_messages
from healthcare_rag.config import Settings, get_settings
from healthcare_rag.evaluation import percentile
from healthcare_rag.rag import INSUFFICIENT
from healthcare_rag.rag_eval import REFUSAL, RESULTS, _meta, mean, paired_bootstrap
from healthcare_rag.types import Chunk, Hit

BASE_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"
FILES = {
    "ori_pqal.json": "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d",
    "test_ground_truth.json": "939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51",
}
DATA_DIR = Path("data/pubmedqa")
INDEX = "pubmedqa"  # OpenSearch index and Qdrant collection
LABELS = ("yes", "no", "maybe")
ANSWER_INSTRUCTION = "Start your answer with Yes, No or Maybe."
SYSTEMS = ("oracle_context", "m1_baseline", "full")
REPORT = Path("docs/pubmedqa_results.md")

_LABEL = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Item:
    pmid: str
    split: str  # "dev" | "test"
    question: str
    context: str  # the abstract without its conclusion
    label: str  # yes | no | maybe
    year: int | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.pmid}:0"

    def to_chunk(self) -> Chunk:
        # No title: PQA-L questions are often the article's title, which would leak.
        return Chunk(
            chunk_id=self.chunk_id,
            doc_id=self.pmid,
            title="",
            text=self.context,
            section="abstract",
            year=self.year,
            topic="pubmedqa",
        )

    def to_hit(self) -> Hit:
        return Hit(self.chunk_id, "", self.context, 0.0, self.pmid, "abstract", self.year)


# ---- data ----------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dest: Path = DATA_DIR) -> None:
    """Fetch the two files once and refuse any whose checksum differs."""
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = dest / name
        if not path.exists():
            resp = httpx.get(BASE_URL + name, follow_redirects=True, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        if sha256(path) != digest:
            path.unlink()
            raise SystemExit(f"{name}: sha256 mismatch, file removed; re-run download")
        print(f"{path} ok")


def parse_items(pqal: dict[str, dict], test_ids: set[str]) -> list[Item]:
    items = []
    for pmid, rec in pqal.items():
        year = rec.get("YEAR")
        items.append(
            Item(
                pmid=pmid,
                split="test" if pmid in test_ids else "dev",
                question=rec["QUESTION"],
                context=" ".join(rec["CONTEXTS"]),
                label=rec["final_decision"],
                year=int(year) if year and str(year).isdigit() else None,
            )
        )
    return items


def load_items(data_dir: Path = DATA_DIR) -> list[Item]:
    pqal = json.loads((data_dir / "ori_pqal.json").read_text(encoding="utf-8"))
    test = json.loads((data_dir / "test_ground_truth.json").read_text(encoding="utf-8"))
    return parse_items(pqal, set(test))


def pubmedqa_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"opensearch_index": INDEX, "qdrant_collection": INDEX})


# ---- grading -------------------------------------------------------------------------------


def parse_label(text: str) -> str | None:
    """The first yes/no/maybe in the reply, or None for a refusal or a reply without one.

    A label at the very start always counts ("No, ..." is an answer). Later in the reply,
    a refusal ("the passages do not provide ... no information") wins over a stray word.
    """
    m = _LABEL.search(text)
    if m and not re.search(r"[A-Za-z0-9]", text[: m.start()]):
        return m.group(1).lower()
    if INSUFFICIENT in text or REFUSAL.search(text):
        return None
    return m.group(1).lower() if m else None


def macro_f1(gold: list[str], pred: list[str | None]) -> float:
    """Mean F1 over yes/no/maybe; a missing prediction is a miss for its gold class."""
    scores = []
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        n_pred = sum(p == c for p in pred)
        n_gold = sum(g == c for g in gold)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        scores.append(2 * precision * recall / (precision + recall) if tp else 0.0)
    return mean(scores)


def doc_recall(passages: list[str], pmid: str, k: int) -> float:
    return float(any(p.split(":")[0] == pmid for p in passages[:k]))


def summarize(rows: list[dict], items: dict[str, Item]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in dict.fromkeys(r["system"] for r in rows):
        rs = [r for r in rows if r["system"] == name]
        gold = [items[r["id"]].label for r in rs]
        pred = [r["pred"] for r in rs]
        lat = [r["latency_s"] for r in rs]
        out[name] = {
            "n": len(rs),
            "accuracy": mean([float(g == p) for g, p in zip(gold, pred, strict=True)]),
            "macro_f1": macro_f1(gold, pred),
            "answered_rate": mean([float(p is not None) for p in pred]),
            "low_score_abstained": mean([float(r["reason"] == "low_score") for r in rs]),
            "pred_counts": {c: pred.count(c) for c in (*LABELS, None)},
            "doc_recall@1": mean([doc_recall(r["passages"], r["id"], 1) for r in rs]),
            "doc_recall@5": mean([doc_recall(r["passages"], r["id"], 5) for r in rs]),
            "n_passages": max((len(r["passages"]) for r in rs), default=0),
            "p50_s": percentile(lat, 50),
            "p95_s": percentile(lat, 95),
            "mean_llm_calls": mean([r.get("llm_calls", 1) for r in rs]),
            "mean_prompt_tokens": mean([r["prompt_tokens"] for r in rs]),
        }
    return out


def per_item_correct(rows: list[dict], items: dict[str, Item], system: str) -> dict[str, float]:
    return {
        r["id"]: float(r["pred"] == items[r["id"]].label) for r in rows if r["system"] == system
    }


# ---- runs ----------------------------------------------------------------------------------


def index_items(settings: Settings, items: list[Item]) -> dict:
    from healthcare_rag.indexing import index_all

    return index_all(pubmedqa_settings(settings), [i.to_chunk() for i in items])


def build_systems(settings: Settings, names: tuple[str, ...]) -> dict[str, Callable]:
    """name -> run(item) returning the rag_eval answer dict for that item."""
    from healthcare_rag.gateway import GatewayClient
    from healthcare_rag.rag_eval import build_answer_systems

    systems: dict[str, Callable] = {}
    if "oracle_context" in names:
        chat = GatewayClient.from_settings(settings, bypass_cache=True)

        def oracle(item: Item) -> dict[str, Any]:
            start = time.perf_counter()
            r = chat.chat(build_messages(item.question, [item.to_hit()], ANSWER_INSTRUCTION))
            return {
                "answer": r.text,
                "abstained": False,
                "reason": "answered",
                "passages": [item.chunk_id],
                "top_score": None,
                "latency_s": time.perf_counter() - start,
                "timings": {"generate": r.latency_s},
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "llm_calls": 1,
            }

        systems["oracle_context"] = oracle
    rag_names = [n for n in names if n != "oracle_context"]
    if rag_names:
        pq = pubmedqa_settings(settings)
        built = build_answer_systems(pq, pq.abstain_threshold)
        for n in rag_names:
            run = built[n]
            systems[n] = lambda item, run=run: run(item.question, ANSWER_INSTRUCTION)
    return systems


def _row(item: Item, system: str, r: dict[str, Any]) -> dict[str, Any]:
    pred = None if r["abstained"] else parse_label(r["answer"])
    return {
        "id": item.pmid,
        "system": system,
        "pred": pred,
        "answer": r["answer"],
        "abstained": r["abstained"],
        "reason": r["reason"],
        "passages": r["passages"],
        "top_score": r.get("top_score"),
        "latency_s": r["latency_s"],
        "timings": r.get("timings", {}),
        "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
        "llm_calls": r.get("llm_calls", 1),
    }


def run(settings: Settings, items: list[Item], names: tuple[str, ...], out: Path) -> dict[str, Any]:
    """Run each system over the items, saving as it goes; rows already in `out` (same
    system and id) are kept, so an interrupted run resumes."""
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"rows": []}
    done = {(r["system"], r["id"]) for r in data["rows"]}
    data["meta"] = _meta(settings) | {
        "pubmedqa_index": INDEX,
        "abstain_threshold": settings.abstain_threshold,
        "prompt_version": settings.prompt_version,
        "citation_repair": settings.citation_repair,
    }
    systems = build_systems(settings, names)
    warm = items[0]
    for fn in systems.values():
        fn(warm)
    for name, fn in systems.items():
        for i, item in enumerate(items, start=1):
            if (name, item.pmid) in done:
                continue
            data["rows"].append(_row(item, name, fn(item)))
            if i % 25 == 0 or i == len(items):
                out.write_text(json.dumps(data, indent=1), encoding="utf-8")
                print(f"  {name} {i}/{len(items)}", flush=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


# ---- report --------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(data: dict, items: dict[str, Item], split: str, path: Path = REPORT) -> str:
    rows = [r for r in data["rows"] if items[r["id"]].split == split]
    s = summarize(rows, items)
    gold = [i.label for i in items.values() if i.split == split]
    meta = data["meta"]
    lines = [
        f"# PubMedQA (PQA-L) results, {split} split",
        "",
        f"Generated by `make pubmedqa` on {meta['date']} (power: {meta.get('power', '?')}). "
        "All numbers are measured.",
        "",
        f"- **Questions:** {len(gold)} expert-annotated questions "
        f"(yes {gold.count('yes')}, no {gold.count('no')}, maybe {gold.count('maybe')}); "
        "the official test split.",
        "- **Index:** the 1,000 PQA-L abstracts without their conclusions, one chunk each, "
        f"in a separate OpenSearch index and Qdrant collection (`{INDEX}`).",
        f'- **Prompt:** every system appends "{ANSWER_INSTRUCTION}" to the question; the '
        "first yes/no/maybe in the reply is the prediction, and an abstention is wrong.",
        f"- **Models:** `{meta['chat_model']}` via the gateway (temperature 0); reranker "
        f"`{meta['reranker_model']}`; abstention threshold {meta.get('abstain_threshold', '')}"
        " (unchanged from PMC).",
        "",
        "| Metric | " + " | ".join(s) + " |",
        "|---|" + "---|" * len(s),
    ]
    table = [
        ("Accuracy", "accuracy", _pct),
        ("Macro-F1", "macro_f1", lambda x: f"{x:.3f}"),
        ("Answered (a label was given)", "answered_rate", _pct),
        ("Declined by the reranker threshold", "low_score_abstained", _pct),
        ("Gold abstract in passage 1", "doc_recall@1", _pct),
        ("Gold abstract in the passages shown", "doc_recall@5", _pct),
        ("Passages shown", "n_passages", str),
        ("Latency p50", "p50_s", lambda x: f"{1000 * x:.0f} ms"),
        ("Latency p95", "p95_s", lambda x: f"{1000 * x:.0f} ms"),
        ("LLM calls per question", "mean_llm_calls", lambda x: f"{x:.2f}"),
    ]
    for label, key, fmt in table:
        lines.append(f"| {label} | " + " | ".join(fmt(m[key]) for m in s.values()) + " |")
    lines += ["", "Predicted labels (yes / no / maybe / none):", ""]
    for name, m in s.items():
        c = m["pred_counts"]
        lines.append(f"- {name}: {c['yes']} / {c['no']} / {c['maybe']} / {c[None]}")
    if {"m1_baseline", "full"} <= set(s):
        a = per_item_correct(rows, items, "m1_baseline")
        b = per_item_correct(rows, items, "full")
        ids = sorted(set(a) & set(b))
        d = paired_bootstrap([a[i] for i in ids], [b[i] for i in ids])
        lines += [
            "",
            f"Accuracy, full minus m1_baseline: {100 * d[0]:+.1f} points (95% paired "
            f"bootstrap CI {100 * d[1]:+.1f} to {100 * d[2]:+.1f}).",
        ]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


# ---- CLI -----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=["download", "index", "run", "report", "all"])
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--out", type=Path, default=None, help="results JSON path")
    args = parser.parse_args(argv)
    settings = get_settings()
    out = args.out or RESULTS / f"pubmedqa_{args.split}.json"

    if args.cmd in ("download", "all"):
        download()
    if args.cmd == "download":
        return
    items = load_items()
    by_id = {i.pmid: i for i in items}
    if args.cmd in ("index", "all"):
        print(json.dumps(index_items(settings, items), indent=1))
    if args.cmd in ("run", "all"):
        chosen = [i for i in items if i.split == args.split]
        chosen = chosen[: args.limit] if args.limit else chosen
        names = tuple(n.strip() for n in args.systems.split(",") if n.strip())
        unknown = set(names) - set(SYSTEMS)
        if unknown:
            raise SystemExit(f"unknown systems {sorted(unknown)}; choose from {SYSTEMS}")
        out.parent.mkdir(parents=True, exist_ok=True)
        data = run(settings, chosen, names, out)
        print(json.dumps(summarize(data["rows"], by_id), indent=1, default=str))
    if args.cmd in ("report", "all"):
        data = json.loads(out.read_text(encoding="utf-8"))
        print(write_report(data, by_id, args.split))


if __name__ == "__main__":
    main()
