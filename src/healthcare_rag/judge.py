"""Second-pass LLM judging with a model from another family than the generator.

Runs after generation, as its own pass, so loading the judge never evicts the generator
mid-run. Two jobs:

    # answer correctness: question + source passage + reference keyword groups
    python -m healthcare_rag.judge correctness eval/results/answers.json --split test
    # faithfulness scorers vs the 20 manual labels in eval/qa/judge_audit.json
    python -m healthcare_rag.judge audit
    # faithfulness rate of answered rows with one scorer
    python -m healthcare_rag.judge faithfulness eval/results/answers.json --scorer claims
    # a scorer's verdicts vs the 88 blind hand labels of the full system's test answers
    python -m healthcare_rag.judge labels eval/results/answers__faithfulness_claims_test.json

Keyword grading misses right answers phrased in other words; the correctness judge is a
multi-reference check of that. Agreement is reported with Cohen's kappa and every
disagreement is listed, so the two can be read side by side.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from healthcare_rag.citations import LexicalScorer, SupportScorer, check
from healthcare_rag.config import Settings, get_settings
from healthcare_rag.evaluation import keywords_match
from healthcare_rag.rag import number_passages
from healthcare_rag.rag_eval import JUDGE_PROMPT, RESULTS, load_questions, mean
from healthcare_rag.types import Hit

JUDGE_MODEL = "qwen2.5:7b-instruct"
AUDIT = Path("eval/qa/judge_audit.json")
LABELS = Path("eval/qa/faithfulness_labels_test.json")

# Claim-by-claim judge: the model lists the answer's claims and quotes support for each
# before it decides. The one-word judge (JUDGE_PROMPT) has to decide in its first token.
CLAIMS_PROMPT = (
    "You check whether an answer to a biomedical question is supported by numbered source "
    "passages.\n"
    "1. List every factual claim the answer makes, one per line. Ignore passage numbers "
    "such as [2] and phrases such as 'according to the study'.\n"
    "2. Under each claim, copy the words from the passages that state it, or write "
    "NOT FOUND. Paraphrase, abbreviations and rounding are fine. A claim that gives a "
    "number, group, outcome or direction different from the passages is NOT FOUND, and so "
    "is a claim that only outside knowledge supports.\n"
    "3. End with exactly one line: 'VERDICT: YES' if every claim was found, otherwise "
    "'VERDICT: NO'."
)

CORRECTNESS_PROMPT = (
    "You grade answers to biomedical questions. You get the question, the source passage "
    "the question was written from, and the reference answer as groups of terms (every "
    "group must be expressed; any alternative in a group, a synonym, an equivalent number "
    "format or a paraphrase counts). Reply with exactly one word: CORRECT if the answer "
    "states the reference answer, otherwise INCORRECT. An answer that declines, or that "
    "answers a different question, is INCORRECT."
)


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n (sound at small n and rates near 1)."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def agreement(reference: list[bool], other: list[bool]) -> dict[str, Any]:
    pos = [o for r, o in zip(reference, other, strict=True) if r]
    neg = [o for r, o in zip(reference, other, strict=True) if not r]
    return {
        "n": len(reference),
        "agreement": mean([float(r == o) for r, o in zip(reference, other, strict=True)]),
        "kappa": cohen_kappa(reference, other),
        "reference_true_matched": f"{sum(pos)}/{len(pos)}",
        "reference_false_matched": f"{sum(not o for o in neg)}/{len(neg)}",
    }


def fetch_chunks(settings: Settings, ids: list[str]) -> dict[str, Hit]:
    from healthcare_rag.bm25 import _hit, client_from_settings

    client = client_from_settings(settings)
    out: dict[str, Hit] = {}
    unique = list(dict.fromkeys(ids))
    for i in range(0, len(unique), 500):
        resp = client.mget(index=settings.opensearch_index, body={"ids": unique[i : i + 500]})
        out |= {d["_id"]: _hit(d["_source"], 0.0) for d in resp["docs"] if d.get("found")}
    return out


def judge_client(settings: Settings, model: str):
    from healthcare_rag.gateway import GatewayClient

    judge_settings = settings.model_copy(update={"chat_model": model})
    return GatewayClient.from_settings(judge_settings, bypass_cache=True)


def ask_correct(chat, question: str, passage: str, groups: list[list[str]], answer: str) -> bool:
    reference = "; ".join(" OR ".join(g) for g in groups)
    messages = [
        {"role": "system", "content": CORRECTNESS_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nSource passage:\n{passage}\n\n"
            f"Reference answer terms: {reference}\n\nAnswer to grade:\n{answer}",
        },
    ]
    reply = chat.chat(messages, max_tokens=4).text.strip().upper()
    return reply.startswith("CORRECT")


def ask_faithful(chat, passages: str, answer: str) -> bool:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Passages:\n{passages}\n\nAnswer:\n{answer}"},
    ]
    return chat.chat(messages, max_tokens=3).text.strip().upper().startswith("YES")


# The one-word judge, but it also sees the question, so a short answer such as "82.9% [2]"
# can be read as the claim it makes ("82.9% of patients ...") instead of a bare number.
QUESTION_JUDGE_PROMPT = (
    "You check whether an answer to a question is supported by source passages. Read the "
    "answer as a reply to the question. Reply with exactly one word: YES if every factual "
    "claim in the answer is stated in or directly follows from the passages, otherwise NO."
)


def ask_faithful_with_question(chat, question: str, passages: str, answer: str) -> bool:
    messages = [
        {"role": "system", "content": QUESTION_JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nPassages:\n{passages}\n\nAnswer:\n{answer}",
        },
    ]
    return chat.chat(messages, max_tokens=3).text.strip().upper().startswith("YES")


def parse_verdict(text: str) -> bool:
    """The last VERDICT line decides; a reply without one counts as NO."""
    found = [line for line in text.upper().splitlines() if "VERDICT" in line]
    return bool(found) and "YES" in found[-1].split("VERDICT", 1)[1]


def ask_faithful_claims(chat, passages: str, answer: str) -> tuple[bool, str]:
    messages = [
        {"role": "system", "content": CLAIMS_PROMPT},
        {"role": "user", "content": f"Passages:\n{passages}\n\nAnswer:\n{answer}"},
    ]
    text = chat.chat(messages, max_tokens=700).text
    return parse_verdict(text), text


def scorer_faithful(answer: str, passages: list[Hit], scorer: SupportScorer, thr: float) -> bool:
    """Every sentence is supported by a passage it cites (citations.check)."""
    texts = [f"{h.title}. {h.text}" for h in passages]
    return check(answer, texts, scorer, thr).all_supported


# ---- commands ------------------------------------------------------------------------------


def cmd_correctness(args: argparse.Namespace, settings: Settings) -> None:
    questions = {q.id: q for q in load_questions()}
    data = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    systems = set(args.systems.split(",")) if args.systems else None
    rows = [
        r
        for r in data["rows"]
        if questions[r["id"]].answerable
        and (args.split == "all" or questions[r["id"]].split == args.split)
        and (systems is None or r["system"] in systems)
    ]
    source = fetch_chunks(settings, [questions[r["id"]].relevant_chunks[0] for r in rows])
    chat = judge_client(settings, args.model)
    out = []
    for r in rows:
        q = questions[r["id"]]
        chunk = source[q.relevant_chunks[0]]
        kw = not r["abstained"] and keywords_match(r["answer"], q.answer_keywords)
        llm = not r["abstained"] and ask_correct(
            chat, q.question, f"{chunk.title}. {chunk.text}", q.answer_keywords, r["answer"]
        )
        out.append({"id": r["id"], "system": r["system"], "keyword": kw, "judge": llm})
        print(f"  {r['system']} {r['id']} keyword={kw} judge={llm}", flush=True)
    summary = {}
    for name in dict.fromkeys(o["system"] for o in out):
        sub = [o for o in out if o["system"] == name]
        kw, jd = [o["keyword"] for o in sub], [o["judge"] for o in sub]
        summary[name] = {
            "keyword_accuracy": mean([float(x) for x in kw]),
            "judge_accuracy": mean([float(x) for x in jd]),
            **agreement(kw, jd),
        }
    rows_by_key = {(r["system"], r["id"]): r for r in rows}
    disagreements = [
        o
        | {
            "question": questions[o["id"]].question,
            "keywords": questions[o["id"]].answer_keywords,
            "answer": rows_by_key[(o["system"], o["id"])]["answer"],
        }
        for o in out
        if o["keyword"] != o["judge"]
    ]
    result = {
        "judge_model": args.model,
        "answers": str(args.answers),
        "split": args.split,
        "summary": summary,
        "rows": out,
        "disagreements": disagreements,
    }
    path = Path(args.out or RESULTS / f"{Path(args.answers).stem}__correctness_{args.split}.json")
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"{len(disagreements)} disagreements -> {path}")


def _nli(settings: Settings, model: str):
    from healthcare_rag.nli import NLIScorer

    return NLIScorer(model, device=settings.reranker_device)


def cmd_audit(args: argparse.Namespace, settings: Settings) -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))["items"]
    answers = json.loads((RESULTS / "answers_v1.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in answers["rows"] if r["system"] == "full"}
    hits = fetch_chunks(settings, [p for i in audit for p in rows[i["id"]]["passages"]])
    manual = [bool(i["manual"]) for i in audit]
    verdicts: dict[str, list[bool]] = {"llama3.2 judge (original)": [i["judge"] for i in audit]}
    lexical = LexicalScorer()
    nli = _nli(settings, args.nli_model)
    chat = judge_client(settings, args.model)
    verdicts[f"lexical (support >= {args.lexical_threshold})"] = []
    verdicts[f"nli {args.nli_model} (p >= {args.nli_threshold})"] = []
    verdicts[f"{args.model} judge"] = []
    for item in audit:
        r = rows[item["id"]]
        passages = [hits[p] for p in r["passages"]]
        lists = list(verdicts.values())
        lists[1].append(scorer_faithful(r["answer"], passages, lexical, args.lexical_threshold))
        lists[2].append(scorer_faithful(r["answer"], passages, nli, args.nli_threshold))
        lists[3].append(ask_faithful(chat, number_passages(passages), r["answer"]))
    table = {name: agreement(manual, v) for name, v in verdicts.items()}
    result = {
        "manual_true": sum(manual),
        "n": len(manual),
        "scorers": table,
        "verdicts": {name: v for name, v in verdicts.items()},
        "ids": [i["id"] for i in audit],
    }
    (RESULTS / "faithfulness_audit.json").write_text(json.dumps(result, indent=1), "utf-8")
    print(json.dumps(table, indent=1))


def cmd_faithfulness(args: argparse.Namespace, settings: Settings) -> None:
    questions = {q.id: q for q in load_questions()}
    data = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    rows = [
        r
        for r in data["rows"]
        if not r["abstained"]
        and (args.split == "all" or questions[r["id"]].split == args.split)
        and (not args.systems or r["system"] in args.systems.split(","))
    ]
    hits = fetch_chunks(settings, [p for r in rows for p in r["passages"]])
    reasons: dict[tuple[str, str], str] = {}
    if args.scorer == "judge":
        chat = judge_client(settings, args.model)

        def faithful(r: dict) -> bool:
            passages = number_passages([hits[p] for p in r["passages"]])
            return ask_faithful(chat, passages, r["answer"])
    elif args.scorer == "judge_q":
        chat = judge_client(settings, args.model)

        def faithful(r: dict) -> bool:
            passages = number_passages([hits[p] for p in r["passages"]])
            return ask_faithful_with_question(
                chat, questions[r["id"]].question, passages, r["answer"]
            )
    elif args.scorer == "claims":
        chat = judge_client(settings, args.model)

        def faithful(r: dict) -> bool:
            passages = number_passages([hits[p] for p in r["passages"]])
            verdict, text = ask_faithful_claims(chat, passages, r["answer"])
            reasons[(r["system"], r["id"])] = text
            print(f"  {r['system']} {r['id']} faithful={verdict}", flush=True)
            return verdict
    else:
        scorer = LexicalScorer() if args.scorer == "lexical" else _nli(settings, args.nli_model)
        thr = args.lexical_threshold if args.scorer == "lexical" else args.nli_threshold

        def faithful(r: dict) -> bool:
            return scorer_faithful(r["answer"], [hits[p] for p in r["passages"]], scorer, thr)

    verdicts = [{"id": r["id"], "system": r["system"], "faithful": faithful(r)} for r in rows]
    for v in verdicts:
        if (v["system"], v["id"]) in reasons:
            v["reasoning"] = reasons[(v["system"], v["id"])]
    summary = {}
    for name in dict.fromkeys(v["system"] for v in verdicts):
        sub = [v["faithful"] for v in verdicts if v["system"] == name]
        summary[name] = {
            "n_answered": len(sub),
            "faithful_rate": mean([float(x) for x in sub]),
            "ci95": list(wilson_interval(sum(sub), len(sub))),
        }
    print(json.dumps({"scorer": args.scorer, "split": args.split, "summary": summary}, indent=1))
    path = Path(
        args.out
        or RESULTS / f"{Path(args.answers).stem}__faithfulness_{args.scorer}_{args.split}.json"
    )
    result = {"scorer": args.scorer, "split": args.split, "summary": summary, "rows": verdicts}
    path.write_text(json.dumps(result, indent=1), "utf-8")


def cmd_labels(args: argparse.Namespace, settings: Settings) -> None:
    """Agreement of a faithfulness run (`faithfulness --out ...` JSON) with the hand labels.

    Only rows whose answer text is the one that was labeled are compared."""
    import hashlib

    labels = json.loads(Path(getattr(args, "labels", "") or LABELS).read_text(encoding="utf-8"))
    run = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    answers = {
        r["id"]: r["answer"]
        for r in json.loads(Path(labels["answers"]).read_text(encoding="utf-8"))["rows"]
        if r["system"] == labels["system"]
    }
    verdicts = {v["id"]: v["faithful"] for v in run["rows"] if v["system"] == labels["system"]}
    manual, judged, ids = [], [], []
    for item in labels["items"]:
        sha = hashlib.sha1(answers[item["id"]].encode()).hexdigest()[:12]
        if item["id"] in verdicts and sha == item["answer_sha1"]:
            manual.append(item["faithful"])
            judged.append(verdicts[item["id"]])
            ids.append(item["id"])
    table = agreement(manual, judged)
    table["manual_rate"] = mean([float(x) for x in manual])
    table["judge_rate"] = mean([float(x) for x in judged])
    table["disagreements"] = [
        {"id": i, "manual": m, "judge": j}
        for i, m, j in zip(ids, manual, judged, strict=True)
        if m != j
    ]
    print(json.dumps({"run": args.answers, "scorer": run.get("scorer"), **table}, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=["correctness", "audit", "faithfulness", "labels"])
    parser.add_argument("answers", nargs="?", default=str(RESULTS / "answers.json"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--systems", default="")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument(
        "--scorer", choices=["lexical", "nli", "judge", "judge_q", "claims"], default="judge"
    )
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--nli-threshold", type=float, default=0.5)
    parser.add_argument("--lexical-threshold", type=float, default=0.6)
    parser.add_argument("--out", default="")
    parser.add_argument("--labels", default="", help=f"hand labels for `labels` (default {LABELS})")
    args = parser.parse_args(argv)
    settings = get_settings()
    commands = {
        "correctness": cmd_correctness,
        "audit": cmd_audit,
        "faithfulness": cmd_faithfulness,
        "labels": cmd_labels,
    }
    commands[args.cmd](args, settings)


if __name__ == "__main__":
    main()
