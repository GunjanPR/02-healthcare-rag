import json
from pathlib import Path

import pytest

from healthcare_rag.rag_eval import (
    REFUSAL,
    Question,
    choose_threshold,
    grade_answers,
    load_questions,
    ndcg_at_k,
    paired_bootstrap,
    retrieval_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def q(id_: str = "q1", answerable: bool = True, split: str = "test") -> Question:
    return Question(
        id=id_,
        split=split,
        answerable=answerable,
        topic="asthma",
        question="?",
        relevant_chunks=["PMC1:3"] if answerable else [],
        relevant_docs=["PMC1"] if answerable else [],
        answer_keywords=[["300"]] if answerable else [],
    )


def test_question_set_is_consistent() -> None:
    questions = load_questions(ROOT / "eval/qa/questions.json")
    manifest = json.loads((ROOT / "eval/pmc_manifest.json").read_text(encoding="utf-8"))
    assert len({x.id for x in questions}) == len(questions)
    assert {x.split for x in questions} == {"dev", "test"}
    for x in questions:
        if x.answerable:
            assert x.relevant_docs[0] in manifest["articles"], x.id
            assert x.relevant_chunks[0].startswith(x.relevant_docs[0] + ":"), x.id
            assert x.answer_keywords and all(x.answer_keywords), x.id
        else:
            assert not x.relevant_chunks and not x.answer_keywords, x.id


def test_ndcg_single_relevant() -> None:
    assert ndcg_at_k(["PMC1:3", "x"], ["PMC1:3"]) == 1.0
    assert ndcg_at_k(["x", "PMC1:3"], ["PMC1:3"]) == pytest.approx(1 / 1.5849625)
    assert ndcg_at_k(["x"] * 10 + ["PMC1:3"], ["PMC1:3"]) == 0.0


def test_retrieval_metrics() -> None:
    m = retrieval_metrics(["PMC9:0", "PMC1:7", "PMC1:3"], q())
    assert m["recall@1"] == 0.0 and m["recall@5"] == 1.0
    assert m["mrr@10"] == pytest.approx(1 / 3)
    assert m["doc_hit@5"] == 1.0  # PMC1:7 is from the right article


def test_choose_threshold_separates_scores() -> None:
    best = choose_threshold(answerable=[5.0, 3.0, 1.0], unanswerable=[-4.0, -2.0, 2.0])
    assert -2.0 < best["threshold"] <= 1.0
    assert best["answerable_kept"] == 1.0
    assert best["unanswerable_refused"] == pytest.approx(2 / 3)


def test_paired_bootstrap_ci_contains_mean() -> None:
    a = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    diff, lo, hi = paired_bootstrap(a, b, n=500)
    assert diff == pytest.approx(2 / 6)
    assert lo <= diff <= hi


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ("INSUFFICIENT_EVIDENCE", True),
        ("The passages do not contain this information.", True),
        ("I don't know.", True),
        ("Metformin lowers glucose [1].", False),
    ],
)
def test_refusal_detection(text: str, refusal: bool) -> None:
    assert bool(REFUSAL.search(text)) is refusal


def test_grade_answers() -> None:
    questions = [q("q1"), q("q2"), q("q3", answerable=False)]

    def row(id_: str, answer: str, abstained: bool, cited: list[str]) -> dict:
        return {
            "id": id_,
            "system": "full",
            "answer": answer,
            "abstained": abstained,
            "cited": cited,
            "faithful": True,
            "latency_s": 1.0,
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cost_usd": 0.0,
        }

    data = {
        "rows": [
            row("q1", "The cutoff was 300 cells [1].", False, ["PMC1:3"]),
            row("q2", "x", True, []),
            row("q3", "x", True, []),
        ]
    }
    m = grade_answers(data, questions, "test")["full"]
    assert m["answer_accuracy"] == 0.5
    assert m["answered_rate"] == 0.5
    assert m["accuracy_when_answered"] == 1.0
    assert m["citation_accuracy_chunk"] == 1.0
    assert m["unanswerable_abstained"] == 1.0
    assert m["abstention_precision"] == 0.5  # one of two declines was out of scope
