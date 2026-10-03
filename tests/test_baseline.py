import json
from pathlib import Path

import httpx
import pytest

from healthcare_rag.baseline import SYSTEM_PROMPT, BM25Baseline, build_messages
from healthcare_rag.bm25 import Document, Hit
from healthcare_rag.evaluation import (
    Question,
    evaluate,
    keywords_match,
    load_corpus,
    load_questions,
    percentile,
    recall_at_k,
    reciprocal_rank,
    to_markdown,
)
from healthcare_rag.gateway import ChatResult, GatewayClient, GatewayError

TOY = Path(__file__).resolve().parents[1] / "eval" / "toy"


class FakeRetriever:
    """Returns documents whose title shares a word with the query, in corpus order."""

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def search(self, query: str, k: int = 5) -> list[Hit]:
        words = set(query.lower().split())
        matches = [d for d in self.docs if words & set(d.title.lower().split())]
        return [Hit(d.id, d.title, d.text, 1.0) for d in matches][:k]


class FakeChat:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **_: object) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(self.reply, 100, 10, 0.0, 0.5, "ollama", "BYPASS")


DOCS = [
    Document("d1", "metformin diabetes", "Metformin is first-line for type 2 diabetes."),
    Document("d2", "asthma inhalers", "Inhaled corticosteroids control asthma."),
]


def test_toy_data_is_consistent() -> None:
    docs = load_corpus(TOY / "corpus.json")
    questions = load_questions(TOY / "questions.json")
    ids = {d.id for d in docs}
    assert len(ids) == len(docs)
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert set(q.relevant) <= ids, q.id
        assert q.answer_keywords and all(q.answer_keywords), q.id


def test_recall_and_rr() -> None:
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, ["b"], 1) == 0.0
    assert recall_at_k(ranked, ["b"], 3) == 1.0
    assert recall_at_k(ranked, ["a", "z"], 3) == 0.5
    assert reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["z"]) == 0.0


@pytest.mark.parametrize(
    ("answer", "groups", "expected"),
    [
        ("Use Metformin.", [["metformin"]], True),
        ("Inhaled corticosteroids [1].", [["ics", "inhaled corticosteroid"]], True),
        ("Give antibiotics.", [["ics"]], False),  # whole-term match only
        ("Use SSRIs such as sertraline.", [["ssri"]], True),
        ("FEV1/FVC below 0.70", [["fev1/fvc"], ["0.70", "0.7"]], True),
        ("FEV1/FVC is low", [["fev1/fvc"], ["0.70", "0.7"]], False),
        ("T-score of -2.5 or lower", [["-2.5"]], True),
    ],
)
def test_keywords_match(answer: str, groups: list[list[str]], expected: bool) -> None:
    assert keywords_match(answer, groups) is expected


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    assert percentile([3.0], 95) == 3.0


def test_build_messages_numbers_passages() -> None:
    msgs = build_messages("q?", [Hit("d1", "T1", "x", 1.0), Hit("d2", "T2", "y", 0.5)])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[1] T1\nx" in msgs[1]["content"]
    assert "[2] T2\ny" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: q?")


def test_baseline_passes_top_k_to_chat() -> None:
    chat = FakeChat("Metformin [1].")
    result = BM25Baseline(FakeRetriever(DOCS), chat, k=1).answer("metformin dose")
    assert [h.id for h in result.hits] == ["d1"]
    assert result.answer == "Metformin [1]."
    assert "metformin diabetes" in chat.calls[0][1]["content"]
    assert "asthma" not in chat.calls[0][1]["content"]


def test_evaluate_and_markdown() -> None:
    retriever = FakeRetriever(DOCS)
    pipeline = BM25Baseline(retriever, FakeChat("Metformin [1]."), k=1)
    questions = [
        Question("q1", "metformin use", ["d1"], [["metformin"]]),
        Question("q2", "asthma control", ["d2"], [["corticosteroid"]]),
        Question("q3", "unrelated words", ["d1"], [["metformin"]]),
    ]
    summary, rows = evaluate(questions, retriever, pipeline)
    assert summary["n"] == 3
    assert summary["recall@1"] == pytest.approx(2 / 3)
    assert summary["answer_accuracy"] == pytest.approx(2 / 3)  # q2 answer is wrong
    assert summary["total_cost_usd"] == 0.0
    assert [r["correct"] for r in rows] == [True, False, True]
    md = to_markdown(summary, rows, {"date": "2026-01-01", "Hardware": "test"})
    assert "| Questions (N) | 3 |" in md
    assert "not in top 10" in md


def _gateway(handler) -> GatewayClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GatewayClient("http://gw.test/v1", "k", "llama3.2", bypass_cache=True, http=http)


def test_gateway_client_parses_usage_and_headers() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["cache"] = request.headers.get("cache-control")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
            headers={
                "x-gateway-cost-usd": "0.00001200",
                "x-gateway-backend": "ollama",
                "x-gateway-cache": "BYPASS",
            },
        )

    result = _gateway(handler).chat([{"role": "user", "content": "hello"}])
    assert seen["url"] == "http://gw.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["cache"] == "no-cache"
    assert seen["body"]["model"] == "llama3.2"
    assert seen["body"]["temperature"] == 0.0
    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("Hi", 12, 3)
    assert result.cost_usd == pytest.approx(1.2e-5)
    assert (result.backend, result.cache) == ("ollama", "BYPASS")


def test_gateway_client_raises_on_error() -> None:
    client = _gateway(lambda r: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(GatewayError, match="401"):
        client.chat([{"role": "user", "content": "hello"}])
