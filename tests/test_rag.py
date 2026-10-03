import pytest

from fakes import FakeChat, FakeReranker, FakeRetriever, hit
from healthcare_rag.rag import (
    ABSTAIN_MESSAGE,
    INSUFFICIENT,
    RAGConfig,
    RAGPipeline,
    number_passages,
    parse_citations,
)
from healthcare_rag.types import Filters


@pytest.mark.parametrize(
    ("answer", "n", "expected"),
    [
        ("A [1]. B [2].", 3, [1, 2]),
        ("A [2, 1] and [1].", 3, [2, 1]),
        ("A [1-3].", 5, [1, 2, 3]),
        ("A [1–2; 4].", 5, [1, 2, 4]),
        ("A [7].", 3, []),  # out of range
        ("No citations.", 3, []),
        ("Year [2020] is not a passage", 3, []),
    ],
)
def test_parse_citations(answer: str, n: int, expected: list[int]) -> None:
    assert parse_citations(answer, n) == expected


def test_number_passages_includes_section() -> None:
    text = number_passages([hit(1, "alpha"), hit(2, "beta")])
    assert text.startswith("[1] Title 1 (results)\nalpha")
    assert "[2] Title 2 (results)\nbeta" in text


def test_answer_with_citations() -> None:
    chat = FakeChat("Metformin lowers glucose [1].")
    retriever = FakeRetriever([hit(1), hit(2), hit(3)])
    pipe = RAGPipeline(retriever, chat, config=RAGConfig(retrieval_top_k=10, rerank_top_k=2))
    result = pipe.ask("what lowers glucose?", filters=Filters(sections=("results",)))

    assert not result.abstained
    assert result.reason == "answered"
    assert result.answer == "Metformin lowers glucose [1]."
    assert [p.id for p in result.passages] == ["c1", "c2"]
    assert [h.id for h in result.cited_passages] == ["c1"]
    assert retriever.calls == [("what lowers glucose?", 10, Filters(sections=("results",)))]
    assert result.usage.llm_calls == 1 and result.usage.prompt_tokens == 100
    assert {"retrieve", "rerank", "generate", "total"} <= set(result.timings)
    user_msg = chat.calls[0][1]["content"]
    assert "[1] Title 1" in user_msg and "[3]" not in user_msg


def test_abstains_without_llm_call_when_no_hits() -> None:
    chat = FakeChat("unused")
    result = RAGPipeline(FakeRetriever([]), chat).ask("anything?")
    assert result.abstained and result.reason == "no_hits"
    assert result.answer == ABSTAIN_MESSAGE
    assert chat.calls == []


def test_reranker_reorders_and_low_score_abstains() -> None:
    hits = [hit(1), hit(2), hit(3)]
    chat = FakeChat("Answer [1].")
    cfg = RAGConfig(rerank_top_k=2, abstain_threshold=0.5)

    ok = RAGPipeline(FakeRetriever(hits), chat, FakeReranker([0.9, 0.4, 0.1]), cfg).ask("q?")
    assert [p.id for p in ok.passages] == ["c3", "c2"]
    assert ok.passages[0].score == 0.9
    assert not ok.abstained

    weak = RAGPipeline(FakeRetriever(hits), chat, FakeReranker([0.3, 0.2, 0.1]), cfg).ask("q?")
    assert weak.abstained and weak.reason == "low_score"
    assert len(chat.calls) == 1  # the weak case never reached the LLM


def test_model_insufficient_and_uncited_abstain() -> None:
    hits = [hit(1)]
    insufficient = RAGPipeline(FakeRetriever(hits), FakeChat(INSUFFICIENT)).ask("q?")
    assert insufficient.abstained and insufficient.reason == "model_insufficient"

    uncited = RAGPipeline(FakeRetriever(hits), FakeChat("An answer with no citation.")).ask("q?")
    assert uncited.abstained and uncited.reason == "uncited"
    assert uncited.raw_answer == "An answer with no citation."

    lenient = RAGPipeline(
        FakeRetriever(hits), FakeChat("No citation."), config=RAGConfig(require_citations=False)
    ).ask("q?")
    assert not lenient.abstained


def test_summarize_uses_chunks_in_order() -> None:
    chat = FakeChat("It studied X [1]. It found Y [2].")
    pipe = RAGPipeline(FakeRetriever([]), chat)
    result = pipe.summarize("An article", [hit(1), hit(2), hit(3)], max_passages=2)
    assert result.cited == [1, 2]
    assert "Article: An article" in chat.calls[0][1]["content"]
    assert "[3]" not in chat.calls[0][1]["content"]
    assert pipe.summarize("Empty", []).abstained


def test_uncited_answer_is_repaired_once() -> None:
    chat = FakeChat("An answer with no citation.", "An answer with a citation [1].")
    result = RAGPipeline(FakeRetriever([hit(1)]), chat).ask("q?")
    assert not result.abstained
    assert result.answer == "An answer with a citation [1]."
    assert result.usage.llm_calls == 2
    assert "repair" in result.timings
    assert chat.calls[1][-1]["content"].startswith("Your answer did not cite")
    assert chat.calls[1][-2] == {"role": "assistant", "content": "An answer with no citation."}


def test_repair_gives_up_after_configured_attempts() -> None:
    chat = FakeChat("No citation.", "Still no citation.")
    result = RAGPipeline(FakeRetriever([hit(1)]), chat).ask("q?")
    assert result.abstained and result.reason == "uncited"
    assert result.usage.llm_calls == 2

    none = RAGPipeline(
        FakeRetriever([hit(1)]), FakeChat("No citation."), config=RAGConfig(citation_repairs=0)
    ).ask("q?")
    assert none.abstained and none.usage.llm_calls == 1
