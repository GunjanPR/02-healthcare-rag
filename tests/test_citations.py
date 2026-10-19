import pytest

from fakes import FakeChat, FakeRetriever, hit
from healthcare_rag.citations import (
    LexicalScorer,
    attribute,
    check,
    drop_unsupported,
    numbers,
    relabel,
    split_sentences,
)
from healthcare_rag.rag import RAGConfig, RAGPipeline
from healthcare_rag.rerank import EnsembleReranker
from healthcare_rag.types import Hit

ACT = (
    "The Asthma Control Test is useful for excluding exercise-induced bronchoconstriction "
    "(EIB) in children aged 12 and older."
)
SENS = "Sensitivity of the test was 27.8% and specificity was 43.9%."


def test_split_sentences_keeps_citations_and_abbreviations() -> None:
    text = "ACT helps exclude EIB [1]. Sensitivity was 27.8% [2] e.g. in teens. Done."
    assert split_sentences(text) == [
        "ACT helps exclude EIB [1].",
        "Sensitivity was 27.8% [2] e.g. in teens.",
        "Done.",
    ]
    assert split_sentences("Yes. [1] It works [2].") == ["Yes. [1]", "It works [2]."]
    assert split_sentences("  ") == []


def test_numbers_normalized() -> None:
    assert numbers("1,234 people, 2.50 mg, 10.0%, 3,5 mmol [3]") == {"1234", "2.5", "10", "3.5"}


@pytest.mark.parametrize(
    ("sentence", "passage", "low", "high"),
    [
        ("The ACT is useful for excluding EIB [1].", ACT, 1.0, 1.0),  # acronyms
        ("Sensitivity was 27.8% [1].", SENS, 1.0, 1.0),
        ("Sensitivity was 72.8% [1].", SENS, 0.0, 0.0),  # wrong number
        ("Statins reduce mortality in sepsis [1].", ACT, 0.0, 0.2),
        ("Yes.", ACT, 1.0, 1.0),  # nothing to check
    ],
)
def test_lexical_scorer(sentence: str, passage: str, low: float, high: float) -> None:
    assert low <= LexicalScorer().score(sentence, passage) <= high


def test_negation_is_a_content_word() -> None:
    s = LexicalScorer()
    assert s.score("Metformin did not reduce HbA1c.", "Metformin did reduce HbA1c.") < 1.0


def test_check_flags_wrong_citation_and_relabel_fixes_it() -> None:
    report = check("ACT is useful for excluding EIB [2]. Sensitivity was 27.8% [2].", [ACT, SENS])
    assert [s.supported for s in report.sentences] == [False, True]
    assert report.sentences[0].best_passage == 1
    fixed = relabel(report)
    assert fixed.relabeled == 1
    assert fixed.answer == "ACT is useful for excluding EIB [1]. Sensitivity was 27.8% [2]."
    assert fixed.all_supported


def test_drop_unsupported_removes_hallucinated_sentence() -> None:
    report = check("Sensitivity was 27.8% [2]. It cures asthma in 99% of cases [1].", [ACT, SENS])
    kept = drop_unsupported(relabel(report))
    assert kept.answer == "Sensitivity was 27.8% [2]."


def test_attribute_adds_citations_per_sentence() -> None:
    report = attribute("ACT is useful for excluding EIB. Sensitivity was 27.8%.", [ACT, SENS])
    assert report.attributed
    assert report.answer == "ACT is useful for excluding EIB [1]. Sensitivity was 27.8% [2]."
    nothing = attribute("Statins reduce sepsis mortality.", [ACT, SENS])
    assert not nothing.attributed


def test_pipeline_attribute_mode_skips_the_llm_repair_turn() -> None:
    chat = FakeChat("Sensitivity was 27.8%.")
    pipe = RAGPipeline(
        FakeRetriever([hit(1, ACT), hit(2, SENS)]),
        chat,
        config=RAGConfig(citation_repair="attribute"),
    )
    result = pipe.ask("What was the sensitivity?")
    assert not result.abstained
    assert result.answer == "Sensitivity was 27.8% [2]."
    assert result.attributed and result.usage.llm_calls == 1
    assert result.support_rate == 1.0
    assert [h.id for h in result.cited_passages] == ["c2"]


def test_pipeline_attribute_then_llm_falls_back_to_repair() -> None:
    chat = FakeChat("Statins cure sepsis.", "INSUFFICIENT_EVIDENCE")
    pipe = RAGPipeline(
        FakeRetriever([hit(1, ACT)]), chat, config=RAGConfig(citation_repair="attribute_then_llm")
    )
    result = pipe.ask("q?")
    assert result.abstained and result.reason == "model_insufficient"
    assert result.usage.llm_calls == 2


def test_pipeline_drop_mode_abstains_when_nothing_is_supported() -> None:
    chat = FakeChat("Statins cure sepsis in 99% of cases [1].")
    pipe = RAGPipeline(FakeRetriever([hit(1, ACT)]), chat, config=RAGConfig(citation_check="drop"))
    result = pipe.ask("q?")
    assert result.abstained and result.reason == "unsupported"


def test_pipeline_relabel_mode_fixes_citation() -> None:
    chat = FakeChat("Sensitivity was 27.8% [1].")
    pipe = RAGPipeline(
        FakeRetriever([hit(1, ACT), hit(2, SENS)]), chat, config=RAGConfig(citation_check="relabel")
    )
    result = pipe.ask("q?")
    assert result.answer == "Sensitivity was 27.8% [2]."
    assert result.relabeled == 1


def test_prompt_version_is_validated_and_used() -> None:
    with pytest.raises(ValueError):
        RAGConfig(prompt="v9")
    chat = FakeChat("A [1].")
    RAGPipeline(FakeRetriever([hit(1)]), chat, config=RAGConfig(prompt="v2")).ask("q?")
    assert "Start with the direct answer" in chat.calls[0][0]["content"]


class _Fixed:
    def __init__(self, order: list[str], score: float = 0.0) -> None:
        self.order = order
        self.score = score

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        by_id = {h.id: h for h in hits}
        return [
            Hit(i, by_id[i].title, by_id[i].text, self.score - n, by_id[i].doc_id)
            for n, i in enumerate(self.order)
        ]


def test_ensemble_fuses_order_but_keeps_primary_scores() -> None:
    hits = [hit(1), hit(2), hit(3)]
    primary = _Fixed(["c1", "c2", "c3"], score=9.0)
    second = _Fixed(["c3", "c2", "c1"])
    third = _Fixed(["c2", "c3", "c1"])
    fused = EnsembleReranker([primary, second, third]).rerank("q", hits)
    assert [h.id for h in fused] == ["c2", "c3", "c1"]
    assert {h.id: h.score for h in fused} == {"c1": 9.0, "c2": 8.0, "c3": 7.0}
    with pytest.raises(ValueError):
        EnsembleReranker([])
