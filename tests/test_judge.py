import pytest

from fakes import FakeChat, hit
from healthcare_rag.citations import LexicalScorer
from healthcare_rag.judge import agreement, ask_correct, cohen_kappa, scorer_faithful


def test_cohen_kappa() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    a = agreement([True, True, True, False], [True, False, True, False])
    assert a["agreement"] == 0.75
    assert a["reference_true_matched"] == "2/3" and a["reference_false_matched"] == "1/1"
    assert a["kappa"] == pytest.approx(0.5)


def test_ask_correct_parses_reply() -> None:
    chat = FakeChat("CORRECT")
    assert ask_correct(chat, "q?", "passage", [["58.9"], ["97.4"]], "58.9% to 97.4%")
    assert "58.9; 97.4" in chat.calls[0][-1]["content"]
    assert not ask_correct(FakeChat("INCORRECT"), "q?", "p", [["a"]], "b")


def test_scorer_faithful_needs_every_sentence_supported() -> None:
    passages = [hit(1, "Metformin lowered HbA1c by 1.1% over 12 weeks.")]
    lex = LexicalScorer()
    assert scorer_faithful("Metformin lowered HbA1c by 1.1% [1].", passages, lex, 0.6)
    assert not scorer_faithful("Metformin lowered HbA1c by 2.5% [1].", passages, lex, 0.6)
