import argparse
import json

import pytest

from fakes import FakeChat, hit
from healthcare_rag import judge
from healthcare_rag.citations import LexicalScorer
from healthcare_rag.judge import (
    agreement,
    ask_correct,
    cohen_kappa,
    scorer_faithful,
    wilson_interval,
)
from healthcare_rag.rag_eval import Question


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


def test_wilson_interval() -> None:
    low, high = wilson_interval(76, 88)
    assert low == pytest.approx(0.777, abs=1e-3) and high == pytest.approx(0.920, abs=1e-3)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)  # stays inside [0, 1]
    assert wilson_interval(10, 10)[0] > 0.6


def _question(qid: str, answerable: bool = True) -> Question:
    return Question(qid, "test", answerable, "t", f"question {qid}?", ["c1"], ["PMC1"], [["42"]])


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """judge commands without OpenSearch or the gateway: chunks and the judge are fakes."""
    questions = [_question("q1"), _question("q2"), _question("q3", answerable=False)]
    monkeypatch.setattr(judge, "load_questions", lambda: questions)
    monkeypatch.setattr(judge, "fetch_chunks", lambda settings, ids: {i: hit(1) for i in ids})
    rows = [
        {"id": "q1", "system": "full", "abstained": False, "answer": "It is 42 [1].",
         "passages": ["c1"]},
        {"id": "q2", "system": "full", "abstained": False, "answer": "It is 41 [1].",
         "passages": ["c1"]},
        {"id": "q3", "system": "full", "abstained": True, "answer": "", "passages": []},
    ]  # fmt: skip
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"rows": rows}))
    return answers


def _args(answers, **kw) -> argparse.Namespace:
    base = {"answers": str(answers), "split": "test", "systems": "", "model": "m",
            "scorer": "judge", "out": "", "nli_model": "", "nli_threshold": 0.5,
            "lexical_threshold": 0.6}  # fmt: skip
    return argparse.Namespace(**(base | kw))


def test_faithfulness_skips_abstentions_and_reports_rows_and_ci(offline, monkeypatch, tmp_path):
    chat = FakeChat("YES", "NO")
    monkeypatch.setattr(judge, "judge_client", lambda settings, model: chat)
    out = tmp_path / "faithful.json"
    judge.cmd_faithfulness(_args(offline, out=str(out)), settings=None)
    result = json.loads(out.read_text())
    assert len(chat.calls) == 2  # the abstained row is never judged
    assert result["summary"]["full"]["n_answered"] == 2
    assert result["summary"]["full"]["faithful_rate"] == 0.5
    low, high = result["summary"]["full"]["ci95"]
    assert low < 0.5 < high
    assert [r["faithful"] for r in result["rows"]] == [True, False]


def test_correctness_lists_disagreements(offline, monkeypatch, tmp_path):
    # q1 matches the keyword "42" but the judge says no; q2 matches neither
    monkeypatch.setattr(judge, "judge_client", lambda s, m: FakeChat("INCORRECT", "INCORRECT"))
    out = tmp_path / "correct.json"
    judge.cmd_correctness(_args(offline, out=str(out)), settings=None)
    result = json.loads(out.read_text())
    assert result["summary"]["full"]["keyword_accuracy"] == 0.5
    assert result["summary"]["full"]["judge_accuracy"] == 0.0
    assert [d["id"] for d in result["disagreements"]] == ["q1"]
    assert result["disagreements"][0]["answer"] == "It is 42 [1]."
