import pytest

from fakes import FakeChat, FakeRetriever, hit
from healthcare_rag.baseline import BM25Baseline
from healthcare_rag.pubmedqa import (
    ANSWER_INSTRUCTION,
    Item,
    doc_recall,
    macro_f1,
    parse_items,
    parse_label,
    summarize,
)
from healthcare_rag.rag import RAGPipeline


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Yes, the trial showed a benefit [1].", "yes"),
        ("**No.** Mortality did not differ.", "no"),
        ("Maybe: the evidence is mixed.", "maybe"),
        ("[1] The results suggest yes, with caveats.", "yes"),
        ("The data do not provide enough information; no answer.", None),
        ("INSUFFICIENT_EVIDENCE", None),
        ("The study was not powered for this outcome.", None),
        ("Noted improvements were seen.", None),  # "noted" is not "no"
    ],
)
def test_parse_label(text: str, label: str | None) -> None:
    assert parse_label(text) == label


def test_macro_f1() -> None:
    gold = ["yes", "yes", "no", "maybe"]
    assert macro_f1(gold, ["yes", "yes", "no", "maybe"]) == 1.0
    # yes: P=1 R=1/2; no: P=1/2 R=1; maybe: 0 -> (2/3 + 2/3 + 0) / 3
    assert macro_f1(gold, ["yes", "no", "no", None]) == pytest.approx(4 / 9)
    assert macro_f1(gold, [None] * 4) == 0.0


def test_parse_items_uses_official_split() -> None:
    pqal = {
        "1": {"QUESTION": "q1?", "CONTEXTS": ["a", "b"], "final_decision": "yes", "YEAR": "2011"},
        "2": {"QUESTION": "q2?", "CONTEXTS": ["c"], "final_decision": "no"},
    }
    items = {i.pmid: i for i in parse_items(pqal, {"2"})}
    assert items["1"].split == "dev" and items["2"].split == "test"
    assert items["1"].context == "a b" and items["1"].year == 2011
    chunk = items["1"].to_chunk()
    assert chunk.chunk_id == "1:0" and chunk.doc_id == "1" and chunk.title == ""


def test_doc_recall_and_summarize() -> None:
    assert doc_recall(["9:0", "1:0"], "1", 1) == 0.0
    assert doc_recall(["9:0", "1:0"], "1", 5) == 1.0
    items = {
        "1": Item("1", "test", "q?", "ctx", "yes"),
        "2": Item("2", "test", "q?", "ctx", "no"),
    }

    def row(id_: str, pred: str | None, reason: str = "answered") -> dict:
        return {
            "id": id_,
            "system": "full",
            "pred": pred,
            "reason": reason,
            "passages": [f"{id_}:0"],
            "latency_s": 1.0,
            "prompt_tokens": 10,
            "llm_calls": 1,
        }

    s = summarize([row("1", "yes"), row("2", None, "low_score")], items)["full"]
    assert s["accuracy"] == 0.5 and s["answered_rate"] == 0.5
    assert s["low_score_abstained"] == 0.5 and s["doc_recall@1"] == 1.0
    assert s["pred_counts"] == {"yes": 1, "no": 0, "maybe": 0, None: 1}


def test_instruction_reaches_llm_but_not_retrieval() -> None:
    retriever = FakeRetriever([hit(1, "Aspirin reduced events.")])
    chat = FakeChat("Yes, aspirin reduced events [1].")
    RAGPipeline(retriever, chat).ask("Does aspirin help?", instruction=ANSWER_INSTRUCTION)
    assert retriever.calls[0][0] == "Does aspirin help?"
    assert chat.calls[0][-1]["content"].endswith(f"Does aspirin help? {ANSWER_INSTRUCTION}")

    retriever = FakeRetriever([hit(1)])
    chat = FakeChat("No.")
    BM25Baseline(retriever, chat, k=1).answer("Q?", ANSWER_INSTRUCTION)
    assert retriever.calls[0][0] == "Q?"
    assert chat.calls[0][-1]["content"].endswith(f"Q? {ANSWER_INSTRUCTION}")
