import math

import pytest

from fakes import FakeChat, FakeReranker, FakeRetriever, hit
from healthcare_rag.experiments import (
    default_variants,
    fusion_table,
    parse_overrides,
    sweep_thresholds,
)
from healthcare_rag.rag import INSUFFICIENT, RAGConfig, RAGPipeline


def test_parse_overrides_validates_names() -> None:
    assert parse_overrides(["PROMPT_VERSION=v2", "RERANK_TOP_K = 8"]) == {
        "prompt_version": "v2",
        "rerank_top_k": "8",
    }
    with pytest.raises(SystemExit):
        parse_overrides(["NOT_A_SETTING=1"])
    with pytest.raises(SystemExit):
        parse_overrides(["PROMPT_VERSION"])


def test_sweep_replays_the_gate() -> None:
    rows = [
        {"id": "a", "top_score": 10.0, "abstained": False},
        {"id": "b", "top_score": 2.0, "abstained": False},  # answerable, low score
        {"id": "c", "top_score": 5.0, "abstained": False},  # out of scope, answered
    ]
    correct = {"a": True, "b": True, "c": False}
    answerable = {"a": True, "b": True, "c": False}
    by_t = {s["threshold"]: s for s in sweep_thresholds(rows, correct, answerable)}
    assert by_t[-math.inf]["accuracy"] == 1.0 and by_t[-math.inf]["oos_declined"] == 0.0
    assert by_t[5.0]["accuracy"] == 0.5 and by_t[5.0]["oos_declined"] == 0.0
    assert by_t[10.0]["accuracy"] == 0.5 and by_t[10.0]["oos_declined"] == 1.0


def test_fusion_table_from_static_scores() -> None:
    data = {
        "queries": [{"bench": "pmc", "question": "q", "candidates": ["x", "y"], "relevant": ["y"]}],
        "scores": {"m1": [[0.0, 1.0]], "m2": [[1.0, 0.0]]},
    }
    table = fusion_table(data, default_variants(["m1", "m2"]))
    assert table["m1"]["ndcg@10_pmc"] == 1.0  # m1 puts y first
    assert table["m2"]["ndcg@10_pmc"] < 1.0
    assert table["hybrid (no rerank)"]["ndcg@10_pmc"] < 1.0  # retriever order: x, y
    assert table["rrf(all), first model x2"]["ndcg@10_pmc"] == 1.0


def test_abstain_modes() -> None:
    hits = [hit(1), hit(2)]
    # model mode: no reranker threshold, so a low score still reaches the LLM
    chat = FakeChat("Answer [1].")
    cfg = RAGConfig(abstain_threshold=5.0, abstain_mode="model")
    r = RAGPipeline(FakeRetriever(hits), chat, FakeReranker([1.0, 0.5]), cfg).ask("q")
    assert not r.abstained and len(chat.calls) == 1
    # threshold mode: the prompt has no rule for declining
    chat = FakeChat("Answer [1].")
    cfg = RAGConfig(abstain_threshold=0.0, abstain_mode="threshold")
    RAGPipeline(FakeRetriever(hits), chat, FakeReranker([1.0, 0.5]), cfg).ask("q")
    assert INSUFFICIENT not in chat.calls[0][0]["content"]
    with pytest.raises(ValueError):
        RAGConfig(abstain_mode="never")
