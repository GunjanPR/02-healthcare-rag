import json

import pytest

from healthcare_rag.ingest import (
    _pack,
    build_chunks,
    chunk_article,
    doc_pmcid,
    is_open_license,
    normalize_pmcid,
    read_chunks,
    write_chunks,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("PMC123", "PMC123"), ("pmc123", "PMC123"), ("123", "PMC123"), (" PMC0123 ", "PMC123")],
)
def test_normalize_pmcid(raw: str, expected: str) -> None:
    assert normalize_pmcid(raw) == expected


@pytest.mark.parametrize("raw", ["", "unknown", "PMID123", "PMC12a"])
def test_normalize_pmcid_rejects(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_pmcid(raw)


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("licensed under https://creativecommons.org/licenses/by/4.0/", True),
        ("CC0 public domain dedication", True),
        ("This is an open access article under the CC BY license", True),
        ("https://creativecommons.org/licenses/by-nc/4.0/", False),
        ("CC BY-NC-ND 4.0", False),
        ("All rights reserved", False),
    ],
)
def test_is_open_license(text: str, ok: bool) -> None:
    assert is_open_license(text) is ok


def _passage(section_type: str, ptype: str, text: str) -> dict:
    return {"infons": {"section_type": section_type, "type": ptype}, "text": text}


def bioc_doc(pmc: str = "123") -> dict:
    words = " ".join(f"w{i}" for i in range(200)) + "."
    return {
        "id": pmc,
        "passages": [
            {
                "infons": {"section_type": "TITLE", "type": "front", "article-id_pmc": pmc},
                "text": "A trial of X",
            },
            _passage("ABSTRACT", "abstract", "Background. We tested X in 200 adults over a year."),
            _passage("INTRO", "title_1", "Introduction"),
            _passage("INTRO", "paragraph", words),
            _passage("INTRO", "paragraph", words),
            _passage("METHODS", "paragraph", "Adults were randomized to drug X or to placebo."),
            _passage("TABLE", "table", "a | b | c | d | e | f | g | h | i"),
            _passage("REF", "ref", "Smith J. Some cited paper. 2020."),
            _passage("RESULTS", "paragraph", "short"),  # under 8 words: dropped
        ],
    }


def test_doc_pmcid_falls_back_to_front_matter() -> None:
    doc = bioc_doc("PMC77")
    assert doc_pmcid(doc) == "PMC77"
    doc["id"] = "unknown"
    doc["passages"][0]["infons"]["article-id_pmc"] = "77"
    assert doc_pmcid(doc) == "PMC77"
    doc["passages"][0]["infons"]["article-id_pmc"] = ""
    assert doc_pmcid(doc) is None


def test_chunk_article_is_section_aware() -> None:
    chunks = chunk_article(bioc_doc(), {"year": 2021, "topic": "asthma", "journal": "J"})
    assert [c.section for c in chunks] == ["abstract", "introduction", "introduction", "methods"]
    assert [c.chunk_id for c in chunks] == ["PMC123:0", "PMC123:1", "PMC123:2", "PMC123:3"]
    assert all(c.doc_id == "PMC123" and c.title == "A trial of X" for c in chunks)
    assert all(c.year == 2021 and c.topic == "asthma" for c in chunks)
    assert not any("cited paper" in c.text or "|" in c.text for c in chunks)


@pytest.mark.parametrize("first_word", ["Word", "word"])  # "word": no sentence breaks found
def test_pack_splits_long_paragraphs(first_word: str) -> None:
    sentence = " ".join([first_word] + ["word"] * 49) + "."
    long_paragraph = " ".join([sentence] * 10)  # 500 words
    chunks = list(_pack([long_paragraph]))
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 320 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 500


def test_build_chunks_dedupes_and_round_trips(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "PMC1.json").write_text(json.dumps(bioc_doc("1")))
    (raw / "PMC2.json").write_text(json.dumps(bioc_doc("2")))  # same text, other article
    manifest = {"articles": {"PMC1": {"topic": "a"}, "PMC2": {"topic": "b"}, "PMC3": {}}}
    chunks, stats = build_chunks(raw, manifest)
    # The two identical intro paragraphs collapse too, then all of PMC2 is a duplicate
    assert stats == {"articles": 2, "chunks": 3, "duplicate_chunks_dropped": 5}
    assert {c.doc_id for c in chunks} == {"PMC1"}

    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    assert read_chunks(path) == chunks
