"""Sentence-level citation checking, and attaching citations without another LLM call.

Every answer sentence should cite the passage(s) it comes from. This module scores how
well a passage supports a sentence and uses that score three ways:

- check:     for each sentence, is it supported by a passage it cites?
- relabel:   a sentence whose cited passage does not support it, but another provided
             passage does, gets its citation pointed at that passage.
- attribute: an answer with no citations at all gets, per sentence, the passage that
             supports it best. This replaces the follow-up LLM turn ("please add
             citations"), which was the slowest path of the pipeline.

The default scorer is lexical and deterministic: the share of a sentence's content words
(lower-cased, light plural folding, stopwords removed) that appear in the passage, with a
hard rule that every number in the sentence must appear in the passage. A model-based
scorer (an NLI cross-encoder, see nli.py) can be plugged in through `SupportScorer`.
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

_CITATION = re.compile(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]")
# Split after ., ! or ? (optionally followed by citation markers) when the next sentence
# starts with a capital letter, digit or bracket. Avoids splitting "e.g. the" and "3.5".
_SENTENCE_END = re.compile(r"(?<=[.!?])((?:\s*\[[\d,;\s–-]+\])*)\s+(?=[A-Z0-9(\[])")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")

_STOPWORDS = """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into is
    it its itself just may me might more most must my now of off on once only
    or other our out over own same she should so some such than that the their them then
    there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your study studies patients patient
    passage passages results result showed shows show found reported compared including
    associated among however therefore use used using based"""
# Negations stay content words, so "did not reduce" needs a "not" in the passage.
STOPWORDS = frozenset(_STOPWORDS.split())


def _fold(token: str) -> str:
    """Light plural folding: 'patients' -> 'patient', 'studies' -> 'study'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def strip_citations(text: str) -> str:
    return _CITATION.sub(" ", text)


def content_terms(text: str) -> set[str]:
    tokens = _TOKEN.findall(strip_citations(text).lower().replace(",", ""))
    return {_fold(t) for t in tokens if t not in STOPWORDS and len(t) > 1 or t.isdigit()}


def numbers(text: str) -> set[str]:
    """Numbers as written, with thousands separators removed: '1,234' -> '1234'."""
    found = set()
    for n in _NUMBER.findall(strip_citations(text)):
        n = n.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", n) else n.replace(",", ".")
        found.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return found


def split_sentences(text: str) -> list[str]:
    """Sentences with their trailing citation markers attached."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_END.split(text)
    # re.split with one group returns [s0, cites0, s1, cites1, ..., sN]
    sentences = []
    for i in range(0, len(parts), 2):
        cites = parts[i + 1] if i + 1 < len(parts) else ""
        sentence = (parts[i] + (cites or "")).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def cited_numbers(sentence: str, n_passages: int) -> list[int]:
    from healthcare_rag.rag import parse_citations  # avoid an import cycle

    return parse_citations(sentence, n_passages)


class SupportScorer(Protocol):
    def score(self, sentence: str, passage: str) -> float:
        """Support of `sentence` by `passage`, in [0, 1]."""
        ...


def acronyms(text: str, max_len: int = 6) -> set[str]:
    """Initials of every run of 2..max_len words: 'Asthma Control Test' -> 'act', 'ac', ..."""
    words = [w[0].lower() for w in re.findall(r"[A-Za-z][A-Za-z-]*", text)]
    return {
        "".join(words[i : i + n]) for n in range(2, max_len + 1) for i in range(len(words) - n + 1)
    }


class LexicalScorer:
    """Share of the sentence's content words found in the passage; 0 if a number is not.

    An acronym in the sentence (ACT) also matches the initials of a phrase in the passage
    (Asthma Control Test), since answers often abbreviate what the passage spells out.
    """

    def score(self, sentence: str, passage: str) -> float:
        terms = content_terms(sentence) - {"yes", "no"}
        if not terms:
            return 1.0  # nothing checkable ("Yes." / "No.")
        if not numbers(sentence) <= numbers(passage):
            return 0.0
        missing = terms - content_terms(passage)
        if missing:
            caps = {_fold(a.lower()) for a in re.findall(r"\b[A-Z][A-Za-z]*[A-Z]s?\b", sentence)}
            if missing & caps:
                missing -= caps & acronyms(passage)
        return 1 - len(missing) / len(terms)


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[int]  # 1-based passage numbers the sentence cites
    support: float  # best support among the cited passages (0 if none cited)
    best_passage: int  # 1-based passage with the highest support overall (0 if none)
    best_support: float
    supported: bool


@dataclass
class CitationReport:
    answer: str  # the answer after attribution/relabeling (unchanged when mode is "off")
    sentences: list[SentenceCheck] = field(default_factory=list)
    attributed: bool = False  # citations were added to an uncited answer
    relabeled: int = 0  # sentences whose citation was pointed at a better passage

    @property
    def supported_rate(self) -> float:
        return (
            sum(s.supported for s in self.sentences) / len(self.sentences)
            if self.sentences
            else 0.0
        )

    @property
    def all_supported(self) -> bool:
        return bool(self.sentences) and all(s.supported for s in self.sentences)


def check(
    answer: str,
    passages: list[str],
    scorer: SupportScorer | None = None,
    threshold: float = 0.6,
) -> CitationReport:
    """Score each sentence against the passages it cites (and against all passages)."""
    scorer = scorer or LexicalScorer()
    report = CitationReport(answer=answer)
    for sentence in split_sentences(answer):
        cited = cited_numbers(sentence, len(passages))
        scores = [scorer.score(sentence, p) for p in passages]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else -1
        own = max((scores[n - 1] for n in cited), default=0.0)
        report.sentences.append(
            SentenceCheck(
                sentence=sentence,
                cited=cited,
                support=own,
                best_passage=best + 1,
                best_support=scores[best] if scores else 0.0,
                supported=bool(cited) and own >= threshold,
            )
        )
    return report


def _with_citation(sentence: str, n: int) -> str:
    body = re.sub(r"\s+(?=[.!?,;:])|(?<=\s)\s+", "", _CITATION.sub("", sentence)).rstrip()
    if body and body[-1] in ".!?":
        return f"{body[:-1]} [{n}]{body[-1]}"
    return f"{body} [{n}]"


def attribute(
    answer: str,
    passages: list[str],
    scorer: SupportScorer | None = None,
    threshold: float = 0.6,
) -> CitationReport:
    """Add the best-supporting passage to each sentence of an uncited answer.

    Sentences no passage supports above `threshold` stay uncited; the answer counts as
    attributed only when at least one sentence received a citation.
    """
    report = check(strip_citations(answer), passages, scorer, threshold)
    out = []
    for s in report.sentences:
        if s.best_support >= threshold and s.best_passage:
            out.append(_with_citation(s.sentence, s.best_passage))
            s.cited, s.support, s.supported = [s.best_passage], s.best_support, True
        else:
            out.append(s.sentence)
    report.answer = " ".join(out)
    report.attributed = any(s.cited for s in report.sentences)
    return report


def relabel(report: CitationReport, threshold: float = 0.6) -> CitationReport:
    """Point unsupported citations at the passage that supports the sentence, if any."""
    out = []
    for s in report.sentences:
        if s.cited and not s.supported and s.best_support >= threshold:
            s.sentence = _with_citation(s.sentence, s.best_passage)
            s.cited, s.support, s.supported = [s.best_passage], s.best_support, True
            report.relabeled += 1
        out.append(s.sentence)
    report.answer = " ".join(out)
    return report


def drop_unsupported(report: CitationReport) -> CitationReport:
    """Keep only supported sentences (the answer is empty if none is supported)."""
    report.sentences = [s for s in report.sentences if s.supported]
    report.answer = " ".join(s.sentence for s in report.sentences)
    return report
