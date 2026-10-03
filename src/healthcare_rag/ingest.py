"""PubMed Central Open Access ingestion: download -> parse -> chunk -> index.

Usage:
    python -m healthcare_rag.ingest download --per-topic 50   # NCBI E-utilities + BioC API
    python -m healthcare_rag.ingest chunk                     # data/raw -> data/processed
    python -m healthcare_rag.ingest index                     # OpenSearch (BM25) + Qdrant
    python -m healthcare_rag.ingest all --per-topic 50

Articles come from the PMC Open Access subset, restricted to CC BY / CC0 licenses. Full
text is fetched as BioC JSON, which labels every passage with its section (ABSTRACT,
METHODS, RESULTS, ...). Raw files and chunks are written under DATA_DIR (git-ignored);
the list of PMCIDs, topics and licenses goes to eval/pmc_manifest.json so the corpus can
be rebuilt exactly.

Re-running any step is idempotent: downloads are skipped when the raw file exists, chunk
ids are deterministic (PMCID:position), OpenSearch documents are upserted by chunk id and
Qdrant points by a UUID derived from the chunk id.
"""

import argparse
import hashlib
import json
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from healthcare_rag.config import Settings, get_settings
from healthcare_rag.types import Chunk

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{ids}/unicode"
MANIFEST = Path("eval/pmc_manifest.json")

# Search topics: label -> PMC query (title match keeps articles on-topic).
TOPICS: dict[str, str] = {
    "type2_diabetes": '"type 2 diabetes"[Title]',
    "hypertension": "hypertension[Title]",
    "heart_failure": '"heart failure"[Title]',
    "atrial_fibrillation": '"atrial fibrillation"[Title]',
    "stroke": "stroke[Title]",
    "asthma": "asthma[Title]",
    "sepsis": "sepsis[Title]",
    "tuberculosis": "tuberculosis[Title]",
    "depression": '"major depressive disorder"[Title]',
    "osteoporosis": "osteoporosis[Title]",
}
DATE_RANGE = "2016:2023[pdat]"  # BioC lags PMC by months, so stop at 2023

# BioC section_type -> our section name. Anything not listed (REF, TABLE, FIG, ABBR,
# SUPPL, ACK_FUND, COMP_INT, AUTH_CONT, REVIEW_INFO, ...) is dropped.
SECTIONS = {
    "ABSTRACT": "abstract",
    "INTRO": "introduction",
    "METHODS": "methods",
    "RESULTS": "results",
    "DISCUSS": "discussion",
    "CONCL": "conclusion",
    "CASE": "case",
}
TEXT_TYPES = {"abstract", "paragraph"}
TARGET_WORDS = 220  # chunk size target; chunks never cross a section boundary
MAX_WORDS = 320


# ---- PMCID normalization -----------------------------------------------------------------


def normalize_pmcid(value: str | int) -> str:
    """'PMC123', 'pmc123', '123', ' PMC0123 ' -> 'PMC123'. Raises ValueError otherwise."""
    m = re.fullmatch(r"\s*(?:pmc)?0*(\d+)\s*", str(value), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"not a PMCID: {value!r}")
    return f"PMC{m.group(1)}"


def doc_pmcid(doc: dict[str, Any]) -> str | None:
    """PMCID of a BioC document: its id, else the front matter's article-id_pmc."""
    front = doc["passages"][0]["infons"] if doc.get("passages") else {}
    for value in (doc.get("id", ""), front.get("article-id_pmc", "")):
        try:
            return normalize_pmcid(value)
        except ValueError:
            continue
    return None


def is_open_license(license_text: str) -> bool:
    """CC BY (any version, without NC/ND) or CC0 / public domain."""
    t = license_text.lower()
    if "publicdomain" in t or "cc0" in t:
        return True
    if "creativecommons.org/licenses/by/" in t:
        return True
    return bool(re.search(r"\bcc[ -]by\b(?![ -](nc|nd))", t)) and "non-commercial" not in t


# ---- download ----------------------------------------------------------------------------


class NCBIClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.http = http or httpx.Client(timeout=60, follow_redirects=True)
        self.params = {"tool": "healthcare-rag"}
        if settings.ncbi_email:
            self.params["email"] = settings.ncbi_email
        if settings.ncbi_api_key:
            self.params["api_key"] = settings.ncbi_api_key
        self.min_interval = 0.11 if settings.ncbi_api_key else 0.34  # NCBI: 10 or 3 req/s
        self._last = 0.0

    def _get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            self._last = time.monotonic()
            try:
                resp = self.http.get(url, params=params)
                if resp.status_code != 429 and resp.status_code < 500:
                    return resp
            except httpx.TransportError:
                pass
            time.sleep(2**attempt)
        raise RuntimeError(f"NCBI request failed after retries: {url}")

    def search(self, query: str, retmax: int) -> list[str]:
        term = f'{query} AND "open access"[filter] AND {DATE_RANGE}'
        params = {**self.params, "db": "pmc", "term": term, "retmax": retmax}
        params.update(retmode="json", sort="relevance")
        resp = self._get(f"{EUTILS}/esearch.fcgi", params)
        return [normalize_pmcid(i) for i in resp.json()["esearchresult"]["idlist"]]

    def journals(self, pmcids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(pmcids), 200):
            ids = ",".join(p[3:] for p in pmcids[i : i + 200])
            params = {**self.params, "db": "pmc", "id": ids, "retmode": "json"}
            result = self._get(f"{EUTILS}/esummary.fcgi", params).json().get("result", {})
            for uid in result.get("uids", []):
                out[normalize_pmcid(uid)] = result[uid].get("fulljournalname", "")
        return out

    def bioc(self, pmcids: list[str]) -> list[dict[str, Any]]:
        """BioC documents for the given PMCIDs; articles BioC doesn't have are skipped."""
        resp = self._get(BIOC.format(ids=",".join(pmcids)))
        if not resp.text.lstrip().startswith(("[", "{")):
            return []
        data = resp.json()
        collections = data if isinstance(data, list) else [data]
        return [doc for c in collections for doc in c.get("documents", [])]


def download(settings: Settings, per_topic: int, raw_dir: Path) -> dict[str, Any]:
    """Fetch up to `per_topic` CC BY/CC0 articles per topic. Returns the manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ncbi = NCBIClient(settings)
    manifest: dict[str, Any] = {"date_range": DATE_RANGE, "topics": TOPICS, "articles": {}}
    if MANIFEST.exists():
        manifest["articles"] = json.loads(MANIFEST.read_text(encoding="utf-8"))["articles"]
    articles: dict[str, dict[str, Any]] = manifest["articles"]

    for topic, query in TOPICS.items():
        have = sum(1 for a in articles.values() if a["topic"] == topic)
        if have >= per_topic:
            print(f"{topic}: {have} articles already")
            continue
        # Over-fetch: some articles are not in BioC yet, some are NC/ND licensed, and
        # some were already collected under another topic (deduplicated by PMCID).
        candidates = [p for p in ncbi.search(query, per_topic * 4) if p not in articles]
        for i in range(0, len(candidates), 10):
            if have >= per_topic:
                break
            for doc in ncbi.bioc(candidates[i : i + 10]):
                if have >= per_topic:
                    break
                pmcid = doc_pmcid(doc)
                front = doc["passages"][0]["infons"]
                if pmcid is None:
                    continue
                license_text = front.get("license", "")
                if pmcid in articles or not is_open_license(license_text):
                    continue
                (raw_dir / f"{pmcid}.json").write_text(json.dumps(doc), encoding="utf-8")
                articles[pmcid] = {
                    "topic": topic,
                    "year": int(front["year"]) if str(front.get("year", "")).isdigit() else None,
                    "license": _license_label(license_text),
                    "doi": front.get("article-id_doi", ""),
                }
                have += 1
        print(f"{topic}: {have} articles")

    journals = ncbi.journals([p for p, a in articles.items() if "journal" not in a])
    for pmcid, journal in journals.items():
        articles[pmcid]["journal"] = journal
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def _license_label(text: str) -> str:
    t = text.lower()
    if "publicdomain" in t or "cc0" in t:
        return "CC0"
    m = re.search(r"licenses/by/(\d\.\d)", t)
    return f"CC BY {m.group(1)}" if m else "CC BY"


# ---- parse + chunk -----------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split on sentence ends; anything still over MAX_WORDS is cut on word boundaries."""
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text):
        words = sentence.split()
        for i in range(0, len(words), MAX_WORDS):
            out.append(" ".join(words[i : i + MAX_WORDS]))
    return [s for s in out if s]


def _pack(paragraphs: list[str]) -> Iterator[str]:
    """Greedy-pack paragraphs into ~TARGET_WORDS chunks; split oversized paragraphs."""
    units: list[str] = []
    for p in paragraphs:
        if len(p.split()) <= MAX_WORDS:
            units.append(p)
        else:
            units.extend(_split_sentences(p))
    buf: list[str] = []
    words = 0
    for u in units:
        n = len(u.split())
        if buf and words + n > MAX_WORDS:
            yield " ".join(buf)
            buf, words = [], 0
        buf.append(u)
        words += n
        if words >= TARGET_WORDS:
            yield " ".join(buf)
            buf, words = [], 0
    if buf:
        yield " ".join(buf)


def chunk_article(doc: dict[str, Any], meta: dict[str, Any]) -> list[Chunk]:
    """Section-aware chunks: paragraphs are grouped within one section, never across."""
    pmcid = doc_pmcid(doc) or normalize_pmcid(doc["id"])
    passages = doc["passages"]
    title = next((p["text"] for p in passages if p["infons"].get("type") == "front"), "")
    sections: list[tuple[str, list[str]]] = []
    for p in passages:
        section = SECTIONS.get(p["infons"].get("section_type", ""))
        text = " ".join((p.get("text") or "").split())
        if not section or p["infons"].get("type") not in TEXT_TYPES or len(text.split()) < 8:
            continue
        if sections and sections[-1][0] == section:
            sections[-1][1].append(text)
        else:
            sections.append((section, [text]))

    chunks: list[Chunk] = []
    for section, paragraphs in sections:
        for text in _pack(paragraphs):
            chunks.append(
                Chunk(
                    chunk_id=f"{pmcid}:{len(chunks)}",
                    doc_id=pmcid,
                    title=title,
                    text=text,
                    section=section,
                    position=len(chunks),
                    year=meta.get("year"),
                    journal=meta.get("journal", ""),
                    topic=meta.get("topic", ""),
                )
            )
    return chunks


def _fingerprint(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def build_chunks(raw_dir: Path, manifest: dict[str, Any]) -> tuple[list[Chunk], dict[str, int]]:
    """Chunk every downloaded article, dropping exact-duplicate passages across the corpus."""
    seen: set[str] = set()
    out: list[Chunk] = []
    stats = {"articles": 0, "chunks": 0, "duplicate_chunks_dropped": 0}
    for pmcid, meta in sorted(manifest["articles"].items()):
        path = raw_dir / f"{pmcid}.json"
        if not path.exists():
            continue
        stats["articles"] += 1
        for c in chunk_article(json.loads(path.read_text(encoding="utf-8")), meta):
            fp = _fingerprint(c.text)
            if fp in seen:
                stats["duplicate_chunks_dropped"] += 1
                continue
            seen.add(fp)
            out.append(c)
    stats["chunks"] = len(out)
    return out, stats


def write_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


# ---- CLI ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=["download", "chunk", "index", "all"])
    parser.add_argument("--per-topic", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64, help="embedding batch size")
    args = parser.parse_args(argv)

    settings = get_settings()
    data = Path(settings.data_dir)
    raw_dir, chunks_path = data / "raw", data / "processed" / "chunks.jsonl"

    if args.cmd in ("download", "all"):
        manifest = download(settings, args.per_topic, raw_dir)
        print(f"manifest: {len(manifest['articles'])} articles -> {MANIFEST}")
    if args.cmd in ("chunk", "all"):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        chunks, stats = build_chunks(raw_dir, manifest)
        write_chunks(chunks, chunks_path)
        print(f"chunked: {stats} -> {chunks_path}")
    if args.cmd in ("index", "all"):
        from healthcare_rag.indexing import index_all

        report = index_all(settings, read_chunks(chunks_path), batch_size=args.batch_size)
        print(f"indexed: {report}")


if __name__ == "__main__":
    main()
