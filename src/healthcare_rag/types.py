"""Shared data types for chunks, search hits and filters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One indexed passage. For PMC, doc_id is the normalized PMCID (e.g. "PMC1234567")."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    section: str = ""
    position: int = 0  # order within the article
    year: int | None = None
    journal: str = ""
    topic: str = ""  # the search topic the article was collected under


@dataclass(frozen=True)
class Hit:
    id: str  # chunk_id
    title: str
    text: str
    score: float
    doc_id: str = ""
    section: str = ""
    year: int | None = None
    journal: str = ""


@dataclass(frozen=True)
class Filters:
    """Metadata filters applied in both OpenSearch and Qdrant."""

    sections: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()
    year_min: int | None = None
    year_max: int | None = None

    def is_empty(self) -> bool:
        return not (self.sections or self.topics or self.doc_ids or self.year_min or self.year_max)
