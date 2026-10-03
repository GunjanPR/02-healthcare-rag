"""Cross-encoder reranking (default model: ncbi/MedCPT-Cross-Encoder).

MedCPT's cross-encoder was trained by NCBI on PubMed search logs to score
(query, article passage) pairs. It outputs one relevance logit per pair. We keep the raw
logit as the hit score: a sigmoid saturates at 0 or 1 for most pairs (logits are often
beyond +/-10), which would create ties. ABSTAIN_THRESHOLD is therefore in logit units.

Needs the optional extra: pip install -e ".[rerank]"  (torch + transformers).
"""

from dataclasses import replace
from typing import Protocol

from healthcare_rag.types import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Cross-Encoder",
        device: str = "cpu",
        max_length: int = 512,
        # 32 covers the default 20 candidates in one forward pass (16 needed two)
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval().to(device)
        if device.startswith("cuda"):
            self.model.half()  # fp16 on GPU: same ranking, roughly half the latency
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def scores(self, query: str, texts: list[str]) -> list[float]:
        logits: list[float] = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                pairs = [[query, t] for t in texts[i : i + self.batch_size]]
                enc = self.tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits += self.model(**enc).logits.squeeze(-1).float().tolist()
        return logits

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        logits = self.scores(query, [f"{h.title}. {h.text}" for h in hits])
        scored = [replace(h, score=s) for h, s in zip(hits, logits, strict=True)]
        return sorted(scored, key=lambda h: h.score, reverse=True)


class EnsembleReranker:
    """Reciprocal rank fusion of several rerankers (and optionally the retriever's order).

    Each member reorders the same candidates; a hit's fused score is the weighted sum of
    1 / (rrf_k + rank) over members. The returned `score` stays the *primary* member's
    score (the first reranker), so the abstention threshold keeps its units; only the
    order changes.
    """

    def __init__(
        self,
        rerankers: list[Reranker],
        weights: list[float] | None = None,
        retrieval_weight: float = 0.0,
        rrf_k: int = 60,
    ) -> None:
        if not rerankers:
            raise ValueError("need at least one reranker")
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)
        if len(self.weights) != len(rerankers):
            raise ValueError("one weight per reranker")
        self.retrieval_weight = retrieval_weight
        self.rrf_k = rrf_k

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        rankings = [r.rerank(query, hits) for r in self.rerankers]
        fused: dict[str, float] = {}
        members = list(zip(rankings, self.weights, strict=True))
        if self.retrieval_weight:
            members.append((hits, self.retrieval_weight))
        for ranking, weight in members:
            for rank, h in enumerate(ranking, start=1):
                fused[h.id] = fused.get(h.id, 0.0) + weight / (self.rrf_k + rank)
        primary = {h.id: h for h in rankings[0]}
        return sorted(primary.values(), key=lambda h: fused[h.id], reverse=True)
