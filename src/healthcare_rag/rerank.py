"""Cross-encoder reranking (default model: ncbi/MedCPT-Cross-Encoder).

MedCPT's cross-encoder was trained by NCBI on PubMed search logs to score
(query, article passage) pairs. It outputs one relevance logit per pair. We keep the raw
logit as the hit score: a sigmoid saturates at 0 or 1 for most pairs (logits are often
beyond +/-10), which would create ties. ABSTAIN_THRESHOLD is therefore in logit units.

Needs the optional extra: pip install -e ".[rerank]"  (torch + transformers).
"""

from dataclasses import replace

from healthcare_rag.types import Hit


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "ncbi/MedCPT-Cross-Encoder",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
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
