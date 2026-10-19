"""Model-based support scoring with an NLI cross-encoder (optional; needs the [rerank] extra).

Scores P(entailment) of (passage, sentence). Used as a stricter alternative to the
lexical scorer in citations.py, both at query time and for the faithfulness metric in
evaluation. Default model: cross-encoder/nli-deberta-v3-base (labels: contradiction,
entailment, neutral).
"""

from functools import lru_cache


class NLIScorer:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval().to(device)
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError(f"{model_name} has no 'entailment' label: {labels}")
        self.entailment = labels["entailment"]
        self.device = device
        self.max_length = max_length
        self._cached = lru_cache(maxsize=4096)(self._score)

    def _score(self, sentence: str, passage: str) -> float:
        from healthcare_rag.citations import strip_citations

        with self._torch.inference_mode():
            enc = self.tokenizer(
                passage,
                strip_citations(sentence).strip(),
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            probs = self.model(**enc).logits.softmax(-1)[0]
        return float(probs[self.entailment])

    def score(self, sentence: str, passage: str) -> float:
        return self._cached(sentence, passage)
