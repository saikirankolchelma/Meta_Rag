"""Loads the fine-tuned DeBERTa router once and scores query complexity in <5ms."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.config import ROUTER_MODEL_DIR

_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(ROUTER_MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(
            str(ROUTER_MODEL_DIR), torch_dtype=torch.float32
        ).to(_device).eval()
    return _model, _tokenizer


def predict_complexity(query: str) -> float:
    model, tokenizer = _load()
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=64).to(_device)
    with torch.no_grad():
        score = model(**inputs).logits.item()
    return max(0.0, min(1.0, score))
