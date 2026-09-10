"""Wrapper around all-MiniLM-L6-v2 (384-dim) for embedding chunk text."""

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()
