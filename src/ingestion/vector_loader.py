"""Chunks every document in data/raw/ with all three strategies, embeds the
chunks, and upserts them into their respective isolated Qdrant collections.

Usage:
    python -m src.ingestion.vector_loader
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingestion.chunker import chunk_naive, chunk_parent_child
from src.ingestion.embedder import EMBEDDING_DIM, embed_texts
from src.utils.config import DATA_RAW_DIR, QDRANT_URL

COLLECTION_NAIVE = "collection_naive"
COLLECTION_PARENT = "collection_parent"
COLLECTION_HYDE = "collection_hyde"
UPSERT_BATCH_SIZE = 128


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collections(client: QdrantClient) -> None:
    for name in (COLLECTION_NAIVE, COLLECTION_PARENT, COLLECTION_HYDE):
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def _upsert(client: QdrantClient, collection: str, texts: list[str], payloads: list[dict]) -> int:
    count = 0
    for i in range(0, len(texts), UPSERT_BATCH_SIZE):
        batch_texts = texts[i : i + UPSERT_BATCH_SIZE]
        batch_payloads = payloads[i : i + UPSERT_BATCH_SIZE]
        vectors = embed_texts(batch_texts)
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
            for vec, payload in zip(vectors, batch_payloads)
        ]
        client.upsert(collection_name=collection, points=points)
        count += len(points)
    return count


def ingest_document(client: QdrantClient, source: str, text: str) -> dict:
    counts = {}

    naive_chunks = chunk_naive(text)
    naive_payloads = [{"text": c, "source": source, "chunk_index": i} for i, c in enumerate(naive_chunks)]
    counts["naive"] = _upsert(client, COLLECTION_NAIVE, naive_chunks, naive_payloads)

    # HyDE differs from Naive at retrieval time (embeds a hypothetical answer,
    # not the raw query), not at chunking time, so it reuses the same chunks.
    hyde_payloads = [{"text": c, "source": source, "chunk_index": i} for i, c in enumerate(naive_chunks)]
    counts["hyde"] = _upsert(client, COLLECTION_HYDE, naive_chunks, hyde_payloads)

    pc = chunk_parent_child(text)
    child_texts = [item["child_text"] for item in pc]
    child_payloads = [
        {"text": item["child_text"], "parent_text": item["parent_text"], "source": source, "chunk_index": i}
        for i, item in enumerate(pc)
    ]
    counts["parent"] = _upsert(client, COLLECTION_PARENT, child_texts, child_payloads)

    return counts


def main():
    client = get_client()
    ensure_collections(client)

    paths = sorted(DATA_RAW_DIR.glob("*.txt"))
    if not paths:
        raise SystemExit(f"No .txt files found in {DATA_RAW_DIR}. Run fetch_wikipedia.py first.")

    totals = {"naive": 0, "hyde": 0, "parent": 0}
    for i, path in enumerate(paths, start=1):
        text = path.read_text(encoding="utf-8")
        counts = ingest_document(client, source=path.stem, text=text)
        for k, v in counts.items():
            totals[k] += v
        print(f"[{i}/{len(paths)}] {path.stem}: naive={counts['naive']} hyde={counts['hyde']} parent={counts['parent']}")

    print(f"Done. Totals: {totals}")


if __name__ == "__main__":
    main()
