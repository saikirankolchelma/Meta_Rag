"""Three chunking strategies over token-accurate boundaries (tiktoken cl100k_base).

- chunk_naive: fixed-size overlapping chunks, used by both the Naive and HyDE
  collections (HyDE differs at retrieval time, not at chunking time).
- chunk_parent_child: small child chunks (used for embedding/retrieval) each
  mapped back to their larger parent chunk (used as generation context).
"""

import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")

NAIVE_CHUNK_SIZE = 512
NAIVE_OVERLAP = 50
CHILD_CHUNK_SIZE = 150
PARENT_CHUNK_SIZE = 1000
PARENT_CHILD_OVERLAP = 20


def _chunk_tokens(tokens: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
    if not tokens:
        return []
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(tokens):
        chunk = tokens[start : start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
        start += step
    return chunks


def chunk_naive(text: str, chunk_size: int = NAIVE_CHUNK_SIZE, overlap: int = NAIVE_OVERLAP) -> list[str]:
    tokens = ENCODING.encode(text)
    return [ENCODING.decode(c) for c in _chunk_tokens(tokens, chunk_size, overlap)]


def chunk_parent_child(
    text: str,
    child_size: int = CHILD_CHUNK_SIZE,
    parent_size: int = PARENT_CHUNK_SIZE,
    overlap: int = PARENT_CHILD_OVERLAP,
) -> list[dict]:
    tokens = ENCODING.encode(text)
    results = []
    for parent_tokens in _chunk_tokens(tokens, parent_size, overlap):
        parent_text = ENCODING.decode(parent_tokens)
        for child_tokens in _chunk_tokens(parent_tokens, child_size, overlap):
            results.append({"child_text": ENCODING.decode(child_tokens), "parent_text": parent_text})
    return results
