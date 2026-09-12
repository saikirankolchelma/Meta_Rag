from src.ingestion.chunker import ENCODING, chunk_naive, chunk_parent_child


def _token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def test_chunk_naive_empty_text_returns_no_chunks():
    assert chunk_naive("") == []


def test_chunk_naive_short_text_is_a_single_chunk():
    text = "This is a short sentence about nothing in particular."
    chunks = chunk_naive(text, chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert ENCODING.decode(ENCODING.encode(text)) == chunks[0]


def test_chunk_naive_respects_chunk_size():
    text = " ".join(f"token{i}" for i in range(2000))
    chunks = chunk_naive(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert _token_count(chunk) <= 100


def test_chunk_naive_overlap_repeats_trailing_tokens():
    text = " ".join(f"token{i}" for i in range(2000))
    chunks = chunk_naive(text, chunk_size=100, overlap=20)

    first_tokens = ENCODING.encode(chunks[0])
    second_tokens = ENCODING.encode(chunks[1])
    assert second_tokens[:20] == first_tokens[-20:]


def test_chunk_naive_covers_the_full_text_with_no_gaps():
    text = " ".join(f"token{i}" for i in range(500))
    full_tokens = ENCODING.encode(text)
    chunks = chunk_naive(text, chunk_size=100, overlap=20)

    last_tokens = ENCODING.encode(chunks[-1])
    assert ENCODING.decode(full_tokens[-len(last_tokens):]) == chunks[-1]


def test_chunk_parent_child_empty_text_returns_nothing():
    assert chunk_parent_child("") == []


def test_chunk_parent_child_respects_both_sizes():
    text = " ".join(f"token{i}" for i in range(3000))
    results = chunk_parent_child(text, child_size=50, parent_size=300, overlap=10)

    assert len(results) > 0
    for item in results:
        assert set(item.keys()) == {"child_text", "parent_text"}
        assert _token_count(item["child_text"]) <= 50
        assert _token_count(item["parent_text"]) <= 300


def test_chunk_parent_child_children_share_their_parent_within_a_group():
    text = " ".join(f"token{i}" for i in range(3000))
    results = chunk_parent_child(text, child_size=50, parent_size=300, overlap=10)

    # Consecutive entries with the same parent_text form one parent's children.
    parents_seen = []
    for item in results:
        if not parents_seen or parents_seen[-1] != item["parent_text"]:
            parents_seen.append(item["parent_text"])
    assert len(parents_seen) == len(set(parents_seen))  # no parent text repeats non-consecutively
