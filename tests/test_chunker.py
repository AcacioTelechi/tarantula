from tarantula.chunker import chunk_text, count_tokens


def test_count_tokens_nonempty():
    assert count_tokens("hello world") > 0


def test_short_text_single_chunk():
    chunks = list(chunk_text("Hello world.", target_tokens=2000, overlap_tokens=200))
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].text == "Hello world."


def test_long_text_multiple_chunks_ordered():
    body = "\n\n".join(f"Paragraph {i}." for i in range(500))
    chunks = list(chunk_text(body, target_tokens=200, overlap_tokens=20))
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.ordinal == i
        assert c.text.strip()


def test_chunks_overlap_on_content():
    body = "\n\n".join(f"Para{i}" for i in range(100))
    chunks = list(chunk_text(body, target_tokens=40, overlap_tokens=10))
    if len(chunks) >= 2:
        a = set(chunks[0].text.split())
        b = set(chunks[1].text.split())
        assert a & b


def test_chunks_snap_to_paragraph_boundaries():
    body = "Para A.\n\nPara B.\n\nPara C."
    chunks = list(chunk_text(body, target_tokens=3, overlap_tokens=0))
    for c in chunks:
        assert c.text == c.text.strip()


def test_oversized_single_paragraph_is_split_under_max():
    # A single paragraph (no blank-line separators) far larger than the
    # target must still be split so no chunk exceeds max_tokens. Otherwise
    # the embedding API rejects the batch with a 400 (input too long).
    body = " ".join(f"word{i}" for i in range(5000))
    assert "\n\n" not in body
    max_tokens = 500
    chunks = list(
        chunk_text(body, target_tokens=200, overlap_tokens=20, max_tokens=max_tokens)
    )
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= max_tokens
        assert c.token_count <= max_tokens


def test_oversized_single_line_paragraph_is_split_under_max():
    # No whitespace boundaries to split on at all (e.g. a minified blob):
    # must fall back to hard token slicing.
    body = "x" * 40000
    assert "\n" not in body and " " not in body
    max_tokens = 300
    chunks = list(chunk_text(body, target_tokens=200, max_tokens=max_tokens))
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= max_tokens


def test_many_line_paragraph_stays_under_max():
    # One paragraph (no blank-line separators) made of many short lines.
    # The sum of per-line token counts underestimates the joined text's true
    # token count (newline/merge tokens), so a naive per-line budget can emit
    # a piece that is actually over the limit. Every chunk must measure under
    # max_tokens by the REAL tokenizer count, not a per-line sum.
    body = "\n".join(f"line number {i} of the document" for i in range(2000))
    assert "\n\n" not in body
    max_tokens = 500
    chunks = list(chunk_text(body, target_tokens=400, max_tokens=max_tokens))
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= max_tokens


def test_default_max_tokens_respects_embedding_limit():
    # With defaults, a huge paragraph must never produce a chunk above the
    # 8192-token embedding input limit.
    body = " ".join(f"token{i}" for i in range(60000))
    chunks = list(chunk_text(body))
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= 8192
