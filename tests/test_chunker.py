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
