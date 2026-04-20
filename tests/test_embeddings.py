import math
import pytest
from tarantula.embeddings import pack, unpack, cosine


def test_pack_unpack_roundtrip():
    vec = [0.1, -0.2, 0.3, 0.4, -0.5]
    blob = pack(vec)
    assert isinstance(blob, bytes)
    assert len(blob) == 4 * len(vec)  # float32 = 4 bytes
    out = unpack(blob)
    assert len(out) == len(vec)
    for a, b in zip(vec, out):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def test_cosine_identical_is_one():
    v = [1.0, 2.0, 3.0]
    assert math.isclose(cosine(v, v), 1.0, abs_tol=1e-6)


def test_cosine_orthogonal_is_zero():
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-6)


def test_cosine_opposite_is_minus_one():
    assert math.isclose(cosine([1.0, 0.0], [-1.0, 0.0]), -1.0, abs_tol=1e-6)


def test_cosine_zero_vector_returns_zero():
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


def test_pack_rejects_empty():
    with pytest.raises(ValueError):
        pack([])
