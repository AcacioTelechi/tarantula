from __future__ import annotations

import numpy as np


def pack(vec: list[float]) -> bytes:
    """Serialize a float vector to float32 bytes for SQLite BLOB storage."""
    if not vec:
        raise ValueError("pack: empty vector")
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> list[float]:
    """Inverse of pack. Returns a Python list of floats."""
    return np.frombuffer(blob, dtype=np.float32).tolist()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)
