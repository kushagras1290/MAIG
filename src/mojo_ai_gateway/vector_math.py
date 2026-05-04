from math import sqrt


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    denom = sqrt(dot(a, a)) * sqrt(dot(b, b))
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def top_k_cosine(query: list[float], vectors: list[list[float]], k: int) -> list[tuple[int, float]]:
    if k < 1:
        raise ValueError("k must be positive")
    scored = [(idx, cosine_similarity(query, vector)) for idx, vector in enumerate(vectors)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
