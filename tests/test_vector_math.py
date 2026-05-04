import pytest

from mojo_ai_gateway.vector_math import cosine_similarity, dot, top_k_cosine


def test_dot() -> None:
    assert dot([1, 2, 3], [4, 5, 6]) == 32


def test_dot_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        dot([1], [1, 2])


def test_cosine_identity() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_top_k() -> None:
    result = top_k_cosine([1, 0], [[1, 0], [0, 1], [-1, 0]], 2)
    assert result[0][0] == 0
    assert len(result) == 2
