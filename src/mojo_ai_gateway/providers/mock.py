import hashlib
import math

from mojo_ai_gateway.providers.base import InferenceProvider
from mojo_ai_gateway.vector_math import cosine_similarity


class MockProvider(InferenceProvider):
    embed_model = "mock-embedding-v1"
    rerank_model = "mock-rerank-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        q = self._embed_text(query)
        return [cosine_similarity(q, self._embed_text(doc)) for doc in documents]

    @staticmethod
    def _embed_text(text: str, dims: int = 64) -> list[float]:
        values: list[float] = []
        seed = text.encode("utf-8")
        for i in range(dims):
            digest = hashlib.sha256(seed + i.to_bytes(2, "big")).digest()
            raw = int.from_bytes(digest[:4], "big") / 2**32
            values.append((raw * 2.0) - 1.0)
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]
