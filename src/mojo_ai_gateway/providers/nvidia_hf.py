from functools import cached_property

from mojo_ai_gateway.config import Settings
from mojo_ai_gateway.providers.base import InferenceProvider
from mojo_ai_gateway.vector_math import cosine_similarity


class NvidiaHFProvider(InferenceProvider):
    def __init__(self, settings: Settings) -> None:
        self.embed_model = settings.nvidia_embed_model
        self.rerank_model = settings.nvidia_rerank_model
        self.device = settings.device

    @cached_property
    def _sentence_transformer(self):  # type: ignore[no-untyped-def]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install model extras with: pip install -e '.[models]'") from exc
        return SentenceTransformer(self.embed_model, device=self.device)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._sentence_transformer.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        embeddings = await self.embed([query, *documents])
        return [cosine_similarity(embeddings[0], doc) for doc in embeddings[1:]]
