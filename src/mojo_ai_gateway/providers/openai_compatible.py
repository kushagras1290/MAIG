import httpx

from mojo_ai_gateway.config import Settings
from mojo_ai_gateway.providers.base import InferenceProvider
from mojo_ai_gateway.vector_math import cosine_similarity


class OpenAICompatibleProvider(InferenceProvider):
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.openai_compat_base_url.rstrip("/")
        self.api_key = settings.openai_compat_api_key
        self.embed_model = settings.openai_compat_embed_model
        self.rerank_model = settings.openai_compat_rerank_model
        self.timeout = settings.request_timeout_seconds

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.embed_model, "input": texts},
            )
            response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.rerank_model, "query": query, "documents": documents},
            )
            if response.status_code < 400:
                payload = response.json()
                return [float(item["score"]) for item in payload.get("results", [])]
        embeddings = await self.embed([query, *documents])
        return [cosine_similarity(embeddings[0], doc) for doc in embeddings[1:]]
