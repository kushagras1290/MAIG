from abc import ABC, abstractmethod


class InferenceProvider(ABC):
    embed_model: str
    rerank_model: str

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        raise NotImplementedError
