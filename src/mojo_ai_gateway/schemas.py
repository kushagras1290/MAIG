from pydantic import BaseModel, Field, model_validator


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class EmbeddingResponse(BaseModel):
    model: str
    embeddings: list[list[float]]


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=512)
    top_k: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_top_k(self) -> "RerankRequest":
        if self.top_k is not None and self.top_k > len(self.documents):
            raise ValueError("top_k cannot exceed document count")
        return self


class RankedDocument(BaseModel):
    index: int
    document: str
    score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RankedDocument]


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=2048)
    top_k: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_top_k(self) -> "SemanticSearchRequest":
        if self.top_k > len(self.documents):
            raise ValueError("top_k cannot exceed document count")
        return self


class HealthResponse(BaseModel):
    status: str
    provider: str
