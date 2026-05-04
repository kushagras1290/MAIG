import time
from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, Request, Response

from mojo_ai_gateway.config import get_settings
from mojo_ai_gateway.logging import configure_logging
from mojo_ai_gateway.metrics import INFERENCE, LATENCY, REQUESTS, metrics_response
from mojo_ai_gateway.mojo_bridge import MojoVectorBridge
from mojo_ai_gateway.providers.factory import build_provider
from mojo_ai_gateway.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    RankedDocument,
    RerankRequest,
    RerankResponse,
    SemanticSearchRequest,
)
from mojo_ai_gateway.security import require_bearer_token

settings = get_settings()
configure_logging(settings.log_level)
provider = build_provider(settings)
vector_bridge = MojoVectorBridge()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    LATENCY.labels(path=path, method=request.method).observe(elapsed)
    REQUESTS.labels(path=path, method=request.method, status=str(response.status_code)).inc()
    return response


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", provider=settings.ai_provider)


@app.get("/metrics")
async def metrics() -> Response:
    return metrics_response()


@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(require_bearer_token)])
async def embeddings(payload: EmbeddingRequest) -> EmbeddingResponse:
    with INFERENCE.labels(operation="embed", provider=settings.ai_provider).time():
        vectors = await provider.embed(payload.texts)
    return EmbeddingResponse(model=provider.embed_model, embeddings=vectors)


@app.post("/v1/rerank", response_model=RerankResponse, dependencies=[Depends(require_bearer_token)])
async def rerank(payload: RerankRequest) -> RerankResponse:
    with INFERENCE.labels(operation="rerank", provider=settings.ai_provider).time():
        scores = await provider.rerank(payload.query, payload.documents)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    if payload.top_k is not None:
        ranked = ranked[: payload.top_k]
    return RerankResponse(
        model=provider.rerank_model,
        results=[RankedDocument(index=i, document=payload.documents[i], score=float(score)) for i, score in ranked],
    )


@app.post("/v1/semantic-search", response_model=RerankResponse, dependencies=[Depends(require_bearer_token)])
async def semantic_search(payload: SemanticSearchRequest) -> RerankResponse:
    with INFERENCE.labels(operation="semantic_search", provider=settings.ai_provider).time():
        vectors = await provider.embed([payload.query, *payload.documents])
        top = vector_bridge.top_k_cosine(vectors[0], vectors[1:], payload.top_k)
    return RerankResponse(
        model=provider.embed_model,
        results=[RankedDocument(index=i, document=payload.documents[i], score=float(score)) for i, score in top],
    )
