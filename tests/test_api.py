from fastapi.testclient import TestClient

from mojo_ai_gateway.main import app

client = TestClient(app)
HEADERS = {"Authorization": "Bearer dev-token-change-me"}


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_auth_required() -> None:
    response = client.post("/v1/embeddings", json={"texts": ["hello"]})
    assert response.status_code == 401


def test_embeddings() -> None:
    response = client.post("/v1/embeddings", headers=HEADERS, json={"texts": ["hello", "world"]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["embeddings"]) == 2


def test_rerank() -> None:
    response = client.post(
        "/v1/rerank",
        headers=HEADERS,
        json={"query": "gemstone", "documents": ["gemstone price", "sports news"], "top_k": 1},
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_semantic_search() -> None:
    response = client.post(
        "/v1/semantic-search",
        headers=HEADERS,
        json={"query": "jupiter stone", "documents": ["yellow sapphire", "ruby"], "top_k": 1},
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
