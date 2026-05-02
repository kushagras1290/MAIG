# Mojo AI Inference Gateway

Production-oriented AI inference gateway using a Python control plane and Mojo hot-path kernels.

The project is intentionally designed to run in two modes:

1. **Local/dev mode** with deterministic mock providers, so CI and laptops work without downloading giant models.
2. **NVIDIA model mode** using open NVIDIA Hugging Face models or any OpenAI-compatible local server such as vLLM / MAX / NIM-style endpoints.

## What this builds

- FastAPI inference gateway
- Authenticated API endpoints
- NVIDIA open-model provider layer
- OpenAI-compatible local model server client
- Embedding endpoint
- Rerank endpoint
- In-memory semantic search endpoint
- Dynamic async batching service
- Python vector math fallback
- Mojo kernel bridge with graceful fallback
- Prometheus metrics
- Structured JSON logs
- Docker + Compose
- Kubernetes manifests
- Benchmarks
- Tests
- CI workflow

## Default NVIDIA open models

The defaults are configurable in `.env`:

```env
NVIDIA_EMBED_MODEL=nvidia/llama-embed-nemotron-8b
NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2
NVIDIA_CHAT_MODEL=nvidia/Llama-3.1-Nemotron-Nano-8B-v1
```

The gateway does **not** hardcode closed providers. The OpenAI-compatible client is only a protocol adapter for local/self-hosted servers.

## Architecture

```txt
Client
  -> FastAPI Gateway
      -> Auth + validation + rate hooks
      -> Dynamic batching
      -> Provider router
          -> Mock provider for CI/dev
          -> NVIDIA Hugging Face local provider
          -> OpenAI-compatible local endpoint provider
      -> Mojo bridge for hot-path vector scoring when available
      -> Python fallback when Mojo is unavailable
      -> Prometheus metrics + structured logs
```

## Quick start: dev mode

```bash
cd mojo-ai-inference-gateway
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn mojo_ai_gateway.main:app --reload --host 0.0.0.0 --port 8000
```

Test:

```bash
curl -s http://localhost:8000/healthz | jq
```

Embedding request:

```bash
curl -s http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dev-token-change-me' \
  -d '{"texts":["ruby gemstone", "blue sapphire"]}' | jq
```

Rerank request:

```bash
curl -s http://localhost:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dev-token-change-me' \
  -d '{"query":"yellow sapphire price", "documents":["Pukhraj price depends on origin and clarity", "Football score update"]}' | jq
```

Semantic search request:

```bash
curl -s http://localhost:8000/v1/semantic-search \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dev-token-change-me' \
  -d '{"query":"best pukhraj for astrology", "documents":["Yellow sapphire is used for Jupiter", "Ruby is linked with Sun", "Blue sapphire is linked with Saturn"], "top_k":2}' | jq
```

## Run tests

```bash
pytest -q
ruff check .
mypy src
```

## Run benchmarks

```bash
python benchmarks/bench_vector_similarity.py --vectors 5000 --dims 768
```

If `mojo` is installed and available on `PATH`, the benchmark also attempts the Mojo path. Otherwise it logs a clean skip and uses Python.

## NVIDIA model mode

For real NVIDIA model inference:

```env
AI_PROVIDER=nvidia_hf
DEVICE=cuda
NVIDIA_EMBED_MODEL=nvidia/llama-embed-nemotron-8b
NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2
```

Install model extras:

```bash
pip install -e '.[models]'
```

Then run:

```bash
uvicorn mojo_ai_gateway.main:app --host 0.0.0.0 --port 8000
```

### GPU warning

The default NVIDIA models are serious models, not tiny toy rubbish. Expect GPU memory requirements. Use `AI_PROVIDER=mock` for dev and CI.

## OpenAI-compatible local endpoint mode

Use this mode for vLLM / MAX / NIM-style local servers exposing OpenAI-compatible routes:

```env
AI_PROVIDER=openai_compatible
OPENAI_COMPAT_BASE_URL=http://localhost:8001/v1
OPENAI_COMPAT_API_KEY=local-dev-key
OPENAI_COMPAT_EMBED_MODEL=nvidia/llama-embed-nemotron-8b
OPENAI_COMPAT_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2
OPENAI_COMPAT_CHAT_MODEL=nvidia/Llama-3.1-Nemotron-Nano-8B-v1
```

## Mojo setup

Use Pixi or uv as recommended by Modular docs.

```bash
pixi add mojo
pixi run mojo --version
```

or:

```bash
uv pip install mojo --extra-index-url https://modular.gateway.scarf.sh/simple/
mojo --version
```

The gateway works without Mojo using Python fallback. Mojo is used as an accelerator where available.

## Docker

```bash
docker compose up --build
```

## Production notes

See:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/NVIDIA_MODELS.md`](docs/NVIDIA_MODELS.md)
- [`docs/PRODUCTION.md`](docs/PRODUCTION.md)

## Git push

This generated project is locally committed. To push to your repo:

```bash
git remote add origin git@github.com:<owner>/<repo>.git
git branch -M main
git push -u origin main
```
