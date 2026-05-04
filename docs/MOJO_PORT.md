# MAIG Mojo Port Plan

## Status

This repository should not be mechanically converted from Python to Mojo file-by-file.

The production-safe boundary is:

- Python owns the API/control plane.
- Mojo owns hot-path compute kernels.
- Python fallback remains available when Mojo is unavailable.

A blanket `.py` to `.mojo` rewrite would break the current application because core runtime surfaces are Python-native:

- FastAPI route registration and ASGI middleware
- Pydantic request/response validation
- Pydantic Settings environment loading
- HTTPX async clients
- Prometheus client metrics
- Pytest/FastAPI test client
- Hugging Face / sentence-transformers provider loading

## Implemented in this branch

`mojo_kernels/vector_similarity.mojo` now contains native Mojo kernels for:

- dot product
- vector norm
- cosine similarity
- cosine score materialization for a query against many vectors
- argmax over scores

The Python gateway remains unchanged so existing API tests, Docker, CI, and provider integrations do not break.

## Files that should stay Python for now

| Area | Reason |
|---|---|
| `src/mojo_ai_gateway/main.py` | FastAPI/ASGI runtime and decorators are Python framework surfaces. |
| `src/mojo_ai_gateway/schemas.py` | Pydantic validation should stay Python to preserve API semantics. |
| `src/mojo_ai_gateway/config.py` | Pydantic Settings handles `.env` and deployment config safely. |
| `src/mojo_ai_gateway/providers/*.py` | HTTPX, Hugging Face, sentence-transformers, and provider I/O are Python ecosystem boundaries. |
| `tests/*.py` | Existing API and fallback behavior should remain tested through Python. |

## Files that are valid Mojo migration targets

| File | Migration action |
|---|---|
| `mojo_kernels/vector_similarity.mojo` | Native Mojo implementation. |
| `src/mojo_ai_gateway/vector_math.py` | Keep as fallback and reference implementation. |
| `src/mojo_ai_gateway/mojo_bridge.py` | Extend later to import/call Mojo bindings and fall back safely. |
| `benchmarks/bench_vector_similarity.py` | Extend later to compare Python fallback vs Mojo path. |

## Recommended next implementation step

Expose the Mojo kernels to Python through a stable local toolchain contract once the Modular/Mojo version is pinned.

The intended Python-side import boundary should become:

```python
try:
    from mojo_ai_gateway.vector_math_mojo import top_k_cosine
except Exception:
    from mojo_ai_gateway.vector_math import top_k_cosine
```

This keeps production availability intact while allowing Mojo acceleration where installed.

## Verification commands

Run after checking out this branch locally:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest -q
ruff check .
mypy src
```

When Mojo is installed:

```bash
mojo --version
mojo mojo_kernels/vector_similarity.mojo
```

## Security notes

- No secrets were added.
- No authentication behavior was changed.
- No network provider behavior was changed.
- The Python fallback remains intact.
- The Mojo file only performs numerical vector operations over caller-provided buffers.

## Known limitations

- This branch does not yet expose the Mojo kernel as a Python extension module.
- The Mojo source was updated through repository inspection; execution could not be run from this remote GitHub tool context.
- The correct next step is local toolchain verification and binding integration, not blanket file renaming.

## Why not convert everything?

Because that would be a brittle rewrite with lower reliability. FastAPI, Pydantic, HTTPX, Prometheus, pytest, and Hugging Face are Python-native system boundaries. Replacing them with Mojo today would create risk without performance benefit.

The useful acceleration point is vector similarity, not HTTP plumbing.
