# Architecture

MAIG separates the API control plane from performance-sensitive vector operations.

## Runtime path

```txt
HTTP request
  -> FastAPI validation
  -> bearer token auth
  -> provider router
  -> inference provider
  -> MojoVectorBridge
  -> response schema
```

## Provider model

Providers implement two capabilities:

- `embed(texts) -> list[list[float]]`
- `rerank(query, documents) -> list[float]`

This keeps the API stable while allowing different runtimes underneath.

## Mojo strategy

Mojo is treated as an accelerator, not a single point of failure. The Python vector implementation remains the reference path. Mojo kernels can be enabled incrementally for cosine similarity, top-k selection, and batch scoring.
