# NVIDIA Models

Default configurable models:

| Purpose | Model |
|---|---|
| Embeddings | `nvidia/llama-embed-nemotron-8b` |
| Reranking | `nvidia/llama-nemotron-rerank-1b-v2` |
| Chat / generation | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` |

## Recommended deployment patterns

1. Use `mock` for CI and local API work.
2. Use `openai_compatible` for local GPU servers exposing OpenAI-compatible endpoints.
3. Use `nvidia_hf` only when you explicitly want in-process Transformers loading.

In-process model loading is convenient but can be poor for production isolation. Dedicated inference servers are cleaner.
