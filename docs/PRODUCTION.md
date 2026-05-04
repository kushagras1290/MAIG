# Production Checklist

## Security

- Replace `API_TOKEN` before deployment.
- Terminate TLS at an ingress or load balancer.
- Add per-user rate limits before public exposure.
- Keep model API keys in secret stores, not plain env files.

## Reliability

- Run multiple gateway replicas.
- Use a dedicated inference backend for GPU model serving.
- Add request IDs and distributed tracing.
- Add queue bounds for batching.

## Observability

- Scrape `/metrics` with Prometheus.
- Track p50/p95/p99 latency.
- Track provider errors and fallback counts.
- Alert on elevated 5xx and queue wait time.
