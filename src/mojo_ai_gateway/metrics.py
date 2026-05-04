from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

REQUESTS = Counter("maig_http_requests_total", "Total HTTP requests", ["path", "method", "status"])
LATENCY = Histogram("maig_http_request_duration_seconds", "HTTP request latency", ["path", "method"])
INFERENCE = Histogram("maig_inference_duration_seconds", "Inference latency", ["operation", "provider"])


def metrics_response() -> Response:
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")
