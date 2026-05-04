#!/usr/bin/env bash
set -euo pipefail
uvicorn mojo_ai_gateway.main:app --reload --host 0.0.0.0 --port 8000
