from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "mojo-ai-inference-gateway"
    environment: str = "local"
    log_level: str = "INFO"
    api_token: str = Field(default="dev-token-change-me", min_length=8)

    ai_provider: Literal["mock", "openai_compatible", "nvidia_hf"] = "mock"
    device: str = "cpu"
    request_timeout_seconds: float = 30.0
    batch_max_size: int = 16
    batch_max_delay_ms: int = 20

    nvidia_embed_model: str = "nvidia/llama-embed-nemotron-8b"
    nvidia_rerank_model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    nvidia_chat_model: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

    openai_compat_base_url: str = "http://localhost:8001/v1"
    openai_compat_api_key: str = "local-dev-key"
    openai_compat_embed_model: str = "nvidia/llama-embed-nemotron-8b"
    openai_compat_rerank_model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    openai_compat_chat_model: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
