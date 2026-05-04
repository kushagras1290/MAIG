from mojo_ai_gateway.config import Settings
from mojo_ai_gateway.providers.base import InferenceProvider
from mojo_ai_gateway.providers.mock import MockProvider
from mojo_ai_gateway.providers.nvidia_hf import NvidiaHFProvider
from mojo_ai_gateway.providers.openai_compatible import OpenAICompatibleProvider


def build_provider(settings: Settings) -> InferenceProvider:
    if settings.ai_provider == "mock":
        return MockProvider()
    if settings.ai_provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    if settings.ai_provider == "nvidia_hf":
        return NvidiaHFProvider(settings)
    raise ValueError(f"Unsupported provider: {settings.ai_provider}")
