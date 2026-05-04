import logging
import shutil
import subprocess
from pathlib import Path

from mojo_ai_gateway.vector_math import top_k_cosine

logger = logging.getLogger(__name__)


class MojoVectorBridge:
    def __init__(self, kernel_path: Path | None = None) -> None:
        self.kernel_path = kernel_path or Path("mojo_kernels/vector_similarity.mojo")
        self.mojo_bin = shutil.which("mojo")

    @property
    def available(self) -> bool:
        return bool(self.mojo_bin and self.kernel_path.exists())

    def top_k_cosine(self, query: list[float], vectors: list[list[float]], k: int) -> list[tuple[int, float]]:
        if not self.available:
            return top_k_cosine(query, vectors, k)
        try:
            subprocess.run([self.mojo_bin or "mojo", "--version"], check=True, capture_output=True, text=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Mojo unavailable, falling back to Python", extra={"error": str(exc)})
        return top_k_cosine(query, vectors, k)
