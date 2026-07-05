"""
Ollama connection and inference configuration for KernelMind.

All values can be overridden via environment variables so that no
code changes are required between development and production deployments.

Environment variables (all optional — sensible defaults are provided):
    OLLAMA_HOST         Hostname where Ollama is listening (default: localhost)
    OLLAMA_PORT         Port number                         (default: 11434)
    OLLAMA_MODEL        Model tag to use                    (default: nvidia/nemotron-3)
    OLLAMA_TEMPERATURE  Sampling temperature                (default: 0.7)
    OLLAMA_MAX_TOKENS   Maximum tokens to generate          (default: 2000)
    OLLAMA_TIMEOUT      HTTP timeout in seconds             (default: 30)
    OLLAMA_NUM_RETRIES  Maximum retry attempts              (default: 3)
    OLLAMA_RETRY_DELAY  Base delay between retries (s)      (default: 1.0)
"""

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """All settings required to connect to and query a local Ollama instance.

    Attributes:
        host:        Hostname of the Ollama server.
        port:        Port the Ollama REST API is bound to.
        model:       Ollama model tag (e.g. ``nvidia/nemotron-3``).
        temperature: Sampling temperature — higher values increase creativity.
        max_tokens:  Upper bound on tokens the model may generate per call.
        timeout:     Seconds to wait before declaring an HTTP request timed out.
        num_retries: How many times to retry a failed request before giving up.
        retry_delay: Base delay (seconds) for exponential back-off between retries.
    """

    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("OLLAMA_PORT", "11434")))
    model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "nvidia/nemotron-super")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_MAX_TOKENS", "2000"))
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "30"))
    )
    num_retries: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_NUM_RETRIES", "3"))
    )
    retry_delay: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_RETRY_DELAY", "1.0"))
    )

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def get_ollama_url(self, path: str = "/api/generate") -> str:
        """Return the full URL for the given Ollama API path.

        Args:
            path: API path to append (default ``/api/generate``).

        Returns:
            Fully-qualified URL string, e.g. ``http://localhost:11434/api/generate``.
        """
        return f"http://{self.host}:{self.port}{path}"

    def get_pull_command(self) -> str:
        """Return the ``ollama pull`` command needed to download the configured model.

        Returns:
            Shell command string, e.g. ``ollama pull nvidia/nemotron-3``.
        """
        return f"ollama pull {self.model}"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Check that the Ollama server is reachable and the model is available.

        Makes a lightweight HTTP request to the Ollama tags endpoint; does not
        load the model or run any inference.

        Returns:
            ``True`` if the server responded and the model is listed.

        Raises:
            OllamaConnectionError: Server could not be reached.
            OllamaModelNotFoundError: Server is up but the model is not downloaded.
        """
        # Imported here to avoid a circular-import at module load time.
        from kernelmind.agent.ollama_errors import (
            OllamaConnectionError,
            OllamaModelNotFoundError,
        )

        try:
            import requests  # type: ignore[import]

            tags_url = self.get_ollama_url("/api/tags")
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()

            data = response.json()
            available_models = [m["name"] for m in data.get("models", [])]

            # Normalize: Ollama sometimes appends ":latest" when listing tags.
            model_base = self.model.split(":")[0]
            found = any(
                m.split(":")[0] == model_base for m in available_models
            )

            if not found:
                logger.warning(
                    "Model '%s' not found in Ollama. Run: %s",
                    self.model,
                    self.get_pull_command(),
                )
                raise OllamaModelNotFoundError(
                    f"Model '{self.model}' is not downloaded. "
                    f"Fix: run `{self.get_pull_command()}`"
                )

            logger.info("Ollama validation passed — model '%s' is ready.", self.model)
            return True

        except ImportError as exc:
            raise OllamaConnectionError(
                "The 'requests' library is not installed. "
                "Fix: pip install requests>=2.28.0"
            ) from exc

        except Exception as exc:
            # Re-raise our own errors untouched.
            from kernelmind.agent.ollama_errors import (
                OllamaConnectionError,
                OllamaModelNotFoundError,
            )
            if isinstance(exc, (OllamaConnectionError, OllamaModelNotFoundError)):
                raise

            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.get_ollama_url('/api/tags')}. "
                "Fix: ensure Ollama is running "
                "(e.g. `docker run -d -p 11434:11434 ollama/ollama`)"
            ) from exc


# Module-level singleton — callers import this directly.
ollama_config = OllamaConfig()
