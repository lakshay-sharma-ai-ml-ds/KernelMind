"""
Custom exception hierarchy for Ollama integration in KernelMind.

All exceptions include a human-readable ``how_to_fix`` hint so that users
encountering an error for the first time can self-serve without reading docs.
"""

import logging

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base class for all Ollama-related exceptions.

    Args:
        message:    Human-readable description of what went wrong.
        how_to_fix: Optional hint on how the user can resolve the issue.
    """

    def __init__(self, message: str, how_to_fix: str = "") -> None:
        self.how_to_fix = how_to_fix
        full_message = message
        if how_to_fix:
            full_message = f"{message}\nHow to fix: {how_to_fix}"
        super().__init__(full_message)
        logger.error(full_message)


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached over HTTP.

    Typical causes:
    - Ollama is not running.
    - The wrong host / port is configured.
    - A firewall is blocking the connection.

    Example::

        raise OllamaConnectionError(
            "Connection refused at localhost:11434",
            how_to_fix="Start Ollama: docker run -d -p 11434:11434 ollama/ollama",
        )
    """


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model has not been pulled to the Ollama server.

    Example::

        raise OllamaModelNotFoundError(
            "Model 'nvidia/nemotron-3' not found",
            how_to_fix="ollama pull nvidia/nemotron-3",
        )
    """


class OllamaInferenceError(OllamaError):
    """Raised when the Ollama server returns an error during inference.

    This covers HTTP 4xx / 5xx responses and malformed JSON payloads
    that indicate the model failed to produce a valid response.

    Example::

        raise OllamaInferenceError(
            "Ollama returned HTTP 500",
            how_to_fix="Check Ollama logs: docker logs ollama",
        )
    """


class OllamaTimeoutError(OllamaError):
    """Raised when an inference request exceeds the configured timeout.

    Typically caused by:
    - The model being too large for available hardware.
    - The host machine being under heavy load.
    - An excessively long prompt / max_tokens value.

    Example::

        raise OllamaTimeoutError(
            "Request timed out after 30 s",
            how_to_fix=(
                "Increase OLLAMA_TIMEOUT or reduce OLLAMA_MAX_TOKENS. "
                "For the first run, the model needs ~15 s to load into memory."
            ),
        )
    """
