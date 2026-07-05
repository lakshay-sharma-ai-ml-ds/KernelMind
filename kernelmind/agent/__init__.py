"""
kernelmind.agent — LLM optimization agent sub-package.

Public exports are identical to the original Claude-backed version so that
all existing call sites (examples, tests, main.py) require zero changes.

Import resolution order:
    1. Try OllamaOptimizer (local Nemotron Super via Ollama) — preferred.
    2. If the import fails, re-raise with a clear setup message.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .ollama_optimizer import OptimizationAgent  # noqa: F401
    logger.debug("OptimizationAgent loaded from ollama_optimizer")
except ImportError as _err:
    # Surface a helpful message rather than a raw ImportError stack trace.
    raise ImportError(
        "\nFailed to load the Ollama-backed OptimizationAgent.\n"
        "Make sure the 'requests' package is installed:\n"
        "    pip install requests>=2.28.0\n\n"
        "Then ensure Ollama is running and Nemotron Super is downloaded:\n"
        "    docker run -d -p 11434:11434 ollama/ollama\n"
        "    ollama pull nvidia/nemotron-super\n\n"
        "Or run the one-command setup:\n"
        "    bash scripts/setup_ollama.sh\n\n"
        f"Original error: {_err}"
    ) from _err

from .decision_engine import DecisionEngine  # noqa: F401
from .ollama_errors import (  # noqa: F401
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaInferenceError,
    OllamaTimeoutError,
)

__all__ = [
    "OptimizationAgent",
    "DecisionEngine",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaInferenceError",
    "OllamaTimeoutError",
]
