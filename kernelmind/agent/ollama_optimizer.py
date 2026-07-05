"""
Ollama-backed OptimizationAgent for KernelMind.

This module is a drop-in replacement for ``kernelmind/agent/llm_agent.py``.
Every public method has an identical signature and return type so that all
call sites (examples, tests, interactive menu) continue to work without
modification.

Key differences from the Claude implementation:
- HTTP calls go to the local Ollama REST API instead of Anthropic's cloud.
- Retry logic handles transient connection failures with exponential back-off.
- If Ollama is unreachable the agent degrades gracefully to heuristic
  suggestions — the same fallback used when the Claude API raised exceptions.
- First-run latency (~15 s) is expected while the model loads into VRAM;
  subsequent calls are much faster.
"""

import json
import logging
import time
from typing import Dict, List, Optional

import requests  # type: ignore[import]

from ..core.graph import ComputationalGraph, Node
from ..core.constants import OpType
from .ollama_config import OllamaConfig, ollama_config
from ..utils.logger import get_logger
from .ollama_errors import (
    OllamaConnectionError,
    OllamaInferenceError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)

logger = get_logger(__name__)


class OptimizationAgent:
    """LLM-powered optimization agent backed by a local Ollama instance.

    Uses NVIDIA Nemotron 3 (or any Ollama-compatible model) to analyse a
    ``ComputationalGraph`` and suggest operator-level optimizations.

    This class exposes exactly the same public interface as the original
    Anthropic-backed ``OptimizationAgent``:

    - :py:meth:`optimize`
    - :py:meth:`refine_optimization`
    - :py:meth:`generate_kernel_code`
    - :py:meth:`get_suggestions`
    - :py:meth:`reset_conversation`

    Args:
        ollama_cfg: Optional :class:`~kernelmind.config.ollama_config.OllamaConfig`
            instance.  Defaults to the module-level singleton which reads from
            environment variables.
    """

    # System prompt is identical to the Claude version so that prompt
    # engineering differences do not affect output quality comparisons.
    _SYSTEM_PROMPT: str = (
        "You are an expert ML systems compiler specializing in neural network "
        "optimization.\nAnalyze the computational graph and provide specific, "
        "actionable optimization recommendations.\nFocus on: operator fusion, "
        "kernel optimization, memory efficiency, and latency reduction.\n"
        "Format responses as JSON with a 'suggestions' array containing "
        "optimization strategies."
    )

    def __init__(self, ollama_cfg: Optional[OllamaConfig] = None) -> None:
        self._cfg: OllamaConfig = ollama_cfg or ollama_config
        # Conversation history mirrors the Claude implementation so that
        # multi-turn refinement (refine_optimization) works correctly.
        self.conversation_history: List[Dict[str, str]] = []
        self.optimization_suggestions: List[Dict] = []
        logger.info(
            "OptimizationAgent initialised — model: %s, host: %s:%d",
            self._cfg.model,
            self._cfg.host,
            self._cfg.port,
        )

    # ------------------------------------------------------------------
    # Public API (identical signatures to the Claude implementation)
    # ------------------------------------------------------------------

    def optimize(self, graph: ComputationalGraph) -> ComputationalGraph:
        """Analyse *graph* and populate :attr:`optimization_suggestions`.

        Args:
            graph: Parsed and pre-optimized :class:`ComputationalGraph`.

        Returns:
            The same *graph* object (suggestions are stored on the agent).
        """
        logger.info("Starting LLM-based optimization via Ollama")

        graph_analysis = self._analyze_graph(graph)
        logger.debug("Graph analysis: %s", graph_analysis)

        suggestions = self._get_optimization_suggestions(graph_analysis)
        logger.info("Generated %d optimization suggestion(s)", len(suggestions))

        self.optimization_suggestions = suggestions
        return graph

    def refine_optimization(self, feedback: str) -> List[Dict]:
        """Ask the model to refine the previous suggestions given *feedback*.

        Args:
            feedback: Free-text feedback from the user or automated evaluation.

        Returns:
            Updated list of suggestion dicts in the same format as
            :py:meth:`optimize`.
        """
        logger.info("Refining optimization based on user feedback")

        refinement_prompt = (
            f'The user provided this feedback on the optimization suggestions:\n\n'
            f'"{feedback}"\n\n'
            "Based on this feedback, provide refined optimization recommendations "
            "that address the user's concerns.\n"
            "Format as JSON with the same structure as before."
        )

        try:
            self.conversation_history.append(
                {"role": "user", "content": refinement_prompt}
            )
            assistant_message = self._call_ollama(self._build_full_prompt())
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )
            refined_suggestions = self._parse_suggestions(assistant_message)
            self.optimization_suggestions = refined_suggestions
            return refined_suggestions

        except Exception as exc:
            logger.error("Refinement failed: %s", exc)
            return self.optimization_suggestions

    def get_kernel_generation_prompt(self, node: Node) -> str:
        """Build a kernel-generation prompt for *node*.

        Args:
            node: Graph node describing the operation to generate a kernel for.

        Returns:
            Prompt string ready to send to an LLM.
        """
        return (
            f"Generate an optimized kernel implementation for this operation:\n\n"
            f"Operation Type: {node.operation.op_type.name}\n"
            f"Inputs: {node.inputs}\n"
            f"Outputs: {node.outputs}\n"
            f"Attributes: {json.dumps(node.operation.attributes, default=str)}\n\n"
            "Requirements:\n"
            "1. Optimize for Apple Silicon (Metal) or NVIDIA CUDA\n"
            "2. Use efficient memory access patterns\n"
            "3. Minimize register usage\n"
            "4. Include performance-critical optimizations\n"
            "5. Add detailed comments\n\n"
            "Format the response as compilable kernel code."
        )

    def generate_kernel_code(self, node: Node, backend: str = "metal") -> str:
        """Ask the model to generate optimized kernel source for *node*.

        Args:
            node:    Graph node to generate a kernel for.
            backend: Target backend — ``"metal"`` or ``"triton"``.

        Returns:
            Raw kernel source code string, or ``""`` on failure.
        """
        logger.info("Generating kernel code for '%s' via Ollama", node.name)

        prompt = self.get_kernel_generation_prompt(node)
        prompt += f"\nTarget backend: {backend}"

        try:
            return self._call_ollama(
                prompt,
                temperature=0.5,  # Lower temperature for deterministic code
            )
        except Exception as exc:
            logger.error("Kernel generation failed: %s", exc)
            return ""

    def get_suggestions(self) -> List[Dict]:
        """Return the most recent list of optimization suggestions.

        Returns:
            List of suggestion dicts produced by the last :py:meth:`optimize`
            or :py:meth:`refine_optimization` call.
        """
        return self.optimization_suggestions

    def reset_conversation(self) -> None:
        """Clear conversation history so the next call starts fresh."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_graph(self, graph: ComputationalGraph) -> Dict:
        """Extract statistics from *graph* into a serialisable dict.

        Args:
            graph: The computational graph to analyse.

        Returns:
            Dict containing node counts, FLOPs, memory size, etc.
        """
        total_nodes = len(graph.nodes)
        total_size = graph.total_size_bytes()
        total_flops = graph.total_flops()

        op_counts: Dict[OpType, int] = {}
        for node in graph.nodes.values():
            op_type = node.operation.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        # Guard: get_critical_path() crashes on empty graphs.
        critical_path = graph.get_critical_path() if total_nodes > 0 else []

        return {
            "num_nodes": total_nodes,
            "total_size_mb": total_size / 1e6,
            "total_flops_gflops": total_flops / 1e9,
            "operation_distribution": {str(k.name): v for k, v in op_counts.items()},
            "critical_path_length": len(critical_path),
            "input_nodes": len(graph.input_nodes),
            "output_nodes": len(graph.output_nodes),
        }

    def _get_optimization_suggestions(self, graph_analysis: Dict) -> List[Dict]:
        """Query Ollama for optimization suggestions given *graph_analysis*.

        Falls back to :py:meth:`_get_default_suggestions` when the model is
        unreachable or returns an unparseable response.

        Args:
            graph_analysis: Dict produced by :py:meth:`_analyze_graph`.

        Returns:
            List of suggestion dicts.
        """
        analysis_prompt = (
            "Analyze this computational graph and provide optimization suggestions:\n\n"
            f"Graph Analysis:\n{json.dumps(graph_analysis, indent=2)}\n\n"
            "Provide 3-5 specific optimization recommendations. "
            "For each recommendation:\n"
            "1. Identify the optimization type (fusion, quantization, memory, compute, etc.)\n"
            "2. Explain the potential benefit\n"
            "3. Estimate the expected speedup percentage\n"
            "4. List any constraints or risks\n\n"
            "Format as JSON:\n"
            "{\n"
            '  "suggestions": [\n'
            "    {\n"
            '      "type": "optimization_type",\n'
            '      "description": "what to optimize",\n'
            '      "benefit": "why this helps",\n'
            '      "estimated_speedup_percent": 15,\n'
            '      "constraints": []\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        try:
            self.conversation_history.append(
                {"role": "user", "content": analysis_prompt}
            )
            full_prompt = self._build_full_prompt()
            assistant_message = self._call_ollama(full_prompt)
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )
            return self._parse_suggestions(assistant_message)

        except Exception as exc:
            logger.error("LLM optimization failed: %s — using default suggestions", exc)
            return self._get_default_suggestions(graph_analysis)

    def _build_full_prompt(self) -> str:
        """Flatten conversation history into a single prompt string for Ollama.

        The Ollama ``/api/generate`` endpoint takes a single ``prompt`` field;
        multi-turn context is prepended manually.

        Returns:
            Single string prompt combining system instructions and history.
        """
        parts: List[str] = [f"System: {self._SYSTEM_PROMPT}\n"]
        for turn in self.conversation_history:
            role = turn["role"].capitalize()
            parts.append(f"{role}: {turn['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _call_ollama(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Send *prompt* to Ollama and return the full generated text.

        Implements exponential back-off retry logic (up to
        ``OllamaConfig.num_retries`` attempts).

        Args:
            prompt:      The full prompt string to send.
            temperature: Override the config temperature for this call.

        Returns:
            The model's generated text (whitespace-stripped).

        Raises:
            OllamaConnectionError:    Server not reachable after all retries.
            OllamaModelNotFoundError: Model not downloaded on the server.
            OllamaInferenceError:     Server returned a non-2xx status.
            OllamaTimeoutError:       Request exceeded the configured timeout.
        """
        payload = {
            "model": self._cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._cfg.temperature,
                "num_predict": self._cfg.max_tokens,
            },
        }

        url = self._cfg.get_ollama_url()
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(1, self._cfg.num_retries + 1):
            try:
                logger.info(
                    "Ollama request attempt %d/%d — model: %s",
                    attempt,
                    self._cfg.num_retries,
                    self._cfg.model,
                )
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self._cfg.timeout,
                )

                # Surface model-not-found errors immediately (no point retrying).
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{self._cfg.model}' not found on Ollama server.",
                        how_to_fix=self._cfg.get_pull_command(),
                    )

                if not response.ok:
                    raise OllamaInferenceError(
                        f"Ollama returned HTTP {response.status_code}: {response.text[:200]}",
                        how_to_fix="Check Ollama server logs for details.",
                    )

                data = response.json()
                generated = data.get("response", "").strip()
                logger.info("Ollama response received (%d chars)", len(generated))
                return generated

            except OllamaModelNotFoundError:
                # Non-retryable — re-raise immediately.
                raise

            except requests.exceptions.Timeout as exc:
                last_exc = OllamaTimeoutError(
                    f"Request timed out after {self._cfg.timeout} s (attempt {attempt})",
                    how_to_fix=(
                        "Increase OLLAMA_TIMEOUT or reduce OLLAMA_MAX_TOKENS. "
                        "First-run model loading takes ~15 s."
                    ),
                )
                logger.warning("%s", last_exc)

            except requests.exceptions.ConnectionError as exc:
                last_exc = OllamaConnectionError(
                    f"Cannot connect to Ollama at {url} (attempt {attempt})",
                    how_to_fix=(
                        "Ensure Ollama is running: "
                        "docker run -d -p 11434:11434 ollama/ollama"
                    ),
                )
                logger.warning("%s", last_exc)

            except (OllamaInferenceError, OllamaConnectionError, OllamaTimeoutError) as exc:
                last_exc = exc
                logger.warning("Attempt %d failed: %s", attempt, exc)

            except Exception as exc:
                last_exc = OllamaInferenceError(
                    f"Unexpected error during Ollama call: {exc}",
                    how_to_fix="Check Ollama server logs for details.",
                )
                logger.warning("Attempt %d unexpected error: %s", attempt, exc)

            # Exponential back-off before the next retry.
            if attempt < self._cfg.num_retries:
                delay = self._cfg.retry_delay * (2 ** (attempt - 1))
                logger.info("Retrying in %.1f s…", delay)
                time.sleep(delay)

        raise last_exc

    def _parse_suggestions(self, response_text: str) -> List[Dict]:
        """Extract the ``suggestions`` array from a JSON-formatted response.

        Tolerates preamble/postamble text around the JSON blob.

        Args:
            response_text: Raw text returned by the model.

        Returns:
            List of suggestion dicts, or ``[]`` if parsing fails.
        """
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                return parsed.get("suggestions", [])

        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM suggestions as JSON: %s", exc)

        return []

    def _get_default_suggestions(self, graph_analysis: Dict) -> List[Dict]:
        """Return heuristic suggestions when the LLM is unavailable.

        These are the same defaults used in the original Claude implementation,
        ensuring identical graceful-degradation behaviour.

        Args:
            graph_analysis: Dict produced by :py:meth:`_analyze_graph`.

        Returns:
            List of suggestion dicts.
        """
        suggestions: List[Dict] = []

        if graph_analysis.get("total_flops_gflops", 0) > 1:
            suggestions.append({
                "type": "fusion",
                "description": "Fuse consecutive linear and activation operations",
                "benefit": "Reduces memory bandwidth overhead",
                "estimated_speedup_percent": 10,
                "constraints": ["Limited by kernel size limits"],
            })

        if graph_analysis.get("total_size_mb", 0) > 100:
            suggestions.append({
                "type": "memory_optimization",
                "description": "Optimize tensor memory layout and caching",
                "benefit": "Improves cache locality",
                "estimated_speedup_percent": 5,
                "constraints": [],
            })

        suggestions.append({
            "type": "quantization",
            "description": "Apply INT8 quantization to weights",
            "benefit": "Reduces memory footprint and bandwidth",
            "estimated_speedup_percent": 15,
            "constraints": ["Requires accuracy validation"],
        })

        return suggestions
