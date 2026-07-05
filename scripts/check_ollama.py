#!/usr/bin/env python3
"""
check_ollama.py — Verify that Ollama is running and the model is ready.

Usage::

    python scripts/check_ollama.py
    python scripts/check_ollama.py --host 192.168.1.10 --port 11434

Exit codes:
    0 — everything is ready.
    1 — Ollama is not reachable or the model is not downloaded.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelmind.agent.ollama_config import OllamaConfig
from kernelmind.agent.ollama_errors import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"{YELLOW}⚠{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"{RED}✗{RESET} {msg}")


def check(cfg: OllamaConfig) -> bool:
    """Run all checks against *cfg* and print a summary.

    Args:
        cfg: Ollama configuration to validate.

    Returns:
        ``True`` if all checks pass, ``False`` otherwise.
    """
    print(f"\n{BOLD}KernelMind — Ollama readiness check{RESET}")
    print("─" * 42)
    print(f"  Host  : {cfg.host}:{cfg.port}")
    print(f"  Model : {cfg.model}")
    print("─" * 42)

    all_ok = True

    # ── 1. requests available? ────────────────────────────────────────
    try:
        import requests as _req  # noqa: F401
        _ok("'requests' package installed")
    except ImportError:
        _fail("'requests' package not found")
        print("       Fix: pip install requests>=2.28.0")
        return False  # Cannot do any HTTP checks without requests.

    # ── 2. Ollama reachable? ──────────────────────────────────────────
    import requests

    tags_url = cfg.get_ollama_url("/api/tags")
    try:
        resp = requests.get(tags_url, timeout=5)
        resp.raise_for_status()
        _ok(f"Ollama server reachable at {cfg.host}:{cfg.port}")
    except requests.exceptions.ConnectionError:
        _fail(f"Cannot reach Ollama at {cfg.host}:{cfg.port}")
        print(f"       Fix: docker run -d -p {cfg.port}:{cfg.port} ollama/ollama")
        return False
    except requests.exceptions.Timeout:
        _fail(f"Connection to {cfg.host}:{cfg.port} timed out")
        print("       Fix: check network / firewall settings")
        return False
    except requests.HTTPError as exc:
        _fail(f"Ollama returned HTTP error: {exc}")
        return False

    # ── 3. Model downloaded? ──────────────────────────────────────────
    try:
        data = resp.json()
        available = [m["name"] for m in data.get("models", [])]
        model_base = cfg.model.split(":")[0]
        found = any(m.split(":")[0] == model_base for m in available)

        if found:
            _ok(f"Model '{cfg.model}' is available")
        else:
            _fail(f"Model '{cfg.model}' is NOT downloaded")
            print(f"       Fix: {cfg.get_pull_command()}")
            if available:
                print(f"       Available models: {', '.join(available)}")
            all_ok = False

    except Exception as exc:
        _warn(f"Could not parse model list: {exc}")
        all_ok = False

    # ── 4. Quick smoke-test inference ─────────────────────────────────
    if all_ok:
        print("\n  Running smoke-test inference (may take ~15 s on first run)…")
        payload = {
            "model": cfg.model,
            "prompt": "Reply with the single word: ready",
            "stream": False,
            "options": {"num_predict": 5},
        }
        try:
            r = requests.post(cfg.get_ollama_url(), json=payload, timeout=60)
            r.raise_for_status()
            reply = r.json().get("response", "").strip()
            _ok(f"Smoke-test passed — model replied: '{reply[:60]}'")
        except requests.exceptions.Timeout:
            _warn("Smoke-test timed out (model may still be loading into VRAM)")
            _warn("This is normal on first run. Try again in ~30 s.")
            all_ok = False
        except Exception as exc:
            _warn(f"Smoke-test failed: {exc}")
            all_ok = False

    # ── Summary ───────────────────────────────────────────────────────
    print("─" * 42)
    if all_ok:
        _ok(f"{BOLD}All checks passed — KernelMind is ready to use!{RESET}")
    else:
        _fail("One or more checks failed. Follow the fix instructions above.")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that Ollama is running and the Nemotron 3 model is ready."
    )
    parser.add_argument("--host", default=None, help="Ollama hostname (overrides env)")
    parser.add_argument("--port", type=int, default=None, help="Ollama port (overrides env)")
    parser.add_argument("--model", default=None, help="Model tag (overrides env)")
    args = parser.parse_args()

    cfg = OllamaConfig(
        host=args.host or OllamaConfig().host,
        port=args.port or OllamaConfig().port,
        model=args.model or OllamaConfig().model,
    )

    success = check(cfg)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
