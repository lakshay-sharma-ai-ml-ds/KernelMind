#!/usr/bin/env bash
# setup_ollama.sh — One-command Ollama + Nemotron 3 setup for KernelMind
#
# Usage:
#   bash scripts/setup_ollama.sh
#   bash scripts/setup_ollama.sh --model nvidia/nemotron-3 --port 11434
#
# What this script does:
#   1. Check Docker is available
#   2. Start (or restart) an Ollama Docker container
#   3. Pull the Nemotron 3 model into the container
#   4. Verify Ollama is healthy
#   5. Run the Python readiness check

set -euo pipefail

# ── Defaults (override with --model / --port / --container) ──────────────────
OLLAMA_MODEL="${OLLAMA_MODEL:-nvidia/nemotron-super}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
CONTAINER_NAME="ollama-kernelmind"
OLLAMA_IMAGE="ollama/ollama"

GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
BOLD="\033[1m"
RESET="\033[0m"

ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }
fail() { echo -e "${RED}✗${RESET} $*"; exit 1; }
step() { echo -e "\n${BOLD}▶ $*${RESET}"; }

# ── Parse optional arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     OLLAMA_MODEL="$2";    shift 2 ;;
        --port)      OLLAMA_PORT="$2";     shift 2 ;;
        --container) CONTAINER_NAME="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo -e "\n${BOLD}KernelMind — Ollama Setup Script${RESET}"
echo "────────────────────────────────────────"
echo "  Model     : $OLLAMA_MODEL"
echo "  Port      : $OLLAMA_PORT"
echo "  Container : $CONTAINER_NAME"
echo "────────────────────────────────────────"

# ── Step 1: Check Docker ──────────────────────────────────────────────────────
step "Checking Docker"
if ! command -v docker &>/dev/null; then
    fail "Docker is not installed. Install it from https://docs.docker.com/get-docker/"
fi
if ! docker info &>/dev/null; then
    fail "Docker daemon is not running. Start Docker Desktop or the Docker service."
fi
ok "Docker is available"

# ── Step 2: Start or reuse Ollama container ───────────────────────────────────
step "Starting Ollama container"
if docker ps -q --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
    ok "Container '${CONTAINER_NAME}' is already running"
elif docker ps -aq --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
    warn "Container '${CONTAINER_NAME}' exists but is stopped — starting it"
    docker start "${CONTAINER_NAME}"
    ok "Container started"
else
    echo "  Pulling ${OLLAMA_IMAGE} and starting container…"
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p "${OLLAMA_PORT}:11434" \
        -v ollama-kernelmind-data:/root/.ollama \
        "${OLLAMA_IMAGE}"
    ok "Container '${CONTAINER_NAME}' created and started"
fi

# ── Step 3: Wait for Ollama to be healthy ─────────────────────────────────────
step "Waiting for Ollama to be ready"
MAX_WAIT=30
ELAPSED=0
until curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" &>/dev/null; do
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        fail "Ollama did not become ready within ${MAX_WAIT} s. Check: docker logs ${CONTAINER_NAME}"
    fi
    echo -n "."
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
echo
ok "Ollama API is responding on port ${OLLAMA_PORT}"

# ── Step 4: Pull the model ────────────────────────────────────────────────────
step "Pulling model '${OLLAMA_MODEL}' (this may take several minutes)"
docker exec "${CONTAINER_NAME}" ollama pull "${OLLAMA_MODEL}"
ok "Model '${OLLAMA_MODEL}' downloaded"

# ── Step 5: Verify with the Python check script ───────────────────────────────
step "Running Python readiness check"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

if python3 "${SCRIPT_DIR}/check_ollama.py" \
       --model "${OLLAMA_MODEL}" \
       --port  "${OLLAMA_PORT}"; then
    echo
    echo -e "${GREEN}${BOLD}✓ Setup complete! KernelMind is ready.${RESET}"
    echo
    echo "  Next steps:"
    echo "    pip install -r ${REPO_ROOT}/requirements.txt"
    echo "    pip install -e ${REPO_ROOT}"
    echo "    python3 ${REPO_ROOT}/quickstart.py"
else
    warn "Python readiness check reported issues. Check the output above."
    exit 1
fi
