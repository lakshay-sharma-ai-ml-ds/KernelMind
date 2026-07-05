# KERNELMIND: Agentic ML Compiler & GPU Kernel Optimizer

Analyzes PyTorch models, applies compiler optimizations, generates Metal/Triton GPU kernels via a local **NVIDIA Nemotron Super** LLM agent (no API costs), and benchmarks them — fully automated.

**Optimized for:** Apple Silicon M-series (Metal) · NVIDIA GPUs (Triton)

---

## System Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | Tested up to 3.13 |
| OS | macOS 12.3+ | Metal backend; NVIDIA path works on Linux |
| Disk | ~10 GB | Packages, venv, Nemotron Super weights |
| Ollama | any | Local inference — no API key needed |

---

## Quick Start

```bash
cd ~/kernelmind
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt && pip install -e .
bash scripts/setup_ollama.sh   # start Ollama + pull Nemotron Super
python3 quickstart.py && python3 main.py
```

---

## Ollama Setup (No API Key Required)

> NVIDIA Nemotron Super runs locally via [Ollama](https://ollama.com/) — free and private.

**Docker (recommended):**
```bash
docker run -d --name ollama -p 11434:11434 -v ollama-data:/root/.ollama ollama/ollama
docker exec ollama ollama pull nvidia/nemotron-super
python scripts/check_ollama.py
```

**Native (macOS / Linux):**
```bash
brew install ollama && brew services start ollama   # macOS
curl -fsSL https://ollama.com/install.sh | sh       # Linux
ollama pull nvidia/nemotron-super
```

**Apple Silicon:** Run Ollama natively (not Docker) for full Metal GPU acceleration.

**Key env vars** (copy `.env.example` → `.env` to override):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `nvidia/nemotron-super` | Model tag |
| `OLLAMA_HOST` | `localhost` | Server hostname |
| `OLLAMA_PORT` | `11434` | Server port |
| `OLLAMA_TEMPERATURE` | `0.7` | Sampling temperature |
| `OLLAMA_TIMEOUT` | `30` | HTTP timeout (s) |

---

## Project Structure

```
kernelmind/
├── main.py · quickstart.py · scripts/
└── kernelmind/
    ├── config.py · agent/ · core/ · kernels/ · benchmarks/ · examples/
```

---

## Basic Usage

```python
import torch, torch.nn as nn
from kernelmind import ModelParser, GraphOptimizer, OptimizationAgent
from kernelmind.kernels import KernelGenerator, MetalBackend

model  = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
sample = torch.randn(1, 784)

graph     = ModelParser().parse_model(model, sample)
opt_graph = GraphOptimizer(graph).optimize()
agent     = OptimizationAgent()   # Nemotron Super — no API key
agent.optimize(opt_graph)

kernels = KernelGenerator().generate_kernels(opt_graph, backend="metal")
for name, code in kernels.items():
    if MetalBackend().compile_kernel(name, code):
        print(f"✓ {name}")
```

**Set optimization level:**
```python
from kernelmind.config import config, OptimizationLevel
config.OPTIMIZATION_LEVEL = OptimizationLevel.HIGH
```

---

## Running Examples

```bash
python3 main.py                           # interactive menu (5 options)
python3 examples/simple_linear.py         # MLP — start here
python3 examples/resnet_optimize.py       # CNN conv+BN fusion
python3 examples/transformer_fusion.py    # QKV + attention fusion
```

---

## Configuration

```bash
export KERNELMIND_OPTIMIZATION_LEVEL=AGGRESSIVE
export OLLAMA_MODEL=nvidia/nemotron-super
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: kernelmind` | `pip install -e .` |
| `Connection refused at :11434` | `bash scripts/setup_ollama.sh` |
| `Model not found` | `docker exec ollama ollama pull nvidia/nemotron-super` |
| Timeout on first run | Model loading into VRAM — set `OLLAMA_TIMEOUT=60` |
| Metal/MPS not available | Apple Silicon + macOS 12.3+ · `pip install --upgrade torch` |
| Out of memory | Lower `BENCHMARK_BATCH_SIZES` in `config.py` |

```bash
tail -f kernelmind.log    # live logs
```

---

## Benchmark Results (Apple M5 · MPS · macOS)

### MLP (784 → 256 → 10)
| Batch | Latency (ms) | Throughput |
|---|---|---|
| 1 | 0.65 | 1,537 samples/s |
| 8 | 0.61 | 13,051 samples/s |
| 16 | 0.67 | 23,946 samples/s |

### ResNet-18 (224 × 224)
| Batch | Latency (ms) | Throughput |
|---|---|---|
| 1 | 3.4 | 292 samples/s |
| 4 | 7.1 | 567 samples/s |
| 32 | 88.9 | 360 samples/s |

### Transformer (d=512, seq=128)
| Batch | Latency (ms) | Throughput |
|---|---|---|
| 1 | 11.4 | 87 tok/s |
| 8 | 30.5 | 262 tok/s |
| 16 | 67.7 | 237 tok/s |

### Optimization Decisions (Nemotron Super fallback)
| Type | Expected Speedup | Risk | Decision |
|---|---|---|---|
| Quantization (INT8) | 15% | High | ✅ Approved |

> **Note:** LLM column shows heuristic fallback results. Pull `nvidia/nemotron-super` for live Nemotron-guided decisions.
