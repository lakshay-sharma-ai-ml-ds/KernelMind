# KERNELMIND — Project Summary

> What it is, why it was built, how it works, and how every piece fits together.

---

## What Is KERNELMIND?

KERNELMIND is a production-grade agentic ML compiler and GPU kernel optimizer for PyTorch models. Given any `torch.nn.Module`, it parses the model into a computational graph, applies classical compiler passes, consults an **NVIDIA Nemotron Super** LLM agent (running locally via Ollama — no API key required) for advanced analysis, generates hardware-specific GPU kernels, verifies correctness, and benchmarks the result — all automatically.

**Supported hardware:** Apple Silicon M-series (Metal), NVIDIA GPUs (Triton), CPU fallback.

---

## Why Was It Built?

Manual GPU kernel optimization is time-consuming, hardware-specific, and error-prone. KERNELMIND automates the full workflow so that:

- Research teams can maximize performance without CUDA/Metal expertise
- Production teams can deploy PyTorch models on Apple Silicon or NVIDIA hardware
- Compiler techniques + LLM reasoning together outperform either approach alone

---

## Six-Stage Compilation Pipeline

```
PyTorch Model -> Parse -> Graph Optimize -> LLM Agent -> Kernel Generate -> Compile -> Verify & Benchmark
```

| Stage               | File                              | What happens                                                 |
|---------------------|-----------------------------------|--------------------------------------------------------------|
| 1. Parse            | `core/model_parser.py`            | FX-traces `nn.Module` into a DAG with shape/dtype inference  |
| 2. Optimize         | `core/optimizer.py`               | Constant fold → DCE → CSE → operator fusion → layout opt     |
| 3. LLM Agent        | `agent/ollama_optimizer.py`       | Serializes graph; gets fusion/layout suggestions from Nemotron Super |
| 4. Decide           | `agent/decision_engine.py`.       | Filters suggestions by hardware limits & numerical safety    |
| 5. Generate         | `kernels/generator.py`            | Emits Metal (MSL) or Triton kernel source from decisions     |
| 6. Compile & Verify | `metal_backend.py`, `verifier.py` | Compiles, checks numerics, benchmarks                        |

---

## Module Reference

### `core/` — Graph Engine
| File | Role |
|------|------|
| `graph.py` | `Tensor`, `Operation`, `Node`, `ComputationalGraph` DAG data structures |
| `model_parser.py` | PyTorch FX tracing → `ComputationalGraph` with shape/dtype inference |
| `optimizer.py` | Constant folding, DCE, CSE, fusion, layout optimization passes |
| `constants.py` | Op types, data types, fusion patterns, hardware bandwidth limits |

### `agent/` — LLM Intelligence
| File | Role |
|------|------|
| `ollama_optimizer.py` | Sends graph to local Nemotron Super via Ollama; receives structured optimization suggestions |
| `decision_engine.py` | Scores and filters suggestions against hardware constraints and risk |
| `ollama_errors.py` | Custom exception hierarchy with how-to-fix hints |
| `ollama_config.py` (in `config/`) | Connection settings, env-var overrides, health-check |

### `kernels/` — Code Generation
| File | Role |
|------|------|
| `generator.py` | Template engine for Metal (MSL) and Triton kernel source |
| `metal_backend.py` | Compiles MSL via `xcrun metal`; manages Metal command queues; CPU fallback |
| `triton_backend.py` | Compiles and executes Triton kernels on NVIDIA; CPU fallback |

### `benchmarks/` — Verification & Performance
| File | Role |
|------|------|
| `runner.py` | Measures latency (ms) and throughput (samples/sec) across batch sizes |
| `verifier.py` | Compares optimized outputs to PyTorch reference within numerical tolerance |
| `metrics.py` | Aggregates timing/memory/speedup; exports JSON/CSV |

### `utils/` — Infrastructure
| File | Role |
|------|------|
| `hardware.py` | Detects Apple Silicon / NVIDIA / CPU; reports GPU cores and bandwidth |
| `memory.py` | Context manager tracking peak memory during model runs |
| `helpers.py` | FLOP counting, efficiency metrics, formatting |
| `logger.py` | Structured logging to console + `kernelmind.log` |

### `examples/` — Reference Implementations
| File | Demonstrates |
|------|-------------|
| `simple_linear.py` | End-to-end MLP optimization — recommended starting point |
| `resnet_optimize.py` | Conv + BatchNorm fusion on ResNet-18 |
| `transformer_fusion.py` | QKV projection + attention softmax fusion |

---

## Technology Stack

| Component     | Technology                | Significance                                            |
|---------------|---------------------------|---------------------------------------------------------|
| Language.     | Python 3.10+              | PyTorch and Triton ecosystem compatibility              |
| Graph tracing | PyTorch FX                | Symbolic tracing without model modifications            |
| LLM           | NVIDIA Nemotron Super (Ollama)    | Local inference — no API costs, no API key required             |
| Apple GPU     | Metal Performance Shaders | Native Apple Silicon acceleration                       |
| NVIDIA GPU    | OpenAI Triton             | Python-level kernels without raw CUDA                   |
| Numerics      | NumPy                     | CPU reference for correctness verification              |
| Build         | setuptools editable       | `pip install -e .` for development iteration            |

---

## Optimization Levels

| Level        | Passes Applied                           | Best For                        |
|--------------|------------------------------------------|---------------------------------|
| `NONE`       | None — pure PyTorch                      | Debugging, correctness baseline |
| `LOW`        | Constant fold + DCE                      | Minimal-risk first pass         |
| `MEDIUM`     | + Standard fusions (Linear+Act, Conv+BN) | Most feedforward networks       |
| `HIGH`       | + Complex fusion + threadgroup tiling    | CNNs, RNNs                      |
| `AGGRESSIVE` | + LLM-guided fusion + memory tiling      | Research, novel architectures   |

---

## Project Statistics

| Metric               | Value   |
|----------------------|---------|
| Python files         | 27      |
| Lines of Python code | 7,000+  |
| Supported operations | 30+     |
| Fusion patterns      | 6       |
| Optimization levels  | 5       |
| Included examples    | 3       |
| Documentation        | ~110 KB |



---

## Citation

```bibtex
@software{kernelmind2024,
  title  = {KERNELMIND: Agentic ML Compiler and GPU Kernel Optimizer},
  author = {ML Systems Engineer},
  year   = {2024},
  url    = {https://github.com/yourusername/kernelmind}
}
```
