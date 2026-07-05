# KERNELMIND: Agentic ML Compiler & GPU Kernel Optimizer

Analyzes PyTorch models, applies compiler optimizations, generates Metal/Triton GPU kernels via a Claude LLM agent, and benchmarks them — fully automated.

**Optimized for:** Apple Silicon M-series (Metal) and NVIDIA GPUs (Triton)

---

## System Requirements

|         | Minimum     | Notes                                                                |
|---------|-------------|----------------------------------------------------------------------|
| Python  | 3.10+       | Tested up to 3.13                                                    |
| OS      | macOS 12.3+ | Required for Metal backend; NVIDIA path works on Linux               |
| Disk    | ~4 GB       | For packages + venv                                                  |
| API key | Anthropic   | Free tier at [console.anthropic.com](https://console.anthropic.com/) |

---

## Quick Start (5 minutes)

```bash                     
cd ~/kernelmind
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt && pip install -e .
echo "ANTHROPIC_API_KEY=sk-ant-YOUR-KEY-HERE" > .env
python3 quickstart.py   # verify setup
python3 main.py         # launch interactive menu
```

-------

## Installation

### 1. Xcode Command Line Tools (macOS only)
```bash
xcode-select --install    # provides xcrun metal compiler
```

### 2. Python Environment
```bash
python3 --version         # must be 3.10+
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .          # editable — imports work project-wide without reinstalling
```

### 4. Configure API Key

**.env file (recommended)**
```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env
```

**Shell export (CI / production)**
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key
echo 'export ANTHROPIC_API_KEY=sk-ant-your-key' >> ~/.zshrc && source ~/.zshrc
```

---

## Project Structure

```
kernelmind/
├── README.md              ← This file
├── PROJECT_SUMMARY.md     ← Architecture & component reference
├── ADVANCED_SETUP.md      ← Hardware tuning & custom pipelines
├── requirements.txt / setup.py
├── main.py                ← Interactive menu (5 options)
├── quickstart.py          ← Setup verifier
└── kernelmind/
    ├── config.py          ← All tunables
    ├── core/              ← Graph parsing & optimization
    ├── kernels/           ← Metal + Triton backends
    ├── agent/             ← Claude LLM integration
    ├── benchmarks/        ← Benchmarking & verification
    ├── utils/             ← Hardware detection, logging
    └── examples/          ← simple_linear, resnet, transformer
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

agent = OptimizationAgent()
agent.optimize(opt_graph)

kernels = KernelGenerator().generate_kernels(opt_graph, backend="metal")
for name, code in kernels.items():
    if MetalBackend().compile_kernel(name, code):
        print(f"✓ {name}")
```

**Set optimization level:**
```python
from kernelmind.config import config, OptimizationLevel
config.OPTIMIZATION_LEVEL = OptimizationLevel.HIGH  # NONE/LOW/MEDIUM/HIGH/AGGRESSIVE
```

---

## Running Examples

```bash
python3 main.py                           # interactive menu
python3 examples/simple_linear.py         # MLP — start here
python3 examples/resnet_optimize.py       # CNN conv+BN fusion
python3 examples/transformer_fusion.py    # QKV + attention fusion
python3 quickstart.py                     # verify environment & hardware
```

Interactive menu options: MLP optimization · ResNet-18 · Transformer fusion · Hardware info · Run all

---

## Configuration

All settings live in `kernelmind/config.py`. Override without editing code:

```bash
export KERNELMIND_OPTIMIZATION_LEVEL=AGGRESSIVE
export KERNELMIND_LLM_MODEL=claude-3-opus-20240229
export KERNELMIND_LLM_TEMPERATURE=0.1
export KERNELMIND_BENCHMARK_WARMUP=100
export KERNELMIND_BENCHMARK_MEASURE=500
```

---

## Troubleshooting

| Error                             | Fix                                                                 |
|-----------------------------------|---------------------------------------------------------------------|
| `ModuleNotFoundError: kernelmind` | Run `pip install -e .`; confirm venv is active (`which python3`)    |
| `ANTHROPIC_API_KEY not set`       | `cat .env` to inspect; or `export ANTHROPIC_API_KEY=...`            |
| Metal/MPS not available           | Apple Silicon + macOS 12.3+ required; `pip install --upgrade torch` |
| Out of memory                     | Lower `BENCHMARK_BATCH_SIZES` in `config.py`; close other apps      |
| Inconsistent benchmarks           | Raise warmup runs; run `nice -n -20 python3 script.py`              |

```bash
tail -f kernelmind.log    # real-time log streaming
```

---

## Performance Tips

1. **Float16** — halves memory bandwidth where numerically safe
2. **Warmup runs** — first runs compile kernels; always exclude from timing
3. **Batch size sweep** — throughput improves with batch size up to memory limits
4. **Monitor GPU** — macOS Instruments or `nvtop`; target >80% utilization
5. **Apple Silicon** — GPU and CPU share RAM; close other apps before benchmarking


