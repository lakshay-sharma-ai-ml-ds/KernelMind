# KERNELMIND — Advanced Setup & Optimization

> For users who want hardware-specific tuning, custom pipelines, or production-grade deployment.

---

## Optimization Level Strategy

| Level | Value | Passes | When to use |
|-------|-------|--------|------------|
| `NONE` | 0 | None | Debugging baseline |
| `LOW` | 1 | Constant fold + DCE | Minimal risk |
| `MEDIUM` | 2 | + Operator fusion | Most networks |
| `HIGH` | 3 | + Complex fusion + tiling | CNNs, RNNs |
| `AGGRESSIVE` | 4 | + LLM-guided fusion | Research models |

**Rule:** Always progress LOW → validate → MEDIUM → validate → HIGH. Never skip validation steps.

---

## Hardware-Specific Tuning

### Apple Silicon (M-series)

```python
# kernelmind/config.py
config.MEMORY_BANDWIDTH_GBPS             = 100    # M1:68 | M2:100 | M3:150 | M2 Ultra:800
config.L2_CACHE_SIZE_BYTES               = 24_000_000
config.METAL_THREADGROUP_WIDTH           = 32
config.METAL_THREADGROUP_HEIGHT          = 32
config.METAL_MAX_THREADS_PER_THREADGROUP = 1024
config.METAL_MEMORY_ALIGNMENT            = 256    # bytes — ensures coalesced access
config.ENABLE_AMX_ACCELERATION           = True   # Apple Matrix coprocessor (M1+)
config.PREFER_FP16                       = True
```

Validate kernel syntax: `xcrun metal -c kernel.metal -o /dev/null`

### NVIDIA GPUs

```python
config.TRITON_NUM_WARPS          = 4
config.TRITON_NUM_STAGES         = 4
config.TRITON_BLOCK_SIZE         = 256
config.TRITON_SHARED_MEMORY_SIZE = 32768   # 32 KB shared memory
config.ENABLE_TENSOR_CORES       = True

import torch
if torch.cuda.get_device_properties(0).major >= 8:  # Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
```

---

## Optimization by Model Type

### Large Language Models

```python
config.OPTIMIZATION_LEVEL          = OptimizationLevel.AGGRESSIVE
config.ENABLE_ATTENTION_FUSION      = True
config.ENABLE_KV_CACHE_OPTIMIZATION = True
config.LLM_TEMPERATURE              = 0.1
config.BENCHMARK_BATCH_SIZES        = [1, 2, 4, 8, 16, 32]
```

Key targets: QKV projection fusion · softmax fusion · KV-cache bandwidth reduction.

### Computer Vision (CNNs & ViTs)

```python
config.OPTIMIZATION_LEVEL         = OptimizationLevel.HIGH
config.ENABLE_CONVOLUTION_FUSION   = True
config.ENABLE_BATCHNORM_FUSION     = True
config.ENABLE_POOLING_OPTIMIZATION = True
config.MEMORY_FORMAT_PREFERENCE    = "NHWC"   # better for Apple Silicon & mobile
config.ENABLE_QKV_FUSION           = True     # ViT only
config.ENABLE_MLP_FUSION           = True     # ViT only
```

Key targets: Conv+BN+Act single-pass · NHWC memory layout · patch embedding optimization.

### Graph Neural Networks

```python
config.OPTIMIZATION_LEVEL           = OptimizationLevel.MEDIUM   # start conservative
config.ENABLE_SPARSE_OPTIMIZATIONS  = True
config.ENABLE_NEIGHBOR_SAMPLING_OPT = True
```

Key targets: Sparse-dense message passing · scatter/gather (may need custom kernels).

---

## Custom Kernel Templates

### Metal (Apple Silicon)

```cpp
#include <metal_stdlib>
using namespace metal;

kernel void fused_linear_relu(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    threadgroup float shmem[256];
    shmem[gid % 256] = (gid < n) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid < n)
        output[gid] = max(0.0f, shmem[gid % 256] * weight[gid]);
}
```

Rules: threadgroup ≤ 1024 threads · shared memory ≤ 32 KB · 256-byte aligned buffers.

### Triton (NVIDIA)

```python
import triton, triton.language as tl

@triton.jit
def fused_kernel(x_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    w = tl.load(w_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(0.0, x * w), mask=mask)
```

Tune: `BLOCK_SIZE` (start 128–256) · `num_warps` · `num_stages` for memory-bound kernels.

---

## Framework Integrations

**PyTorch Lightning** — optimize once during `setup()`:
```python
class OptimizedLitModel(LightningModule):
    def setup(self, stage=None):
        if self.trainer.is_global_zero:
            self.model = optimize_model(self.model, torch.randn(1, *self.input_shape), optimization_level="HIGH")
    def forward(self, x): return self.model(x)
```

**Hugging Face Transformers** — compile attention + FFN kernels for inference:
```python
def optimize_hf_for_inference(model, sample_ids, backend="metal"):
    gen, hw = KernelGenerator(), MetalBackend()
    kernels = {**gen.generate_attention_kernels(model, sample_ids, backend=backend),
               **gen.generate_feedforward_kernels(model, sample_ids, backend=backend)}
    return {n: c for n, c in kernels.items() if hw.compile_kernel(n, c)}
```

---

## Production Best Practices

| Area | Recommendation |
|------|--------------|
| Optimization | LOW → MEDIUM → HIGH with validation at each step |
| Warmup | Run N startup inferences before serving (kernel compilation overhead) |
| Fallback | Always maintain a tested CPU fallback path |
| Versioning | Pin KERNELMIND version in `requirements.txt` |
| Monitoring | Track P50/P95/P99 latency; alert on >5% regression |
| Rollback | Feature-flag between baseline and optimized model |

```python
import os
def get_model():
    return load_optimized() if os.getenv("USE_OPTIMIZED", "true") == "true" else load_baseline()
```

---

## Troubleshooting Advanced Issues

| Issue | Diagnostic | Fix |
|-------|-----------|-----|
| Kernel compile fail | `xcrun metal -c k.metal` | Simplify; check threadgroup ≤ 1024 |
| Numerical errors | `torch.allclose(ref, opt)` | FP32 accumulation; check reduction order |
| No speedup | Roofline: compute vs memory bound? | Increase arithmetic intensity; fuse more ops |
| Low GPU occupancy | Profiler shows <50% | Adjust threadgroup size; reduce register use |
| NaN / Inf outputs | FP16 overflow | Accumulate in FP32; clip activations |

```bash
# Debug: enable verbose logs + CPU profiler
export KERNELMIND_LOG_LEVEL=DEBUG && tail -f kernelmind.log
python3 -m cProfile -o prof.out script.py && python3 -c "import pstats; pstats.Stats('prof.out').sort_stats('cumtime').print_stats(15)"
```
