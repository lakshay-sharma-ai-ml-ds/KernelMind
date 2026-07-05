# KERNELMIND — Advanced Setup & Optimization

> Hardware-specific tuning, custom pipelines, and production deployment.

---

## Optimization Level Strategy

| Level | Passes | When to use |
|---|---|---|
| `NONE` | None | Debugging baseline |
| `LOW` | Constant fold + DCE | Minimal risk |
| `MEDIUM` | + Operator fusion | Most networks |
| `HIGH` | + Complex fusion + tiling | CNNs, RNNs |
| `AGGRESSIVE` | + Nemotron Super-guided fusion | Research models |

**Rule:** Always progress LOW → validate → MEDIUM → validate → HIGH.

---

## Hardware-Specific Tuning

### Apple Silicon

```python
config.MEMORY_BANDWIDTH_GBPS             = 100    # M1:68 | M2:100 | M3:150
config.METAL_THREADGROUP_WIDTH           = 32
config.METAL_THREADGROUP_HEIGHT          = 32
config.METAL_MAX_THREADS_PER_THREADGROUP = 1024
config.METAL_MEMORY_ALIGNMENT            = 256
config.ENABLE_AMX_ACCELERATION           = True
config.PREFER_FP16                       = True
```

### NVIDIA GPUs

```python
config.TRITON_NUM_WARPS          = 4
config.TRITON_BLOCK_SIZE         = 256
config.TRITON_SHARED_MEMORY_SIZE = 32768
config.ENABLE_TENSOR_CORES       = True
```

---

## Optimization by Model Type

**LLMs (AGGRESSIVE level):**
```python
config.OPTIMIZATION_LEVEL          = OptimizationLevel.AGGRESSIVE
config.ENABLE_ATTENTION_FUSION      = True
config.ENABLE_KV_CACHE_OPTIMIZATION = True
# export OLLAMA_TEMPERATURE=0.1 && export OLLAMA_MAX_TOKENS=4000
```

**CNNs / ViTs (HIGH level):**
```python
config.OPTIMIZATION_LEVEL       = OptimizationLevel.HIGH
config.ENABLE_CONVOLUTION_FUSION = True
config.ENABLE_BATCHNORM_FUSION   = True
config.MEMORY_FORMAT_PREFERENCE  = "NHWC"
```

**GNNs (MEDIUM level):**
```python
config.OPTIMIZATION_LEVEL          = OptimizationLevel.MEDIUM
config.ENABLE_SPARSE_OPTIMIZATIONS = True
```

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
    if (gid < n) output[gid] = max(0.0f, shmem[gid % 256] * weight[gid]);
}
```

*Rules: threadgroup ≤ 1024 threads · shared memory ≤ 32 KB · 256-byte aligned buffers.*

### Triton (NVIDIA)

```python
import triton, triton.language as tl
@triton.jit
def fused_kernel(x_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.maximum(0.0, tl.load(x_ptr+offs, mask=mask)
             * tl.load(w_ptr+offs, mask=mask)), mask=mask)
```
---

## Framework Integrations

**PyTorch Lightning:**
```python
class OptimizedLitModel(LightningModule):
    def setup(self, stage=None):
        if self.trainer.is_global_zero:
            self.model = optimize_model(self.model, torch.randn(1, *self.input_shape))
```

**Hugging Face Transformers:**
```python
# OptimizationAgent uses Nemotron Super locally — no API key needed
def optimize_hf_for_inference(model, sample_ids, backend="metal"):
    gen, hw = KernelGenerator(), MetalBackend()
    kernels = {**gen.generate_attention_kernels(model, sample_ids, backend=backend),
               **gen.generate_feedforward_kernels(model, sample_ids, backend=backend)}
    return {n: c for n, c in kernels.items() if hw.compile_kernel(n, c)}
```

---

## Production Best Practices

| Area | Recommendation |
|---|---|
| Optimization | LOW → MEDIUM → HIGH with validation at each step |
| Warmup | Run N startup inferences before serving |
| Fallback | Maintain a tested CPU fallback path |
| Monitoring | Alert on >5% P99 latency regression |

## Troubleshooting

| Issue | Fix |
|---|---|
| Kernel compile fail | `xcrun metal -c k.metal` · check threadgroup ≤ 1024 |
| Numerical errors / NaN | FP32 accumulation; check reduction order; clip activations |
| No speedup | Roofline: compute vs memory bound? Fuse more ops |

```bash
export KERNELMIND_LOG_LEVEL=DEBUG && tail -f kernelmind.log
```
